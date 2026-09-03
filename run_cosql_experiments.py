import argparse
import json
import os
import sqlite3

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config
from config import Provider

from evaluation import (
    execution_accuracy,
    convert_to_dataframe,
    exact_match_sql,
    normalize_sql,
    has_top_level_order_by,
)

from llm import get_llm

from spark_nl import (
    get_spark_session,
    get_spark_sql,
    get_spark_agent,
    run_nl_query,
    process_result,
)

from utils import ensure_sqlite_jdbc_driver

from src.clause_predictor import (
    predict_categories_schema_aware,
)

from src.spark_toolkit.prompt import (
    build_clause_guidance,
)


COSQL_ROOT = os.path.join(
    "db",
    "cosql_dataset",
    "cosql_dataset"
)

DATABASE_ROOT = os.path.join(
    COSQL_ROOT,
    "database"
)

SAMPLE_FILE = "cosql_evaluation_sample.json"


def detect_api_error(message):

    if not message:
        return None

    message = str(message).upper()

    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        return "RATE_LIMIT"

    if (
        "UNAVAILABLE" in message
        or "503" in message
        or "HIGH DEMAND" in message
    ):
        return "SERVICE_UNAVAILABLE"

    if "INTERNAL" in message or "500" in message:
        return "SERVER_ERROR"

    if "DEADLINE_EXCEEDED" in message or "504" in message:
        return "TIMEOUT"

    if "NOT_FOUND" in message or "404" in message:
        return "MODEL_NOT_FOUND"

    if "UNAUTHENTICATED" in message or "401" in message:
        return "AUTH_ERROR"

    if "PERMISSION_DENIED" in message or "403" in message:
        return "PERMISSION_ERROR"

    return None


def detect_iteration_limit(message):

    if not message:
        return False

    message = str(message).lower()

    return (
        "iteration limit" in message
        or "stopped due to iteration" in message
    )


def dataframe_preview(df, max_rows=20):

    if df is None:
        return []

    return json.loads(
        df.head(max_rows).to_json(
            orient="records"
        )
    )


def get_database_path(db_id):

    return os.path.join(
        DATABASE_ROOT,
        db_id,
        f"{db_id}.sqlite"
    )


def get_cosql_schema(db_id):

    db_path = get_database_path(db_id)

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found: {db_path}"
        )

    schema = {}

    with sqlite3.connect(db_path) as conn:

        cursor = conn.cursor()

        tables = cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()

        for (table_name,) in tables:

            columns = cursor.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()

            schema[table_name] = [
                column[1]
                for column in columns
            ]

    return schema


def load_cosql_tables(
    spark,
    db_id
):

    db_path = os.path.abspath(
        get_database_path(db_id)
    )

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found: {db_path}"
        )

    print(
        f"--- Loading CoSQL database: "
        f"{db_path} ---"
    )

    with sqlite3.connect(db_path) as conn:

        tables = [
            row[0]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                  AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
        ]

    jdbc_url = (
        "jdbc:sqlite:"
        + db_path.replace("\\", "/")
    )

    for table_name in tables:

        df = (
            spark.read
            .format("jdbc")
            .option(
                "url",
                jdbc_url
            )
            .option(
                "dbtable",
                table_name
            )
            .option(
                "driver",
                "org.sqlite.JDBC"
            )
            .load()
        )

        df.createOrReplaceTempView(
            table_name
        )

        print(
            f" -> Registered table: "
            f"'{table_name}'"
        )


def category_signature(prediction):

    active = [
        label
        for label in [
            "JOIN",
            "GROUP_BY",
            "HAVING"
        ]
        if prediction[label]
    ]

    return (
        " + ".join(active)
        if active
        else "NONE"
    )


def run_experiment(
    mode,
    provider,
    model
):

    with open(
        SAMPLE_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        sample = json.load(f)

    output_json = (
        f"cosql_{mode}_results.json"
    )

    output_csv = (
        f"cosql_{mode}_results.csv"
    )

    checkpoint_file = (
        f"cosql_{mode}_results_checkpoint.json"
    )

    # Resume safely if execution is interrupted.
    if os.path.exists(checkpoint_file):

        with open(
            checkpoint_file,
            "r",
            encoding="utf-8"
        ) as f:
            previous_results = json.load(f)

        # Keep completed/model attempts.
        # API failures can be retried.
        results = [
            r
            for r in previous_results
            if r.get("experiment_status")
            != "API_ERROR"
        ]

    else:
        results = []

    completed_ids = {
        r["dialogue_index"]
        for r in results
    }

    remaining = [
        item
        for item in sample
        if item["dialogue_index"]
        not in completed_ids
    ]

    print("=" * 70)
    print(
        f"COSQL {mode.upper()} EXPERIMENT"
    )
    print("=" * 70)

    print(
        f"Sample size: {len(sample)}"
    )

    print(
        f"Already completed: "
        f"{len(completed_ids)}"
    )

    print(
        f"Remaining: {len(remaining)}"
    )

    jdbc_jar_path = (
        ensure_sqlite_jdbc_driver()
    )

    for item in remaining:

        config.metrics.clear()

        config.metrics.update({
            "total_time": -1,
            "spark_exec_time": -1,
            "translation_time": -1,
            "sparksql_query": None,
            "answer": None,
            "executed_queries": [],
        })

        dialogue_index = item[
            "dialogue_index"
        ]

        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["gold_sql"]

        print()
        print("=" * 70)
        print(
            f"DIALOGUE {dialogue_index}"
        )
        print("=" * 70)

        print(f"Mode: {mode}")
        print(f"Database: {db_id}")
        print(f"Question: {question}")
        print(
            f"Gold structure: "
            f"{item['gold_signature']}"
        )

        schema = get_cosql_schema(
            db_id
        )

        prediction = (
            predict_categories_schema_aware(
                question,
                schema
            )
        )

        predicted_signature = (
            category_signature(
                prediction
            )
        )

        print(
            f"Predicted structure: "
            f"{predicted_signature}"
        )

        if mode == "guided":

            prompt_suffix = (
                build_clause_guidance(
                    prediction["categories"]
                )
            )

        else:
            prompt_suffix = ""

        guidance_applied = bool(
            prompt_suffix
        )

        print(
            f"Guidance applied: "
            f"{guidance_applied}"
        )

        db_path = get_database_path(
            db_id
        )

        # Execute reference CoSQL SQL
        # in its native SQLite environment.
        try:

            with sqlite3.connect(
                db_path
            ) as conn:

                ground_truth = (
                    pd.read_sql_query(
                        gold_sql,
                        conn
                    )
                )

        except Exception as e:

            print(
                "ERROR executing gold SQL:"
            )
            print(e)
            raise

        order_sensitive = (
            has_top_level_order_by(
                gold_sql,
                dialect="sqlite"
            )
        )

        spark = get_spark_session(
            extra_configs={
                "spark.jars":
                    jdbc_jar_path,

                "spark.driver.extraClassPath":
                    jdbc_jar_path,
            }
        )

        try:

            load_cosql_tables(
                spark,
                db_id
            )

            llm = get_llm(
                provider=provider,
                model=model
            )

            spark_sql = (
                get_spark_sql()
            )

            agent = get_spark_agent(
                spark_sql,
                llm
            )

            run_nl_query(
                agent,
                question,
                llm=llm,
                prompt_suffix=prompt_suffix
            )

            json_result = (
                process_result()
            )

            agent_message = (
                config.metrics.get(
                    "answer",
                    ""
                )
            )

            api_error = (
                detect_api_error(
                    agent_message
                )
            )

            iteration_limit = (
                detect_iteration_limit(
                    agent_message
                )
            )

            inferred_df = None
            execution_acc = None

            if api_error:

                experiment_status = (
                    "API_ERROR"
                )

            elif (
                json_result.get(
                    "execution_status"
                )
                == "VALID"
            ):

                inferred_result = (
                    json_result.get(
                        "query_result"
                    )
                )

                inferred_df = (
                    convert_to_dataframe(
                        inferred_result
                    )
                )

                execution_acc = (
                    execution_accuracy(
                        ground_truth,
                        inferred_result,
                        order_sensitive=
                            order_sensitive
                    )
                )

                experiment_status = (
                    "COMPLETED"
                )

            elif (
                json_result.get(
                    "execution_status"
                )
                == "ERROR"
            ):

                execution_acc = 0.0

                experiment_status = (
                    "SQL_ERROR"
                )

            else:

                execution_acc = 0.0

                experiment_status = (
                    "MODEL_ERROR"
                )

            if (
                experiment_status
                == "API_ERROR"
            ):

                generated_sql = None
                exact_match = None

            else:

                generated_sql = (
                    json_result.get(
                        "sparksql_query"
                    )
                )

                if generated_sql:

                    exact_match = (
                        exact_match_sql(
                            gold_sql,
                            generated_sql
                        )
                    )

                else:

                    exact_match = 0.0

            if inferred_df is not None:

                generated_rows = len(
                    inferred_df
                )

                generated_columns = (
                    inferred_df.shape[1]
                )

                generated_preview = (
                    dataframe_preview(
                        inferred_df
                    )
                )

            else:

                generated_rows = None
                generated_columns = None
                generated_preview = []

            result = {

                "dialogue_index":
                    dialogue_index,

                "db_id":
                    db_id,

                "question":
                    question,

                "experiment_mode":
                    mode,

                "gold_signature":
                    item[
                        "gold_signature"
                    ],

                "predicted_signature":
                    predicted_signature,

                "predicted_JOIN":
                    bool(
                        prediction["JOIN"]
                    ),

                "predicted_GROUP_BY":
                    bool(
                        prediction[
                            "GROUP_BY"
                        ]
                    ),

                "predicted_HAVING":
                    bool(
                        prediction["HAVING"]
                    ),

                "guidance_applied":
                    guidance_applied,

                "gold_sql":
                    gold_sql,

                "generated_sql":
                    generated_sql,

                "gold_sql_normalized":
                    normalize_sql(
                        gold_sql,
                        "sqlite"
                    ),

                "generated_sql_normalized":
                    (
                        normalize_sql(
                            generated_sql,
                            "spark"
                        )
                        if generated_sql
                        else None
                    ),

                "experiment_status":
                    experiment_status,

                "api_error":
                    api_error,

                "execution_status":
                    json_result.get(
                        "execution_status"
                    ),

                "execution_accuracy":
                    execution_acc,

                "exact_match":
                    exact_match,

                "iteration_limit_reached":
                    iteration_limit,

                "ground_truth_rows":
                    len(
                        ground_truth
                    ),

                "ground_truth_columns":
                    ground_truth.shape[1],

                "ground_truth_preview":
                    dataframe_preview(
                        ground_truth
                    ),

                "generated_result_rows":
                    generated_rows,

                "generated_result_columns":
                    generated_columns,

                "generated_result_preview":
                    generated_preview,

                "total_time":
                    json_result.get(
                        "total_time"
                    ),

                "llm_requests":
                    json_result.get(
                        "llm_requests"
                    ),

                "input_tokens":
                    json_result.get(
                        "input_tokens"
                    ),

                "output_tokens":
                    json_result.get(
                        "output_tokens"
                    ),

                "executed_queries":
                    json_result.get(
                        "executed_queries",
                        []
                    ),

                "spark_error":
                    json_result.get(
                        "spark_error"
                    ),
            }

            results.append(
                result
            )

            with open(
                checkpoint_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    results,
                    f,
                    indent=4,
                    ensure_ascii=False
                )

            print()
            print(
                f"Generated SQL: "
                f"{generated_sql}"
            )

            print(
                f"Execution Accuracy: "
                f"{execution_acc}"
            )

            print(
                f"Exact Match: "
                f"{exact_match}"
            )

        finally:

            spark.stop()

    # Save final files
    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8"
    )

    valid = df[
        df["experiment_status"]
        != "API_ERROR"
    ]

    print()
    print("=" * 70)
    print(
        f"COSQL {mode.upper()} SUMMARY"
    )
    print("=" * 70)

    print(
        f"Queries: {len(df)}"
    )

    print(
        f"Valid attempts: "
        f"{len(valid)}"
    )

    print(
        f"API failures: "
        f"{len(df) - len(valid)}"
    )

    if len(valid) > 0:

        print(
            f"EA: "
            f"{valid['execution_accuracy'].mean():.2%}"
        )

        print(
            f"Normalized EM: "
            f"{valid['exact_match'].mean():.2%}"
        )

        print(
            f"Total time: "
            f"{valid['total_time'].sum():.1f} s"
        )

        print(
            f"LLM requests: "
            f"{valid['llm_requests'].sum()}"
        )

        print(
            f"Input tokens: "
            f"{valid['input_tokens'].sum()}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "baseline",
            "guided"
        ]
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=Provider.GOOGLE.value
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.5-flash-lite"
    )

    args = parser.parse_args()

    run_experiment(
        mode=args.mode,
        provider=args.provider,
        model=args.model
    )