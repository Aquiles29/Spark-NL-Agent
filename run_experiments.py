#!/usr/bin/env python3
"""
Test script for natural language to SparkSQL conversion.

Usage:
    python test_nl_query.py
    python test_nl_query.py --force-thoughts
    python test_nl_query.py --provider openai --model o3-mini
    python test_nl_query.py --provider openai --model o3-mini --force-thoughts

Requirements:
    - GOOGLE_API_KEY environment variable (or .env file)
    - OPENAI_API_KEY environment variable (for OpenAI models)
    - Database files in db/bird-1/
"""

import sys
import os
import json
import sqlite3
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config
from config import Provider
from evaluation import (
    execution_accuracy,
    convert_to_dataframe,
    exact_match_sql,
    normalize_sql,
    has_top_level_order_by
)
from llm import get_llm
from load_db import load_tables
from spark_nl import (
    get_spark_session, get_spark_sql, get_spark_agent,
    run_nl_query, process_result, print_results, AgentMonitoringCallback
)
from utils import ensure_sqlite_jdbc_driver, pretty_print_result
from src.clause_predictor import (
    predict_categories_schema_aware,
)

from src.schema_utils import (
    get_database_schema,
)

from src.spark_toolkit.prompt import (
    build_clause_guidance,
)
import argparse


# Configuration
BENCHMARK_SPEC_FILE = "db/bird-1/dev.json"
MAX_QUERIES = 5
COMPARISON_QUERY_IDS = {
    78,    # GROUP_BY
    251,   # JOIN + GROUP_BY
    729,   # JOIN
    720,   # JOIN + GROUP_BY + HAVING
    1325,  # NONE
    1444,  # GROUP_BY + HAVING
}

def dataframe_preview(df, max_rows=20):
    """
    Convert the first rows of a DataFrame into JSON-serializable records.
    Used only for debugging/analysis, not for evaluation.
    """
    if df is None:
        return []

    return json.loads(
        df.head(max_rows).to_json(orient="records")
    )

def detect_api_error(message):
    if not message:
        return None

    message = str(message).upper()

    # Rate / quota limits
    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        return "RATE_LIMIT"

    # Temporary provider/server availability problems
    if (
        "UNAVAILABLE" in message
        or "503" in message
        or "HIGH DEMAND" in message
    ):
        return "SERVICE_UNAVAILABLE"

    # Temporary server-side failures
    if "INTERNAL" in message or "500" in message:
        return "SERVER_ERROR"

    # Timeout / gateway failures
    if "DEADLINE_EXCEEDED" in message or "504" in message:
        return "TIMEOUT"

    # Invalid/retired model
    if "NOT_FOUND" in message or "404" in message:
        return "MODEL_NOT_FOUND"

    # Authentication
    if "UNAUTHENTICATED" in message or "401" in message:
        return "AUTH_ERROR"

    # Permissions
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

def recalculate_results(experiment_mode):
    """
    Recalculate baseline Execution Accuracy offline.

    No LLM is used:
    - Gold SQL is executed in the original SQLite database.
    - Generated SQL already stored in the baseline results is
        executed again in Spark.
    - EA is recalculated using the corrected order handling.
    """

    input_json = f"{experiment_mode}_bird_results.json"
    output_csv = f"{experiment_mode}_bird_results.csv"
    checkpoint_file = f"{experiment_mode}_bird_results_checkpoint.json"

    if not os.path.exists(input_json):
        raise FileNotFoundError(
            f"Could not find {input_json}"
        )

    with open(
        input_json,
        "r",
        encoding="utf-8"
    ) as f:
        results = json.load(f)

    print(
        f"Loaded {len(results)} {experiment_mode} results."
    )
    print(
        "OFFLINE recalculation only -- no LLM will be called."
    )

    jdbc_jar_path = ensure_sqlite_jdbc_driver()

    changed_ids = []

    # Process one database at a time so Spark does not
    # need to restart for every individual query.
    database_names = sorted({
        result["db_id"]
        for result in results
        if result.get("experiment_status") != "API_ERROR"
    })

    for db_name in database_names:

        print("\n" + "=" * 60)
        print(f"Database: {db_name}")
        print("=" * 60)

        spark = get_spark_session(
            extra_configs={
                "spark.jars": jdbc_jar_path,
                "spark.driver.extraClassPath": jdbc_jar_path,
            }
        )

        load_tables(
            spark,
            db_name
        )

        db_path = os.path.join(
            "db",
            "bird-1",
            db_name,
            f"{db_name}.sqlite"
        )

        for result in results:

            if result["db_id"] != db_name:
                continue

            question_id = result["question_id"]

            # API failures are not model attempts and remain excluded.
            if result.get("experiment_status") == "API_ERROR":
                result["execution_accuracy"] = None
                continue

            gold_sql = result.get("gold_sql")
            generated_sql = result.get("generated_sql")

            old_ea = result.get("execution_accuracy")

            # No executable generated SQL = genuine model failure.
            if not generated_sql:
                result["execution_accuracy"] = 0.0
                continue

            try:
                # Gold result in native BIRD SQLite dialect
                with sqlite3.connect(db_path) as conn:
                    ground_truth = pd.read_sql_query(
                        gold_sql,
                        conn
                    )

                # Generated result in Spark dialect
                generated_df = spark.sql(
                    generated_sql
                ).toPandas()

                order_sensitive = has_top_level_order_by(
                    gold_sql,
                    dialect="sqlite"
                )

                new_ea = execution_accuracy(
                    ground_truth,
                    generated_df,
                    order_sensitive=order_sensitive
                )

                result["execution_accuracy"] = new_ea

                # Keep result metadata consistent
                result["ground_truth_rows"] = len(
                    ground_truth
                )
                result["ground_truth_columns"] = (
                    ground_truth.shape[1]
                )

                result["generated_result_rows"] = len(
                    generated_df
                )
                result["generated_result_columns"] = (
                    generated_df.shape[1]
                )

                if old_ea != new_ea:
                    changed_ids.append(
                        (
                            question_id,
                            old_ea,
                            new_ea
                        )
                    )

                print(
                    f"Query {question_id}: "
                    f"EA {old_ea} -> {new_ea}"
                )

            except Exception as e:

                # If the stored generated SQL cannot execute,
                # it remains a genuine Text-to-SQL failure.
                result["execution_accuracy"] = 0.0

                print(
                    f"Query {question_id}: "
                    f"EA = 0.0 "
                    f"(generated SQL failed: {e})"
                )

        spark.stop()

    # Save corrected JSON
    with open(
        input_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    # Keep checkpoint consistent with corrected results
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

    # Regenerate CSV
    df_results = pd.DataFrame(results)

    csv_columns = [
        "question_id",
        "db_id",
        "question",
        "experiment_mode",
        "predicted_JOIN",
        "predicted_GROUP_BY",
        "predicted_HAVING",
        "guidance_applied",
        "gold_sql",
        "generated_sql",
        "experiment_status",
        "api_error",
        "execution_status",
        "execution_accuracy",
        "exact_match",
        "ground_truth_rows",
        "generated_result_rows",
        "total_time",
        "llm_requests",
        "input_tokens",
        "output_tokens",
        "spark_error"
    ]

    df_results[csv_columns].to_csv(
        output_csv,
        index=False,
        encoding="utf-8"
    )

    # Final metrics exclude API/infrastructure failures
    completed_attempts = df_results[
        df_results["experiment_status"] != "API_ERROR"
    ]

    overall_ea = (
        completed_attempts["execution_accuracy"].mean()
    )

    overall_em = (
        completed_attempts["exact_match"].mean()
    )

    print("\n" + "=" * 60)
    print(f"RECALCULATED {experiment_mode.upper()} SUMMARY")
    print("=" * 60)

    print(
        f"Valid benchmark attempts: "
        f"{len(completed_attempts)}"
    )

    print(
        f"Execution Accuracy: "
        f"{overall_ea:.2%}"
    )

    print(
        f"Exact Match:        "
        f"{overall_em:.2%}"
    )

    print(
        f"EA values changed: "
        f"{len(changed_ids)}"
    )

    for question_id, old_ea, new_ea in changed_ids:
        print(
            f"  Query {question_id}: "
            f"{old_ea} -> {new_ea}"
        )

def main(provider, force_thoughts=False, model=None):

    if force_thoughts:
        config.FORCE_THOUGHT_GENERATION = True

    with open(BENCHMARK_SPEC_FILE, "r", encoding="utf-8") as f:
        benchmark_spec = json.load(f)

    # Load the fixed stratified evaluation sample
    with open(
        "bird_evaluation_sample.json",
        "r",
        encoding="utf-8"
    ) as f:
        evaluation_sample = json.load(f)

    # Recover the original BIRD records from dev.json.
    # This guarantees that we use the original question and gold SQL.
    benchmark_by_id = {
        item["question_id"]: item
        for item in benchmark_spec
    }

    benchmark_subset = [
        benchmark_by_id[item["question_id"]]
        for item in evaluation_sample
        if item["question_id"] in benchmark_by_id
    ]


    # First verify that all 26 sampled questions exist in BIRD dev.json
    if len(benchmark_subset) != len(evaluation_sample):
        raise RuntimeError(
            "Some evaluation-sample questions could not be found in BIRD dev.json."
        )


    print(
        f"Evaluation sample loaded: "
        f"{len(benchmark_subset)} queries"
    )


    # For the paired guided comparison, use one query
    # from each structural category.
    if config.EXPERIMENT_MODE == "guided":

        benchmark_subset = [
            item
            for item in benchmark_subset
            if item["question_id"] in COMPARISON_QUERY_IDS
        ]

        print(
            f"Guided comparison subset: "
            f"{len(benchmark_subset)} queries"
        )

    output_csv = (
        f"{config.EXPERIMENT_MODE}_bird_results.csv"
    )

    output_json = (
        f"{config.EXPERIMENT_MODE}_bird_results.json"
    )

    checkpoint_file = (
        f"{config.EXPERIMENT_MODE}_bird_results_checkpoint.json"
    )

    # Resume from checkpoint if a previous run exists
    if os.path.exists(checkpoint_file):

        with open(
            checkpoint_file,
            "r",
            encoding="utf-8"
        ) as f:
            previous_results = json.load(f)

        # Keep genuine completed/model attempts.
        # API failures are removed so they can be retried.
        results = [
            result
            for result in previous_results
            if result.get("experiment_status") != "API_ERROR"
        ]

    else:
        results = []


    completed_ids = {
        result["question_id"]
        for result in results
    }


    # Do not rerun questions already completed
    benchmark_subset = [
        item
        for item in benchmark_subset
        if item["question_id"] not in completed_ids
    ]


    print(
        f"Already completed: {len(completed_ids)}"
    )

    print(
        f"Remaining queries: {len(benchmark_subset)}"
    )

    jdbc_jar_path = ensure_sqlite_jdbc_driver()

    for item in benchmark_subset:

        config.metrics.clear()
        config.metrics.update({
            "total_time": -1,
            "spark_exec_time": -1,
            "translation_time": -1,
            "sparksql_query": None,
            "answer": None,
            "executed_queries": []
        })

        question_id = item["question_id"]
        nl_query = item["question"]
        db_name = item["db_id"]
        golden_query = item["SQL"]
        order_sensitive = has_top_level_order_by(
            golden_query,
            dialect="sqlite"
        )

        # Predict clause categories using question + schema
        schema = get_database_schema(db_name)

        prediction = predict_categories_schema_aware(
            nl_query,
            schema
        )

        predicted_categories = prediction["categories"]

        # Build query-specific guidance only in guided mode
        if config.EXPERIMENT_MODE == "guided":
            prompt_suffix = build_clause_guidance(
                predicted_categories
            )
        else:
            prompt_suffix = ""

        print("\n" + "=" * 60)
        print(f"QUERY {question_id}")
        print("=" * 60)
        print(f"Database: {db_name}")
        print(f"Question: {nl_query}")

        spark = get_spark_session(extra_configs={
            "spark.jars": jdbc_jar_path,
            "spark.driver.extraClassPath": jdbc_jar_path,
        })

        load_tables(spark, db_name)

        llm = get_llm(provider=provider, model=model)
        spark_sql = get_spark_sql()

        # Execute BIRD gold SQL using its native SQLite dialect
        db_path = os.path.join(
            "db",
            "bird-1",
            db_name,
            f"{db_name}.sqlite"
        )

        with sqlite3.connect(db_path) as conn:
            ground_truth = pd.read_sql_query(
                golden_query,
                conn
            )

        agent = get_spark_agent(spark_sql, llm)
        run_nl_query(agent,nl_query,llm=llm,prompt_suffix=prompt_suffix)

        json_result = process_result()

        execution_acc = 0.0

        agent_message = config.metrics.get("answer", "")
        api_error = detect_api_error(agent_message)
        iteration_limit = detect_iteration_limit(agent_message)

        execution_acc = None
        experiment_status = None
        inferred_df = None

        if api_error:
            # Infrastructure/API failure: do NOT count as an incorrect model prediction
            experiment_status = "API_ERROR"

        elif json_result.get("execution_status") == "VALID":
            inferred_result = json_result.get("query_result")
            inferred_df = convert_to_dataframe(inferred_result)

            execution_acc = execution_accuracy(
                ground_truth,
                inferred_result,
                order_sensitive=order_sensitive
            )

            experiment_status = "COMPLETED"

        elif json_result.get("execution_status") == "ERROR":
            # The model produced SQL but Spark could not execute it.
            # This is a genuine Text-to-SQL failure.
            execution_acc = 0.0
            experiment_status = "SQL_ERROR"

        else:
            # The model failed to produce an executable query.
            execution_acc = 0.0
            experiment_status = "MODEL_ERROR"

        ground_truth_preview = dataframe_preview(ground_truth)

        ground_truth_rows = len(ground_truth)
        ground_truth_columns = ground_truth.shape[1]

        if inferred_df is not None:
            generated_result_preview = dataframe_preview(inferred_df)
            generated_result_rows = len(inferred_df)
            generated_result_columns = inferred_df.shape[1]
        else:
            generated_result_preview = []
            generated_result_rows = None
            generated_result_columns = None

        last_executed_sql = json_result.get("sparksql_query")

        if experiment_status == "API_ERROR":
            generated_sql = None
        else:
            generated_sql = last_executed_sql

        if experiment_status == "API_ERROR":
            exact_match = None

        elif generated_sql:
            exact_match = exact_match_sql(
                golden_query,
                generated_sql
            )

        else:
            exact_match = 0.0
                
        results.append({
            "question_id": question_id,
            "db_id": db_name,
            "question": nl_query,

            "experiment_mode": config.EXPERIMENT_MODE,

            "predicted_categories": predicted_categories,
            "predicted_JOIN": prediction["JOIN"],
            "predicted_GROUP_BY": prediction["GROUP_BY"],
            "predicted_HAVING": prediction["HAVING"],

            "guidance_applied": bool(prompt_suffix),

            "gold_sql": golden_query,
            "generated_sql": generated_sql,
            "last_executed_sql": last_executed_sql,

            "exact_match": exact_match,
            "gold_sql_normalized": normalize_sql(
                golden_query,
                "sqlite"
            ),
            "generated_sql_normalized": (
                normalize_sql(generated_sql, "spark")
                if generated_sql
                else None
            ),
            "executed_queries": json_result.get("executed_queries", []),
            "iteration_limit_reached": iteration_limit,

            "experiment_status": experiment_status,
            "api_error": api_error,
            "execution_status": json_result.get("execution_status"),
            "execution_accuracy": execution_acc,

            "ground_truth_rows": ground_truth_rows,
            "ground_truth_columns": ground_truth_columns,
            "ground_truth_preview": ground_truth_preview,

            "generated_result_rows": generated_result_rows,
            "generated_result_columns": generated_result_columns,
            "generated_result_preview": generated_result_preview,

            "total_time": json_result.get("total_time"),
            "llm_requests": json_result.get("llm_requests"),
            "input_tokens": json_result.get("input_tokens"),
            "output_tokens": json_result.get("output_tokens"),

            "spark_error": json_result.get("spark_error"),
            "agent_error": agent_message if experiment_status == "API_ERROR" else None
        }) 

        with open(
            checkpoint_file,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"\nGenerated SQL: {json_result.get('sparksql_query')}")
        if execution_acc is not None:
            print(f"Execution Accuracy: {execution_acc:.0%}")
        else:
            print("Execution Accuracy: N/A")

        if exact_match is not None:
            print(f"Exact Match: {exact_match:.0%}")
        else:
            print("Exact Match: N/A")
        spark.stop()

    df_results = pd.DataFrame(results)

    csv_columns = [
        "question_id",
        "db_id",
        "question",
        "experiment_mode",
        "predicted_JOIN",
        "predicted_GROUP_BY",
        "predicted_HAVING",
        "guidance_applied",
        "gold_sql",
        "generated_sql",
        "experiment_status",
        "api_error",
        "execution_status",
        "execution_accuracy",
        "exact_match",
        "ground_truth_rows",
        "generated_result_rows",
        "total_time",
        "llm_requests",
        "input_tokens",
        "output_tokens",
        "spark_error"
    ]

    df_results[csv_columns].to_csv(
        output_csv,
        index=False,
        encoding="utf-8"
    )

    df_results.to_json(
        output_json,
        orient="records",
        indent=4,
        force_ascii=False
    )

    completed_attempts = df_results[
        df_results["experiment_status"] != "API_ERROR"
    ]

    api_errors = df_results[
        df_results["experiment_status"] == "API_ERROR"
    ]

    if len(completed_attempts) > 0:
        overall_ea = completed_attempts["execution_accuracy"].mean()
        overall_em = completed_attempts["exact_match"].mean()
    else:
        overall_ea = None
        overall_em = None

    print("\n" + "=" * 60)
    print(
        f"{config.EXPERIMENT_MODE.upper()} SUMMARY"
    )
    print("=" * 60)
    print(f"Queries selected: {len(df_results)}")
    print(f"Valid benchmark attempts: {len(completed_attempts)}")
    print(f"API/infrastructure failures: {len(api_errors)}")

    if overall_ea is not None:
        print(f"Execution Accuracy: {overall_ea:.2%}")
        print(f"Exact Match:        {overall_em:.2%}")
    else:
        print("Execution Accuracy: N/A")
        print("Exact Match:        N/A")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test NL to SparkSQL agent.")
    parser.add_argument("--provider", type=str, default=Provider.GOOGLE.value, help="LLM provider (default: google)")
    parser.add_argument("--model", type=str, help="Specific model name (e.g., o1, o3-mini, gpt-4)")
    parser.add_argument("--force-thoughts", action="store_true", help="Force text thought generation before tool calls")
    parser.add_argument("--recalculate-baseline",action="store_true",help="Recalculate baseline EA offline without calling the LLM")
    parser.add_argument("--recalculate-guided",action="store_true",help="Recalculate guided EA offline without calling the LLM")
    args = parser.parse_args()

if args.recalculate_baseline:

    recalculate_results("baseline")

elif args.recalculate_guided:

    recalculate_results("guided")

else:

    provider = args.provider

    main(
        provider,
        force_thoughts=args.force_thoughts,
        model=args.model
    )