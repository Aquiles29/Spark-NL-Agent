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
    normalize_sql
)
from llm import get_llm
from load_db import load_tables
from spark_nl import (
    get_spark_session, get_spark_sql, get_spark_agent,
    run_nl_query, process_result, print_results, AgentMonitoringCallback
)
from utils import ensure_sqlite_jdbc_driver, pretty_print_result
import argparse


# Configuration
BENCHMARK_SPEC_FILE = "db/bird-1/dev.json"
MAX_QUERIES = 5

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

def main(provider, force_thoughts=False, model=None):

    if force_thoughts:
        config.FORCE_THOUGHT_GENERATION = True

    with open(BENCHMARK_SPEC_FILE, "r", encoding="utf-8") as f:
        benchmark_spec = json.load(f)

    benchmark_subset = benchmark_spec[1:2]

    results = []

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

        ground_truth = spark.sql(golden_query).toPandas()

        agent = get_spark_agent(spark_sql, llm)
        run_nl_query(agent, nl_query, llm)

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
                inferred_result
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
            "baseline_bird_results_checkpoint.json",
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
        "baseline_bird_results.csv",
        index=False,
        encoding="utf-8"
    )

    df_results.to_json(
        "baseline_bird_results.json",
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
    print("BASELINE SUMMARY")
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
    args = parser.parse_args()
    provider = args.provider
    main(provider, force_thoughts=args.force_thoughts, model=args.model)
