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
from evaluation import execution_accuracy
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

def main(provider, force_thoughts=False, model=None):

    if force_thoughts:
        config.FORCE_THOUGHT_GENERATION = True

    with open(BENCHMARK_SPEC_FILE, "r", encoding="utf-8") as f:
        benchmark_spec = json.load(f)

    benchmark_subset = benchmark_spec[:MAX_QUERIES]

    results = []

    jdbc_jar_path = ensure_sqlite_jdbc_driver()

    for item in benchmark_subset:

        config.metrics.clear()
        config.metrics.update({
            "total_time": -1,
            "spark_exec_time": -1,
            "translation_time": -1,
            "sparksql_query": None,
            "answer": None
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

        if json_result.get("execution_status") == "VALID":
            inferred_result = json_result.get("query_result")

            execution_acc = execution_accuracy(
                ground_truth,
                inferred_result
            )

        results.append({
            "question_id": question_id,
            "db_id": db_name,
            "question": nl_query,
            "gold_sql": golden_query,
            "generated_sql": json_result.get("sparksql_query"),
            "execution_status": json_result.get("execution_status"),
            "execution_accuracy": execution_acc,
            "total_time": json_result.get("total_time"),
            "llm_requests": json_result.get("llm_requests"),
            "input_tokens": json_result.get("input_tokens"),
            "output_tokens": json_result.get("output_tokens"),
            "spark_error": json_result.get("spark_error"),
        })

        print(f"\nGenerated SQL: {json_result.get('sparksql_query')}")
        print(f"Execution Accuracy: {execution_acc:.0%}")

        spark.stop()

    df_results = pd.DataFrame(results)

    df_results.to_csv(
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

    overall_ea = df_results["execution_accuracy"].mean()

    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"Queries evaluated: {len(df_results)}")
    print(f"Execution Accuracy: {overall_ea:.2%}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test NL to SparkSQL agent.")
    parser.add_argument("--provider", type=str, default=Provider.GOOGLE.value, help="LLM provider (default: google)")
    parser.add_argument("--model", type=str, help="Specific model name (e.g., o1, o3-mini, gpt-4)")
    parser.add_argument("--force-thoughts", action="store_true", help="Force text thought generation before tool calls")
    args = parser.parse_args()
    provider = args.provider
    main(provider, force_thoughts=args.force_thoughts, model=args.model)
