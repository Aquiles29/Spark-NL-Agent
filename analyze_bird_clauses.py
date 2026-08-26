#!/usr/bin/env python3

import json
import pandas as pd

from src.clause_categorization import categorize_sql


BENCHMARK_FILE = "db/bird-1/dev.json"

OUTPUT_CSV = "bird_clause_categories.csv"
OUTPUT_JSON = "bird_clause_categories.json"


def main():

    # Load BIRD development set
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    results = []

    for item in benchmark:

        sql = item["SQL"]

        category_result = categorize_sql(sql)

        categories = category_result["categories"]

        # Useful label for combinations
        if category_result["parse_error"]:
            category_signature = "PARSE_ERROR"
        elif categories:
            category_signature = "+".join(categories)
        else:
            category_signature = "NONE"

        results.append({
            "question_id": item["question_id"],
            "db_id": item["db_id"],
            "question": item["question"],
            "gold_sql": sql,

            "JOIN": category_result["JOIN"],
            "GROUP_BY": category_result["GROUP_BY"],
            "HAVING": category_result["HAVING"],

            "categories": categories,
            "category_signature": category_signature,

            "parse_error": category_result["parse_error"],
        })

    df = pd.DataFrame(results)

    # Save CSV
    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8"
    )

    # Save full JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    # -----------------------------------------
    # Summary
    # -----------------------------------------

    total = len(df)

    print("\n" + "=" * 60)
    print("BIRD CLAUSE ANALYSIS")
    print("=" * 60)

    print(f"Total queries: {total}")

    print("\nIndividual clauses:")

    for clause in ["JOIN", "GROUP_BY", "HAVING"]:
        count = int(df[clause].sum())
        percentage = count / total * 100

        print(
            f"{clause:10s}: "
            f"{count:4d} "
            f"({percentage:.2f}%)"
        )

    print("\nClause combinations:")

    combination_counts = (
        df["category_signature"]
        .value_counts()
    )

    for combination, count in combination_counts.items():

        percentage = count / total * 100

        print(
            f"{combination:30s}: "
            f"{count:4d} "
            f"({percentage:.2f}%)"
        )

    # Parse failures
    parse_errors = df["parse_error"].notna().sum()

    print("\nParsing:")
    print(f"Successfully parsed: {total - parse_errors}")
    print(f"Parse errors:        {parse_errors}")

    print("\nSaved:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_JSON}")


if __name__ == "__main__":
    main()