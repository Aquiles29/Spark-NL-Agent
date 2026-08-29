import os
import json
import pandas as pd

from src.clause_predictor import (
    predict_categories,
    predict_categories_schema_aware,
)

from src.schema_utils import get_database_schema


INPUT_FILE = "bird_clause_categories.json"
OUTPUT_FILE = "bird_clause_prediction_results.csv"

CLAUSES = [
    "JOIN",
    "GROUP_BY",
    "HAVING"
]


def safe_divide(a, b):
    return a / b if b else 0.0


def compute_metrics(
    df,
    clause,
    prediction_prefix
):

    gold = df[f"gold_{clause}"]
    pred = df[
        f"{prediction_prefix}_{clause}"
    ]

    tp = int(
        ((gold == True) & (pred == True)).sum()
    )

    fp = int(
        ((gold == False) & (pred == True)).sum()
    )

    fn = int(
        ((gold == True) & (pred == False)).sum()
    )

    tn = int(
        ((gold == False) & (pred == False)).sum()
    )

    precision = safe_divide(
        tp,
        tp + fp
    )

    recall = safe_divide(
        tp,
        tp + fn
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall
    )

    accuracy = safe_divide(
        tp + tn,
        tp + fp + fn + tn
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    bird_db_root = os.path.join(
        "db",
        "bird-1"
    )

    available_db_ids = set()

    for root, dirs, files in os.walk(bird_db_root):

        for filename in files:

            if filename.endswith(".sqlite"):
                available_db_ids.add(
                    os.path.splitext(filename)[0]
                )

    total_questions = len(data)

    data = [
        item
        for item in data
        if item["db_id"] in available_db_ids
    ]

    print("\nAvailable BIRD databases:")
    for db_id in sorted(available_db_ids):
        print(f"  - {db_id}")

    print(
        f"\nQuestions in full BIRD dev set: "
        f"{total_questions}"
    )

    print(
        f"Questions available locally: "
        f"{len(data)}"
    )

    results = []
    schema_cache = {}

    for item in data:

        question = item["question"]
        db_id = item["db_id"]

        # Question-only baseline
        question_prediction = predict_categories(
            question
        )

        # Load each database schema only once
        if db_id not in schema_cache:
            schema_cache[db_id] = (
                get_database_schema(db_id)
            )

        schema = schema_cache[db_id]

        # Schema-aware prediction
        schema_prediction = (
            predict_categories_schema_aware(
                question,
                schema
            )
        )

        question_signature = (
            "+".join(
                question_prediction["categories"]
            )
            if question_prediction["categories"]
            else "NONE"
        )

        schema_signature = (
            "+".join(
                schema_prediction["categories"]
            )
            if schema_prediction["categories"]
            else "NONE"
        )

        results.append({
            "question_id": item["question_id"],
            "db_id": db_id,
            "question": question,

            "gold_JOIN": item["JOIN"],
            "gold_GROUP_BY": item["GROUP_BY"],
            "gold_HAVING": item["HAVING"],

            "question_pred_JOIN":
                question_prediction["JOIN"],
            "question_pred_GROUP_BY":
                question_prediction["GROUP_BY"],
            "question_pred_HAVING":
                question_prediction["HAVING"],

            "schema_pred_JOIN":
                schema_prediction["JOIN"],
            "schema_pred_GROUP_BY":
                schema_prediction["GROUP_BY"],
            "schema_pred_HAVING":
                schema_prediction["HAVING"],

            "gold_signature":
                item["category_signature"],

            "question_pred_signature":
                question_signature,

            "schema_pred_signature":
                schema_signature,
        })

    df = pd.DataFrame(results)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    for predictor_name, prefix in [
        (
            "QUESTION-ONLY",
            "question_pred"
        ),
        (
            "SCHEMA-AWARE",
            "schema_pred"
        ),
    ]:

        print("\n" + "=" * 70)
        print(predictor_name)
        print("=" * 70)

        metric_results = {}

        for clause in CLAUSES:

            metrics = compute_metrics(
                df,
                clause,
                prefix
            )

            metric_results[clause] = metrics

            print(f"\n{clause}")
            print("-" * 40)

            print(
                f"TP: {metrics['TP']:4d}   "
                f"FP: {metrics['FP']:4d}"
            )

            print(
                f"FN: {metrics['FN']:4d}   "
                f"TN: {metrics['TN']:4d}"
            )

            print(
                f"Precision: "
                f"{metrics['precision']:.3f}"
            )

            print(
                f"Recall:    "
                f"{metrics['recall']:.3f}"
            )

            print(
                f"F1:        "
                f"{metrics['f1']:.3f}"
            )

            print(
                f"Accuracy:  "
                f"{metrics['accuracy']:.3f}"
            )

        exact_set_accuracy = (
            df["gold_signature"]
            == df[f"{prefix}_signature"]
        ).mean()

        macro_f1 = sum(
            metrics["f1"]
            for metrics
            in metric_results.values()
        ) / len(CLAUSES)

        print("\nOVERALL")
        print("-" * 40)

        print(
            f"Exact category-set accuracy: "
            f"{exact_set_accuracy:.3f}"
        )

        print(
            f"Macro F1:                    "
            f"{macro_f1:.3f}"
        )


if __name__ == "__main__":
    main()