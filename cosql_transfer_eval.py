import json
import sqlite3
from pathlib import Path

import sqlglot
from sqlglot import exp

from src.clause_predictor import predict_categories_schema_aware


COSQL_ROOT = Path("db/cosql_dataset/cosql_dataset")

DEV_FILE = (
    COSQL_ROOT
    / "sql_state_tracking"
    / "cosql_dev.json"
)

DATABASE_ROOT = COSQL_ROOT / "database"


def get_database_path(db_id):
    return (
        DATABASE_ROOT
        / db_id
        / f"{db_id}.sqlite"
    )


def get_database_schema(db_id):
    """
    Read table and column names directly from the CoSQL SQLite database.

    This matches the type of schema information used by the BIRD
    schema-aware predictor: table names and column names only.
    """
    db_path = get_database_path(db_id)

    if not db_path.exists():
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
            WHERE type = 'table'
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


def extract_gold_categories(sql):
    """
    Extract JOIN, GROUP BY and HAVING labels from gold SQL.

    Gold SQL is used only offline for evaluation and never supplied
    to the predictor.
    """
    try:
        tree = sqlglot.parse_one(
            sql.strip().rstrip(";"),
            read="sqlite"
        )
    except Exception:
        return None

    return {
        "JOIN": any(
            isinstance(node, exp.Join)
            for node in tree.walk()
        ),
        "GROUP_BY": any(
            isinstance(node, exp.Group)
            for node in tree.walk()
        ),
        "HAVING": any(
            isinstance(node, exp.Having)
            for node in tree.walk()
        ),
    }


def calculate_metrics(records, label):

    tp = 0
    fp = 0
    fn = 0

    for record in records:

        gold = record["gold"][label]
        pred = record["predicted"][label]

        if gold and pred:
            tp += 1

        elif not gold and pred:
            fp += 1

        elif gold and not pred:
            fn += 1

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    support = sum(
        1
        for record in records
        if record["gold"][label]
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def category_signature(labels):

    active = [
        label
        for label in [
            "JOIN",
            "GROUP_BY",
            "HAVING"
        ]
        if labels[label]
    ]

    if not active:
        return "NONE"

    return " + ".join(active)


def main():

    if not DEV_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DEV_FILE}"
        )

    with open(
        DEV_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        dialogues = json.load(f)

    print(
        f"Loaded {len(dialogues)} CoSQL dev dialogues."
    )

    print(
        "Evaluating FIRST TURN ONLY."
    )

    print(
        "Frozen BIRD predictor; no CoSQL tuning."
    )

    print(
        "OFFLINE evaluation only -- no LLM/API calls."
    )

    records = []

    parse_errors = 0
    missing_databases = 0
    invalid_examples = 0

    schema_cache = {}

    for dialogue_index, dialogue in enumerate(
        dialogues
    ):

        db_id = dialogue.get("database_id")

        interaction = dialogue.get(
            "interaction",
            []
        )

        if not db_id or not interaction:
            invalid_examples += 1
            continue

        # Only the first turn is used so that the question
        # does not depend on previous dialogue context.
        turn = interaction[0]

        question = turn.get("utterance")
        gold_sql = turn.get("query")

        if (
            not question
            or not gold_sql
            or not gold_sql.strip()
        ):
            invalid_examples += 1
            continue

        gold = extract_gold_categories(
            gold_sql
        )

        if gold is None:
            parse_errors += 1
            continue

        try:

            if db_id not in schema_cache:

                schema_cache[db_id] = (
                    get_database_schema(
                        db_id
                    )
                )

            schema = schema_cache[db_id]

        except FileNotFoundError:
            missing_databases += 1
            continue

        prediction = (
            predict_categories_schema_aware(
                question,
                schema
            )
        )

        predicted = {
            "JOIN": bool(
                prediction["JOIN"]
            ),
            "GROUP_BY": bool(
                prediction["GROUP_BY"]
            ),
            "HAVING": bool(
                prediction["HAVING"]
            ),
        }

        records.append({
            "dialogue_index": dialogue_index,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,

            "gold": gold,
            "predicted": predicted,

            "gold_signature": (
                category_signature(gold)
            ),

            "predicted_signature": (
                category_signature(
                    predicted
                )
            ),
        })

    print()
    print("=" * 70)
    print(
        "COSQL FIRST-TURN CLAUSE TRANSFER RESULTS"
    )
    print("=" * 70)

    print(
        f"Eligible examples: {len(records)}"
    )

    print(
        f"Parse errors: {parse_errors}"
    )

    print(
        f"Missing databases: "
        f"{missing_databases}"
    )

    print(
        f"Invalid/skipped examples: "
        f"{invalid_examples}"
    )

    print()

    all_metrics = {}

    for label in [
        "JOIN",
        "GROUP_BY",
        "HAVING"
    ]:

        metrics = calculate_metrics(
            records,
            label
        )

        all_metrics[label] = metrics

        print(label)

        print(
            f"  Precision: "
            f"{metrics['precision']:.3f}"
        )

        print(
            f"  Recall:    "
            f"{metrics['recall']:.3f}"
        )

        print(
            f"  F1:        "
            f"{metrics['f1']:.3f}"
        )

        print(
            f"  Support:   "
            f"{metrics['support']}"
        )

        print()

    macro_f1 = sum(
        all_metrics[label]["f1"]
        for label in all_metrics
    ) / 3

    exact_correct = sum(
        record["gold"]
        == record["predicted"]
        for record in records
    )

    exact_accuracy = (
        exact_correct / len(records)
        if records
        else 0.0
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.3f}"
    )

    print(
        "Exact category-set accuracy: "
        f"{exact_accuracy:.2%}"
    )

    print()
    print("=" * 70)
    print("BIRD DEVELOPMENT REFERENCE")
    print("=" * 70)

    print("JOIN F1:       0.489")
    print("GROUP BY F1:   0.449")
    print("HAVING F1:     0.353")
    print("Macro F1:      0.430")
    print("Exact set:     39.33%")

    output = {
        "evaluation": (
            "CoSQL first-turn transfer"
        ),

        "number_of_examples": len(
            records
        ),

        "parse_errors": parse_errors,

        "missing_databases": (
            missing_databases
        ),

        "invalid_examples": (
            invalid_examples
        ),

        "metrics": all_metrics,

        "macro_f1": macro_f1,

        "exact_category_set_accuracy": (
            exact_accuracy
        ),

        "records": records,
    }

    output_file = (
        "cosql_transfer_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False
        )

    print()
    print(
        f"Saved results to {output_file}"
    )


if __name__ == "__main__":
    main()