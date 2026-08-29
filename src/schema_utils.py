import os
import re
import sqlite3


BIRD_DB_ROOT = "db/bird-1"


def normalize_identifier(text):
    """
    Normalize table/column names for lexical matching.
    """

    text = str(text)

    # CamelCase -> Camel Case
    text = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        text
    )

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return " ".join(text.split())


def get_database_schema(db_id):
    """
    Read table and column metadata directly from a BIRD SQLite DB.
    """

    db_path = os.path.join(
        BIRD_DB_ROOT,
        db_id,
        f"{db_id}.sqlite"
    )

    # If it is not there, search recursively inside db/bird-1
    if not os.path.exists(db_path):

        db_path = None

        for root, dirs, files in os.walk(BIRD_DB_ROOT):

            expected_file = f"{db_id}.sqlite"

            if expected_file in files:
                db_path = os.path.join(
                    root,
                    expected_file
                )
                break

    # Still not found
    if db_path is None or not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database '{db_id}' could not be found "
            f"anywhere inside {BIRD_DB_ROOT}"
        )

    connection = sqlite3.connect(db_path)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
    """)

    tables = [
        row[0]
        for row in cursor.fetchall()
    ]

    schema = {}

    for table in tables:

        cursor.execute(
            f'PRAGMA table_info("{table}")'
        )

        columns = [
            row[1]
            for row in cursor.fetchall()
        ]

        schema[table] = columns

    connection.close()

    return schema