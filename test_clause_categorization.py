from src.schema_utils import get_database_schema


schema = get_database_schema(
    "california_schools"
)

for table, columns in schema.items():

    print("\nTABLE:", table)

    for column in columns:
        print("  -", column)