from src.evaluation import normalize_sql, exact_match_sql


tests = [
    (
        "SELECT name FROM students",
        "select name from students",
        1.0
    ),

    (
        "SELECT name FROM students;",
        "SELECT name FROM students",
        1.0
    ),

    (
        "SELECT name FROM students WHERE age > 18",
        "SELECT name FROM students WHERE age > 20",
        0.0
    ),

    (
        "SELECT name FROM students ORDER BY name ASC",
        "SELECT name FROM students ORDER BY name DESC",
        0.0
    ),

    (
        "SELECT name FROM students LIMIT 3",
        "SELECT name FROM students LIMIT 5",
        0.0
    ),
]


for gold, generated, expected in tests:

    em = exact_match_sql(
        gold,
        generated
    )

    print("\nGold:")
    print(gold)

    print("Generated:")
    print(generated)

    print("Normalized gold:")
    print(normalize_sql(gold, "sqlite"))

    print("Normalized generated:")
    print(normalize_sql(generated, "spark"))

    print(f"EM: {em}")
    print(f"Expected: {expected}")

    assert em == expected


print("\nAll Exact Match tests passed.")