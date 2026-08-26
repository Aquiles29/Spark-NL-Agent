from src.clause_categorization import categorize_sql


queries = [
    "SELECT name FROM students",

    """
    SELECT s.name, c.name
    FROM students s
    JOIN classes c ON s.class_id = c.id
    """,

    """
    SELECT department, COUNT(*)
    FROM employees
    GROUP BY department
    """,

    """
    SELECT department, COUNT(*)
    FROM employees
    GROUP BY department
    HAVING COUNT(*) > 5
    """,

    """
    SELECT d.name, COUNT(e.id)
    FROM departments d
    JOIN employees e ON d.id = e.department_id
    GROUP BY d.name
    HAVING COUNT(e.id) > 5
    """
]


for sql in queries:
    result = categorize_sql(sql)

    print("\nSQL:")
    print(sql.strip())

    print("Categories:", result["categories"])
    print("Parse error:", result["parse_error"])