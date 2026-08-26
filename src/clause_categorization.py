import sqlglot
from sqlglot import exp


CORE_CLAUSES = [
    "JOIN",
    "GROUP_BY",
    "HAVING",
]


def categorize_sql(sql):
    """
    Extract clause-level categories from a SQL query.

    This function is intended for OFFLINE analysis of gold SQL.
    Gold SQL must not be used to determine categories at inference time.
    """

    result = {
        "JOIN": False,
        "GROUP_BY": False,
        "HAVING": False,
        "categories": [],
        "parse_error": None,
    }

    try:
        tree = sqlglot.parse_one(sql, read="sqlite")

    except Exception as e:
        result["parse_error"] = str(e)
        return result

    # JOIN
    result["JOIN"] = any(tree.find_all(exp.Join))

    # GROUP BY
    result["GROUP_BY"] = any(tree.find_all(exp.Group))

    # HAVING
    result["HAVING"] = any(tree.find_all(exp.Having))

    result["categories"] = [
        clause
        for clause in CORE_CLAUSES
        if result[clause]
    ]

    return result