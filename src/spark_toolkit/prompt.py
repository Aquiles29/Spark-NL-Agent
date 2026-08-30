# flake8: noqa

SQL_PREFIX = """You are an agent designed to interact with Spark SQL.
Given an input question, create a syntactically correct Spark SQL query to run, then look at the results of the query and return the answer.
Don't limit the result size.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.
"""

SQL_PREFIX_WITH_THOUGHTS = """You are an agent designed to interact with Spark SQL.
Given an input question, create a syntactically correct Spark SQL query to run, then look at the results of the query and return the answer.
Don't limit the result size.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

IMPORTANT: Before using any tool, you MUST first explain your reasoning in a "Thought:" section. 
Describe what you plan to do and why, then proceed with the tool call.
For example:
Thought: I need to see what tables are available in the database first.
Action: list_tables_sql_db
Action Input: {{}}
"""

SQL_SUFFIX = """Begin!

Question: {input}
Thought: I should look at the tables in the database to see what I can query.
{agent_scratchpad}"""

# flake8: noqa
QUERY_CHECKER = """
{query}
Double check the Spark SQL query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Use ` for the in-query strings
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query."""

CLAUSE_GUIDANCE = {
    "JOIN": """
JOIN GUIDANCE:
- Determine which tables contain the information required by the question.
- Identify the appropriate key relationships between those tables before writing the query.
- Use explicit JOIN conditions and avoid Cartesian products.
- Verify that the JOIN keys represent the intended relationship between entities.
""",

    "GROUP_BY": """
GROUP BY GUIDANCE:
- If the question requires an aggregate for each entity, category, or group, identify the grouping columns explicitly.
- Every selected non-aggregated column should be compatible with the GROUP BY.
- Distinguish between an overall aggregate and an aggregate calculated separately for groups.
""",

    "HAVING": """
HAVING GUIDANCE:
- Use HAVING when a condition applies to an aggregated group rather than to individual rows.
- Apply row-level conditions with WHERE before aggregation.
- Apply aggregate conditions such as COUNT, AVG, SUM, MIN, or MAX with HAVING after GROUP BY.
""",
}


def build_clause_guidance(categories):
    """
    Build query-specific guidance from predicted clause categories.
    """

    if not categories:
        return ""

    sections = [
        CLAUSE_GUIDANCE[category]
        for category in categories
        if category in CLAUSE_GUIDANCE
    ]

    if not sections:
        return ""

    return (
        "\n\nQUERY-SPECIFIC SQL GUIDANCE:\n"
        + "\n".join(sections)
        + "\nUse this guidance only when appropriate for the current question."
    )