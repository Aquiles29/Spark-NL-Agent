import re

from src.schema_utils import normalize_identifier



AGGREGATION_WORDS = [
    "count",
    "number of",
    "how many",
    "average",
    "avg",
    "mean",
    "total",
    "sum",
    "maximum",
    "minimum",
    "highest",
    "lowest",
]


def _contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)


def predict_group_by(question):
    """
    Predict whether a SQL GROUP BY clause is likely required
    using only the natural-language question.
    """

    q = question.lower().strip()

    aggregation = _contains_any(
        q,
        AGGREGATION_WORDS
    )

    grouping_cues = [
        "for each",
        "for every",
        "per ",
        "by each",
        "grouped by",
        "group by",
        "in each",
        "of each",
        "among each",
    ]

    explicit_grouping = _contains_any(
        q,
        grouping_cues
    )

    # GROUP BY is most likely when an aggregation is requested
    # separately for different entities/categories.
    return aggregation and explicit_grouping


def predict_having(question):
    """
    Predict whether HAVING is likely required.

    Typical HAVING questions impose a condition on an aggregated
    group, e.g.:
        "Which departments have more than 5 employees?"
    """

    q = question.lower().strip()

    threshold_patterns = [
        r"\bmore than\s+\d+",
        r"\bless than\s+\d+",
        r"\bfewer than\s+\d+",
        r"\bat least\s+\d+",
        r"\bat most\s+\d+",
        r"\bgreater than\s+\d+",
        r"\bno more than\s+\d+",
        r"\bno fewer than\s+\d+",
    ]

    threshold = any(
        re.search(pattern, q)
        for pattern in threshold_patterns
    )

    group_relation_cues = [
        "which ",
        "that have",
        "that has",
        "with more than",
        "with less than",
        "with fewer than",
        "with at least",
        "with at most",
        "whose number",
        "whose count",
        "whose average",
        "whose total",
    ]

    group_relation = _contains_any(
        q,
        group_relation_cues
    )

    return threshold and group_relation


def predict_join(question):
    """
    Predict whether JOIN is likely required using question text only.

    This is intentionally conservative. JOIN prediction from language
    alone is difficult because the need for a JOIN depends strongly
    on database schema structure.
    """

    q = question.lower().strip()

    relationship_cues = [
        "and their",
        "with their",
        "along with their",
        "corresponding",
        "associated with",
        "belonging to",
        "related to",
        "whose ",
    ]

    return _contains_any(
        q,
        relationship_cues
    )


def predict_categories(question):
    """
    Predict clause categories from the NL question only.

    Returns the same structure used by gold SQL categorization,
    allowing direct offline comparison.
    """

    join = predict_join(question)
    group_by = predict_group_by(question)
    having = predict_having(question)

    categories = []

    if join:
        categories.append("JOIN")

    if group_by:
        categories.append("GROUP_BY")

    if having:
        categories.append("HAVING")

    return {
        "JOIN": join,
        "GROUP_BY": group_by,
        "HAVING": having,
        "categories": categories,
    }

GENERIC_SCHEMA_WORDS = {
    "id",
    "code",
    "name",
    "number",
    "type",
    "value",
    "data",
}


def _normalize_token(token):
    """
    Lightweight normalization for question/schema matching.
    """

    token = token.lower()

    # Simple plural normalization
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"

    elif (
        token.endswith("s")
        and len(token) > 3
        and not token.endswith("ss")
    ):
        token = token[:-1]

    # A few morphological forms useful for schema matching
    aliases = {
        "mailing": "mail",
        "addresses": "address",
    }

    return aliases.get(token, token)


def _tokenize(text):
    normalized = normalize_identifier(text)

    return {
        _normalize_token(token)
        for token in normalized.split()
        if len(token) > 1
    }


def _get_table_tokens(table_name, columns):
    """
    Return lexical tokens describing a table.
    """

    tokens = set()

    tokens.update(
        _tokenize(table_name)
    )

    for column in columns:
        tokens.update(
            _tokenize(column)
        )

    return tokens


def predict_join_schema_aware(question, schema):
    """
    Predict JOIN using both the natural-language question
    and database schema.

    The main idea is that a JOIN is likely when concepts mentioned
    in the question are distributed across multiple tables.
    """

    # Keep explicit NL relationship cues from the original predictor
    if predict_join(question):
        return True

    question_tokens = _tokenize(question)

    # Remove very generic words that provide little schema evidence
    question_tokens = {
        token
        for token in question_tokens
        if token not in GENERIC_SCHEMA_WORDS
    }

    table_tokens = {}

    for table_name, columns in schema.items():
        table_tokens[table_name] = (
            _get_table_tokens(
                table_name,
                columns
            )
        )

    # Determine which tables match each question concept
    token_to_tables = {}

    for token in question_tokens:

        matching_tables = {
            table_name
            for table_name, tokens
            in table_tokens.items()
            if token in tokens
        }

        if matching_tables:
            token_to_tables[token] = matching_tables

    # Evidence unique to individual tables
    unique_evidence = {
        table_name: set()
        for table_name in schema
    }

    # General overlap between question and each table
    all_evidence = {
        table_name: set()
        for table_name in schema
    }

    for token, matching_tables in token_to_tables.items():

        for table_name in matching_tables:
            all_evidence[table_name].add(token)

        if len(matching_tables) == 1:
            table_name = next(iter(matching_tables))
            unique_evidence[table_name].add(token)

    # Strongest signal:
    # different question concepts uniquely point to >= 2 tables
    unique_tables = [
        table_name
        for table_name, evidence
        in unique_evidence.items()
        if evidence
    ]

    if len(unique_tables) >= 2:
        return True

    # Secondary signal:
    # two tables both have substantial lexical evidence
    ranked_tables = sorted(
        all_evidence.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    if len(ranked_tables) >= 2:

        first_score = len(ranked_tables[0][1])
        second_score = len(ranked_tables[1][1])

        combined_evidence = (
            ranked_tables[0][1]
            | ranked_tables[1][1]
        )

        if (
            first_score >= 2
            and second_score >= 2
            and len(combined_evidence) >= 3
        ):
            return True

    return False

def predict_categories_schema_aware(
    question,
    schema
):
    """
    Predict clause categories using the question and schema.
    """

    join = predict_join_schema_aware(
        question,
        schema
    )

    # For now GROUP BY and HAVING remain question-based.
    group_by = predict_group_by(question)
    having = predict_having(question)

    categories = []

    if join:
        categories.append("JOIN")

    if group_by:
        categories.append("GROUP_BY")

    if having:
        categories.append("HAVING")

    return {
        "JOIN": join,
        "GROUP_BY": group_by,
        "HAVING": having,
        "categories": categories,
    }