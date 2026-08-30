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

    The rules cover:
    - explicit grouping ("for each", "per", ...)
    - frequency/ranking questions ("most common", "least numerous", ...)
    - nested aggregation ("average number of ...")
    - count-per-entity constructions ("how many X does Y have?")
    """

    q = question.lower().strip()

    aggregation = _contains_any(
        q,
        AGGREGATION_WORDS
    )

    explicit_grouping_cues = [
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
        explicit_grouping_cues
    )

    # Questions asking which category/entity occurs most or least often
    frequency_cues = [
        "most common",
        "least common",
        "most numerous",
        "least numerous",
        "majority",
        "most number of",
        "least number of",
        "highest number of",
        "lowest number of",
        "largest number of",
        "smallest number of",
    ]

    frequency_grouping = _contains_any(
        q,
        frequency_cues
    )

    # Nested aggregation:
    # count something per entity, then average those counts
    nested_aggregation_patterns = [
        r"\baverage number of\b",
        r"\bon average how many\b",
        r"\baverage count of\b",
    ]

    nested_aggregation = any(
        re.search(pattern, q)
        for pattern in nested_aggregation_patterns
    )

    # Example:
    # "How many double bonds does TR006 have?"
    #
    # BIRD often groups these when the SQL also selects
    # information about the parent entity.
    count_per_entity_patterns = [
        r"\bhow many\b.+\bdoes\b.+\bhave\b",
        r"\bhow many\b.+\bdo\b.+\bhave\b",
        r"\bhow many\b.+\bhas\b",
    ]

    count_per_entity = any(
        re.search(pattern, q)
        for pattern in count_per_entity_patterns
    )

    return (
        (aggregation and explicit_grouping)
        or frequency_grouping
        or nested_aggregation
        or count_per_entity
    )

NUMBER_PATTERN = (
    r"(?:"
    r"\d+(?:\.\d+)?"
    r"|one|two|three|four|five|six|seven|eight|nine|ten"
    r"|eleven|twelve|thirteen|fourteen|fifteen"
    r"|sixteen|seventeen|eighteen|nineteen|twenty"
    r"|thirty|forty|fifty|sixty|seventy|eighty|ninety"
    r"|hundred"
    r")"
)

def predict_having(question):
    """
    Predict whether HAVING is likely required.

    HAVING normally represents a condition applied after aggregation,
    such as:
        "superheroes with over 15 super powers"
        "events attended by more than 10 members"
        "expenses that spend more than fifty dollars on average"
    """

    q = question.lower().strip()

    threshold_patterns = [
        rf"\bmore than\s+{NUMBER_PATTERN}\b",
        rf"\bover\s+{NUMBER_PATTERN}\b",
        rf"\bless than\s+{NUMBER_PATTERN}\b",
        rf"\bfewer than\s+{NUMBER_PATTERN}\b",
        rf"\bat least\s+{NUMBER_PATTERN}\b",
        rf"\bat most\s+{NUMBER_PATTERN}\b",
        rf"\bgreater than\s+{NUMBER_PATTERN}\b",

        # Number before comparator:
        # "two or more", "10 or more"
        rf"\b{NUMBER_PATTERN}\s+or more\b",
        rf"\b{NUMBER_PATTERN}\s+or fewer\b",
        rf"\b{NUMBER_PATTERN}\s+or less\b",
    ]

    threshold = any(
        re.search(pattern, q)
        for pattern in threshold_patterns
    )

    if not threshold:
        return False

    # Explicit aggregation concepts
    aggregation_condition_cues = [
        "average",
        "avg",
        "mean",
        "count",
        "number of",
        "total",
        "sum",
    ]

    aggregation_condition = _contains_any(
        q,
        aggregation_condition_cues
    )

    # Relational threshold constructions often describe an
    # aggregate over related rows:
    #
    # "superheroes with over 15 super powers"
    # "events attended by more than 10 members"
    relational_threshold_patterns = [
        rf"\bwith\s+(?:more than|over|at least|greater than)\s+{NUMBER_PATTERN}\b",
        rf"\b(?:have|has|had|having)\b.+(?:more than|over|at least|greater than)\s+{NUMBER_PATTERN}\b",
        rf"\battended by\s+(?:more than|over|at least)\s+{NUMBER_PATTERN}\b",
        rf"\battendance\b.+(?:more than|over|at least)\s+{NUMBER_PATTERN}\b",
        rf"\bincurred\b.+(?:more than|over)\s+{NUMBER_PATTERN}\b",
    ]

    relational_threshold = any(
        re.search(pattern, q)
        for pattern in relational_threshold_patterns
    )

    # If GROUP BY is already strongly indicated and the question
    # contains an aggregate threshold, HAVING is a plausible clause.
    group_by_context = predict_group_by(question)

    return (
        aggregation_condition
        or relational_threshold
        or group_by_context
    )


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