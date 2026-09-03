import ast
import json

import sqlglot
import pandas as pd
import pyspark.sql
import numpy as np

NULL_TOKEN = "__NULL__"

def columns_equal(series_a, series_b, rtol=1e-8, atol=1e-10):
    """
    Compare two result columns.

    Numeric values are compared with floating-point tolerance.
    Non-numeric values are compared as normalized strings.
    """

    if len(series_a) != len(series_b):
        return False

    numeric_a = pd.to_numeric(series_a, errors="coerce")
    numeric_b = pd.to_numeric(series_b, errors="coerce")

    # If both complete columns are numeric, compare numerically
    if numeric_a.notna().all() and numeric_b.notna().all():
        return np.allclose(
            numeric_a.to_numpy(dtype=float),
            numeric_b.to_numpy(dtype=float),
            rtol=rtol,
            atol=atol,
            equal_nan=True
        )

    # Otherwise compare as strings
    normalized_a = (
        series_a
        .astype(str)
        .str.strip()
        .tolist()
    )

    normalized_b = (
        series_b
        .astype(str)
        .str.strip()
        .tolist()
    )

    return normalized_a == normalized_b

def result_to_obj(s):
    if s and isinstance(s, str):
        try:
            parsed = json.loads(s)
        except Exception:
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                parsed = [{"value": s}]
        result = parsed
    else:
        result = s

    return result


def convert_to_dataframe(obj):
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj
    elif isinstance(obj, pyspark.sql.DataFrame):
        return obj.toPandas()
    else:
        return pd.DataFrame(obj)

def _flatten_values(df):
    arr = df.to_numpy(copy=False).astype(object, copy=False)

    mask = pd.isna(arr)
    if mask.any():
        arr = arr.copy()
        arr[mask] = NULL_TOKEN

    # normalize numerics -> "49729", 49729, 49729.0 ---> "49729"
    # convert everything to string
    def canon(v):
        try:
            if isinstance(v, str):
                v = v.strip()
                i = int(v)
                return str(i)
        except Exception:
            pass
        try:
            f = float(v)
            if float(f).is_integer():
                return str(int(f))
        except Exception:
            pass

        # convert all null values to a single token
        try:
            s = str(v).strip()
            s_low = s.lower()

            if s_low in {"none", "null", "nan", "na", "n/a"}:
                return NULL_TOKEN

            return s
        except Exception:
            pass

    out = np.empty(arr.size, dtype=object)
    flat = arr.ravel()
    for i, v in enumerate(flat):
        out[i] = canon(v)

    return out


def _normalize_value(v):
    """Normalize a single value to a canonical string representation."""
    if pd.isna(v):
        return NULL_TOKEN

    try:
        if isinstance(v, str):
            v = v.strip()
            i = int(v)
            return str(i)
    except Exception:
        pass

    try:
        f = float(v)
        if float(f).is_integer():
            return str(int(f))
    except Exception:
        pass

    try:
        s = str(v).strip()
        s_low = s.lower()

        if s_low in {"none", "null", "nan", "na", "n/a"}:
            return NULL_TOKEN

        return s
    except Exception:
        pass

    return str(v)


def _get_column_values(df, col_idx):
    """Return a tuple of normalized values for the given column index."""
    return tuple(_normalize_value(v) for v in df.iloc[:, col_idx])

import math


def _values_equal(a, b, rtol=1e-8, atol=1e-10):
    """
    Compare two normalized result values.

    Numeric values are compared with floating-point tolerance.
    Non-numeric values are compared exactly.
    """

    # Handle None
    if a is None or b is None:
        return a is None and b is None

    # Try numeric comparison
    try:
        a_num = float(a)
        b_num = float(b)

        # Handle NaN
        if math.isnan(a_num) and math.isnan(b_num):
            return True

        return math.isclose(
            a_num,
            b_num,
            rel_tol=rtol,
            abs_tol=atol
        )

    except (ValueError, TypeError):
        pass

    # Non-numeric values remain exact
    return a == b


def _columns_equal(col_a, col_b, ignore_order=False):
    if len(col_a) != len(col_b):
        return False

    # When ORDER BY matters, preserve row order
    if not ignore_order:
        return all(
            _values_equal(a, b)
            for a, b in zip(col_a, col_b)
        )

    # Without ORDER BY, SQL does not guarantee row order.
    # Compare as multisets while preserving duplicates.
    unmatched = list(col_b)

    for value_a in col_a:
        match_index = None

        for i, value_b in enumerate(unmatched):
            if _values_equal(value_a, value_b):
                match_index = i
                break

        if match_index is None:
            return False

        unmatched.pop(match_index)

    return True

def has_top_level_order_by(sql, dialect="sqlite"):
    """
    Return True only when the outer SQL query has ORDER BY.

    ORDER BY clauses inside subqueries do not make the final
    result order-sensitive.
    """
    if not sql:
        return False

    try:
        tree = sqlglot.parse_one(
            sql.strip().rstrip(";"),
            read=dialect
        )

        return tree.args.get("order") is not None

    except Exception:
        # Safer fallback: treat ordering as non-significant
        return False

def _row_equal(row_a, row_b):
    """
    Compare two result rows while preserving the relationship
    between values across columns.
    """
    if len(row_a) != len(row_b):
        return False

    return all(
        _values_equal(a, b)
        for a, b in zip(row_a, row_b)
    )


def _rows_equal(rows_a, rows_b, ignore_order=False):
    """
    Compare two collections of rows.

    If ignore_order=True, rows are compared as multisets:
    row order is ignored but duplicates and value associations
    across columns are preserved.
    """
    if len(rows_a) != len(rows_b):
        return False

    if not ignore_order:
        return all(
            _row_equal(a, b)
            for a, b in zip(rows_a, rows_b)
        )

    unmatched = list(rows_b)

    for row_a in rows_a:
        match_index = None

        for i, row_b in enumerate(unmatched):
            if _row_equal(row_a, row_b):
                match_index = i
                break

        if match_index is None:
            return False

        unmatched.pop(match_index)

    return True

def execution_accuracy(df_gt, df_inf, order_sensitive=False):
    """
    Compute study-specific execution accuracy.

    A generated result is considered correct when the complete
    ground-truth result can be matched to a projection of the
    generated result.

    Column names and column positions do not need to match, and
    additional generated columns are permitted. However, value
    associations across columns are preserved.

    When the reference query has no top-level ORDER BY, rows are
    compared as an unordered multiset while preserving duplicates.
    """

    df_gt = convert_to_dataframe(df_gt)
    df_inf = convert_to_dataframe(df_inf)

    if df_gt.empty and df_inf.empty:
        return 1.0

    if df_gt.empty or df_inf.empty:
        return 0.0

    n_gt_cols = df_gt.shape[1]
    n_inf_cols = df_inf.shape[1]

    # Generated result cannot contain all gold columns
    # if it has fewer columns than the reference result.
    if n_inf_cols < n_gt_cols:
        return 0.0

    # Normalize inferred columns once.
    inf_cols = [
        _get_column_values(df_inf, col_idx)
        for col_idx in range(n_inf_cols)
    ]

    # Candidate inferred columns for each gold column.
    # This is used only to reduce the number of possible mappings.
    candidates = {}

    for gt_idx in range(n_gt_cols):
        gt_col = _get_column_values(df_gt, gt_idx)

        candidates[gt_idx] = [
            inf_idx
            for inf_idx in range(n_inf_cols)
            if _columns_equal(
                gt_col,
                inf_cols[inf_idx],
                ignore_order=not order_sensitive
            )
        ]

        if not candidates[gt_idx]:
            return 0.0

    # Try the most constrained gold columns first.
    search_order = sorted(
        range(n_gt_cols),
        key=lambda idx: len(candidates[idx])
    )

    mapping = {}
    used_inf_cols = set()

    def mapping_is_correct():
        gold_rows = [
            tuple(
                _normalize_value(df_gt.iloc[row_idx, gt_idx])
                for gt_idx in range(n_gt_cols)
            )
            for row_idx in range(len(df_gt))
        ]

        inferred_rows = [
            tuple(
                _normalize_value(
                    df_inf.iloc[row_idx, mapping[gt_idx]]
                )
                for gt_idx in range(n_gt_cols)
            )
            for row_idx in range(len(df_inf))
        ]

        return _rows_equal(
            gold_rows,
            inferred_rows,
            ignore_order=not order_sensitive
        )

    def search(position):
        if position == len(search_order):
            return mapping_is_correct()

        gt_idx = search_order[position]

        for inf_idx in candidates[gt_idx]:
            if inf_idx in used_inf_cols:
                continue

            mapping[gt_idx] = inf_idx
            used_inf_cols.add(inf_idx)

            if search(position + 1):
                return True

            used_inf_cols.remove(inf_idx)
            del mapping[gt_idx]

        return False

    return 1.0 if search(0) else 0.0

def normalize_sql(sql, dialect="sqlite"):
    """
    Parse and normalize SQL for structural Exact Match comparison.

    Returns:
        Normalized SQL string, or None if parsing fails.
    """

    if not sql or not isinstance(sql, str):
        return None

    try:
        tree = sqlglot.parse_one(
            sql.strip().rstrip(";"),
            read=dialect
        )

        return tree.sql(
            dialect="sqlite",
            pretty=False,
            normalize=True
        )

    except Exception:
        return None


def exact_match_sql(gold_sql, generated_sql):
    """
    Compute normalized SQL Exact Match.

    Gold BIRD SQL is parsed as SQLite.
    Generated agent SQL is parsed as Spark SQL.

    Returns:
        1.0 if normalized SQL is identical
        0.0 otherwise
    """

    if not gold_sql or not generated_sql:
        return 0.0

    gold_normalized = normalize_sql(
        gold_sql,
        dialect="sqlite"
    )

    generated_normalized = normalize_sql(
        generated_sql,
        dialect="spark"
    )

    if gold_normalized is None or generated_normalized is None:
        return 0.0

    return float(
        gold_normalized == generated_normalized
    )
