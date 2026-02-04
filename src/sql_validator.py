"""SQL validation layer: validate queries before execution."""

import re
import sqlite3
from typing import Tuple

# Forbidden keywords - only SELECT (and WITH for CTEs) allowed
FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE",
    "TRUNCATE", "REPLACE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
]


def validate_syntax(sql: str) -> Tuple[bool, str]:
    """
    Validate SQL syntax using sqlparse.
    Returns (is_valid, error_message).
    """
    try:
        import sqlparse
    except ImportError:
        sql_upper = sql.upper().strip()
        if not sql_upper.startswith(("SELECT", "WITH")):
            return False, "SQL must start with SELECT or WITH"
        return True, ""

    try:
        parsed = sqlparse.parse(sql)
        if not parsed:
            return False, "Empty or unparseable SQL"
        return True, ""
    except Exception as e:
        return False, f"Syntax validation failed: {str(e)}"


def validate_safety(sql: str) -> Tuple[bool, str]:
    """
    Validate that SQL contains no dangerous operations.
    Only SELECT and WITH (CTE) are allowed.
    Returns (is_valid, error_message).
    """
    sql_upper = sql.upper()
    # Remove string literals to avoid false positives
    sql_no_strings = re.sub(r"'[^']*'", "''", sql_upper)
    sql_no_strings = re.sub(r'"[^"]*"', '""', sql_no_strings)

    for keyword in FORBIDDEN_KEYWORDS:
        # Match whole word
        pattern = r"\b" + keyword + r"\b"
        if re.search(pattern, sql_no_strings):
            return False, f"Forbidden keyword '{keyword}' is not allowed. Only SELECT queries permitted."

    # Must start with SELECT or WITH
    stripped = sql.strip().upper()
    if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
        return False, "Query must be a SELECT statement (or WITH for CTEs)."

    return True, ""


def validate_schema(sql: str, conn: sqlite3.Connection) -> Tuple[bool, str]:
    """
    Validate that SQL references exist in the database using EXPLAIN QUERY PLAN.
    Returns (is_valid, error_message).
    """
    try:
        cursor = conn.cursor()
        cursor.execute("EXPLAIN QUERY PLAN " + sql)
        cursor.fetchall()
        return True, ""
    except sqlite3.Error as e:
        return False, f"Schema/execution validation failed: {str(e)}"


def validate_sql(sql: str, conn: sqlite3.Connection) -> Tuple[bool, str]:
    """
    Run all validations: syntax, safety, schema.
    Returns (is_valid, error_message).
    """
    valid, msg = validate_syntax(sql)
    if not valid:
        return False, f"[Syntax] {msg}"

    valid, msg = validate_safety(sql)
    if not valid:
        return False, f"[Safety] {msg}"

    valid, msg = validate_schema(sql, conn)
    if not valid:
        return False, f"[Schema] {msg}"

    return True, ""
