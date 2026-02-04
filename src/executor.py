"""Execute SQL queries against the in-memory SQLite database."""

from typing import Union

import pandas as pd


def execute_query(conn, sql: str) -> Union[pd.DataFrame, str]:
    """Execute SQL and return results as DataFrame or error message."""
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    except Exception as e:
        return f"Error: {str(e)}"
