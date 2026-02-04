"""RAG-style pipeline: question → Text-to-SQL → validate → retrieve → LLM answer."""

from .config import (
    DEFAULT_SQL_MODEL,
    DEFAULT_ANSWER_MODEL,
    FALLBACK_SQL_MODEL,
    FALLBACK_ANSWER_MODEL,
    MAX_ROWS_FOR_ANSWER,
    MAX_SQL_RETRIES,
)
from .data_loader import load_all_tables, get_sqlite_connection
from .text_to_sql import generate_sql
from .sql_validator import validate_sql as validate_sql_query
from .executor import execute_query
from .llm_client import generate_answer


def run_pipeline(
    question: str,
    sql_model: str = None,
    answer_model: str = None,
    max_rows: int = None,
) -> dict:
    """
    Run the full pipeline: question → SQL → validate (with retry) → execute → answer.
    Returns dict with keys: sql, data, answer, error (if any), validation_passed.
    """
    sql_model = sql_model or DEFAULT_SQL_MODEL
    answer_model = answer_model or DEFAULT_ANSWER_MODEL
    max_rows = max_rows or MAX_ROWS_FOR_ANSWER

    tables = load_all_tables()
    conn = get_sqlite_connection(tables)

    # 1. Generate SQL (with retry on validation failure; fallback model on API failure)
    sql = None
    validation_error = None
    last_error = None
    api_failed = False

    for attempt in range(MAX_SQL_RETRIES):
        model = FALLBACK_SQL_MODEL if api_failed else sql_model
        try:
            sql = generate_sql(
                question,
                tables,
                model=model,
                previous_sql=sql if attempt > 0 else None,
                validation_error=validation_error if attempt > 0 else None,
            )
            api_failed = False
        except Exception as e:
            last_error = str(e)
            api_failed = True
            if attempt < MAX_SQL_RETRIES - 1:
                continue
            conn.close()
            return {
                "sql": None,
                "data": None,
                "answer": None,
                "error": f"SQL generation failed: {last_error}",
                "validation_passed": False,
            }
        # 2. Validate SQL
        valid, validation_error = validate_sql_query(sql, conn)
        if valid:
            break

        if attempt == MAX_SQL_RETRIES - 1:
            conn.close()
            return {
                "sql": sql,
                "data": None,
                "answer": None,
                "error": f"Validation failed after {MAX_SQL_RETRIES} attempts: {validation_error}",
                "validation_passed": False,
            }

    # 3. Execute
    result = execute_query(conn, sql)

    if isinstance(result, str):
        conn.close()
        return {"sql": sql, "data": None, "answer": None, "error": result, "validation_passed": True}

    # 4. Truncate result for answer LLM (row limit)
    rows_for_answer = result.head(max_rows)
    total_rows = len(result)
    truncated = total_rows > max_rows

    # 5. Format data for LLM
    data_str = rows_for_answer.to_string() if len(rows_for_answer) > 0 else "(No rows returned)"
    if truncated:
        data_str += f"\n\n(Showing first {max_rows} of {total_rows} rows. Truncated for display.)"

    # 6. Generate answer (with fallback)
    answer = None
    for model in [answer_model, FALLBACK_ANSWER_MODEL]:
        try:
            answer = generate_answer(question, data_str, model=model)
            break
        except Exception as e:
            last_error = str(e)
            if model == FALLBACK_ANSWER_MODEL:
                # Fallback: return raw data as answer
                answer = (
                    f"Unable to generate a natural language answer ({last_error}). "
                    f"Here are the retrieved results ({len(result)} rows):\n\n{data_str[:3000]}"
                    + ("..." if len(data_str) > 3000 else "")
                )
                break
            continue

    conn.close()
    return {
        "sql": sql,
        "data": result,
        "answer": answer,
        "error": None,
        "validation_passed": True,
    }
