"""
Build evaluation/benchmark.json from data: run each expected_sql and set expected_answer
by formatting the result (no LLM). Run from project root: python tools/build_benchmark.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_all_tables, get_sqlite_connection
from src.executor import execute_query

# New benchmark items: questions derived from data, NOT from the 15 test-set questions.
# Each expected_answer will be filled by running expected_sql and formatting the result.
BENCHMARK_ITEMS = [
    {"question": "How many clients are there?", "expected_sql": "SELECT COUNT(*) AS cnt FROM Clients;"},
    {"question": "List all client names and their countries.", "expected_sql": "SELECT client_name, country FROM Clients;"},
    {"question": "How many invoices are there in total?", "expected_sql": "SELECT COUNT(*) AS cnt FROM Invoices;"},
    {"question": "What are the distinct industries in the Clients table?", "expected_sql": "SELECT DISTINCT industry FROM Clients ORDER BY industry;"},
    {"question": "How many line items are there in total?", "expected_sql": "SELECT COUNT(*) AS cnt FROM InvoiceLineItems;"},
    {"question": "For each invoice status, how many invoices have that status?", "expected_sql": "SELECT status, COUNT(*) AS cnt FROM Invoices GROUP BY status;"},
    {"question": "Which countries do we have clients in?", "expected_sql": "SELECT DISTINCT country FROM Clients ORDER BY country;"},
    {"question": "How many invoices were issued in 2024?", "expected_sql": "SELECT COUNT(*) AS cnt FROM Invoices WHERE strftime('%Y', invoice_date) = '2024';"},
    {"question": "What is the total number of line items for invoice I1001?", "expected_sql": "SELECT COUNT(*) AS cnt FROM InvoiceLineItems WHERE invoice_id = 'I1001';"},
    {"question": "List client name and industry for the first three clients by client_id.", "expected_sql": "SELECT client_name, industry FROM Clients ORDER BY client_id LIMIT 3;"},
]


def format_result_as_answer(df) -> str:
    """Format a query result DataFrame as a short ground-truth answer (no LLM)."""
    if df is None or len(df) == 0:
        return "No rows returned."
    if len(df) == 1 and len(df.columns) == 1:
        val = df.iloc[0, 0]
        return f"The result is {val}."
    if len(df) == 1:
        parts = [f"{col}={df.iloc[0][col]}" for col in df.columns]
        return "One row: " + "; ".join(parts) + "."
    # Multiple rows: summarize
    n = len(df)
    if n <= 3:
        return f"Total {n} rows. " + df.to_string(index=False)
    return f"Total {n} rows. First few: " + df.head(3).to_string(index=False) + "."


def main():
    tables = load_all_tables()
    conn = get_sqlite_connection(tables)

    benchmark = []
    for item in BENCHMARK_ITEMS:
        sql = item["expected_sql"]
        result = execute_query(conn, sql)
        if isinstance(result, str):
            expected_answer = f"Query error: {result}"
        else:
            expected_answer = format_result_as_answer(result)
        benchmark.append({
            "question": item["question"],
            "expected_sql": sql,
            "expected_answer": expected_answer,
        })

    conn.close()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "benchmark.json"
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Wrote {len(benchmark)} items to {out_path}")


if __name__ == "__main__":
    main()
