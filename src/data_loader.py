"""Load Excel files into pandas DataFrames. All data files must be inside the data folder."""

import pandas as pd
from pathlib import Path

# Data files (Excel, JSON) must live in project_root/data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_all_tables() -> dict[str, pd.DataFrame]:
    """Load all Excel tables from the data folder into a dict of DataFrames."""
    tables = {}
    for name in ["Clients", "Invoices", "InvoiceLineItems"]:
        path = DATA_DIR / f"{name}.xlsx"
        if path.exists():
            tables[name] = pd.read_excel(path)
    return tables


def get_sqlite_connection(tables: dict[str, pd.DataFrame]):
    """Create an in-memory SQLite DB from DataFrames for SQL execution."""
    import sqlite3

    conn = sqlite3.connect(":memory:")

    for table_name, df in tables.items():
        # SQLite-friendly table name (no spaces)
        safe_name = table_name.replace(" ", "_")
        df.to_sql(safe_name, conn, index=False, if_exists="replace")

    return conn
