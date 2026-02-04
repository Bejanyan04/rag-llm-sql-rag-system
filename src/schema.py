"""
Schema descriptions for Text-to-SQL. Injects four types of knowledge before LLM generation:
1. Structural (schema)  2. Relational (FKs, joins)  3. Semantic (meaning, rules)  4. Usage constraints
"""

from typing import Optional, List, Dict, Any

from .domain_knowledge import (
    TABLE_SEMANTICS,
    COLUMN_SEMANTICS,
    COUNTRY_REGIONS,
    ALLOWED_JOIN_PATHS,
    TIME_SEMANTICS,
    FEW_SHOT_EXAMPLES,
)
from .retrieval import load_few_shot_examples_from_file


def _infer_relationships(tables: dict) -> list:
    """Infer FK relationships and cardinality from column names."""
    rels = []
    seen = set()
    cols_by_table = {t.replace(" ", "_"): list(df.columns) for t, df in tables.items()}
    for table, cols in cols_by_table.items():
        for c in cols:
            c_lower = str(c).lower()
            key = (table, c)
            if "client" in c_lower and "id" in c_lower and table != "Clients" and key not in seen:
                rels.append(f"- {table}.{c} -> Clients (many-to-one: many invoices per client)")
                seen.add(key)
            if "invoice" in c_lower and "id" in c_lower and table == "InvoiceLineItems" and key not in seen:
                rels.append(f"- InvoiceLineItems.{c} -> Invoices (many-to-one: many line items per invoice)")
                seen.add(key)
    return rels if rels else [
        "- Invoices.client_id -> Clients (many-to-one)",
        "- InvoiceLineItems.invoice_id -> Invoices (many-to-one)",
    ]


def _get_column_samples(df, col, max_samples: int = 5) -> str:
    """Get sample values for a column (for categorical/text columns)."""
    try:
        unique = df[col].dropna().unique()
        samples = list(unique)[:max_samples]
        str_samples = [repr(str(s))[:50] for s in samples]
        return ", ".join(str_samples)
    except Exception:
        return "(unable to sample)"


def _get_column_semantic(col_name: str) -> Optional[str]:
    """Get semantic description for a column if available."""
    col_norm = str(col_name).lower().strip().replace(" ", "_")
    for key, desc in COLUMN_SEMANTICS.items():
        if key.lower().replace(" ", "_") == col_norm:
            return desc
    return None


def _get_table_semantic(table_name: str) -> Optional[str]:
    """Get semantic description for a table if available."""
    safe = table_name.replace(" ", "_")
    return TABLE_SEMANTICS.get(safe)


def _format_few_shot_section(examples: List[Dict[str, Any]]) -> str:
    """Format a list of {question, sql} dicts into a few-shot examples section."""
    if not examples:
        return ""
    lines = [
        "",
        "=" * 60,
        "FEW-SHOT EXAMPLES",
        "=" * 60,
        "",
        "Answer the following question with ONLY a SQLite SQL query (no markdown, no explanation).",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i} – Question: {ex.get('question', '')}")
        lines.append(f"SQL: {ex.get('sql', '')}")
        lines.append("")
    lines.append("Now answer the following question with only SQL.")
    return "\n".join(lines)


def build_schema_prompt(
    tables: dict,
    include_samples: bool = True,
    max_sample_values: int = 5,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    table_names: Optional[List[str]] = None,
) -> str:
    """
    Build full knowledge prompt: structural, relational, semantic, usage constraints.
    Optionally filter to specific tables (for retrieval) and/or append few-shot examples.
    """
    if table_names is not None:
        tables = {k: v for k, v in tables.items() if k.replace(" ", "_") in [t.replace(" ", "_") for t in table_names]}
    if few_shot_examples is None:
        few_shot_examples = load_few_shot_examples_from_file()

    lines = [
        "You are a SQL expert. Write SQLite SQL queries using the knowledge below.",
        "Use ALL four knowledge types: structural, relational, semantic, and usage constraints.",
        "",
        "=" * 60,
        "1. STRUCTURAL KNOWLEDGE (Schema)",
        "=" * 60,
        "",
    ]

    # --- 1. Structural: tables, columns, data types ---
    for table_name, df in tables.items():
        safe_name = table_name.replace(" ", "_")
        row_count = len(df)
        sem = _get_table_semantic(safe_name)
        sem_line = f"  # {sem}" if sem else ""
        lines.append(f"**{safe_name}** ({row_count} rows){sem_line}")
        lines.append("")

        for col in df.columns:
            dtype = str(df[col].dtype)
            col_ref = f'"{col}"' if " " in str(col) else str(col)
            col_sem = _get_column_semantic(col)
            sem_suffix = f"  # {col_sem}" if col_sem else ""

            if include_samples and (df[col].dtype.kind == "O" or str(df[col].dtype) == "object"):
                samples = _get_column_samples(df, col, max_sample_values)
                lines.append(f"  - {col_ref} ({dtype}): samples: {samples}{sem_suffix}")
            elif include_samples and (
                df[col].dtype.kind == "M" or "datetime" in str(df[col].dtype).lower()
            ):
                samples = _get_column_samples(df, col, max_sample_values)
                lines.append(f"  - {col_ref} ({dtype}): samples: {samples}{sem_suffix}")
            else:
                null_count = df[col].isna().sum()
                null_info = f", {null_count} nulls" if null_count > 0 else ""
                lines.append(f"  - {col_ref} ({dtype}{null_info}){sem_suffix}")

        lines.append("")

    # --- 2. Relational: FKs, join paths, cardinality ---
    lines.append("=" * 60)
    lines.append("2. RELATIONAL KNOWLEDGE (Foreign keys, Join paths, Cardinality)")
    lines.append("=" * 60)
    lines.append("")
    lines.extend(_infer_relationships(tables))
    lines.append("")

    # --- 3. Semantic: table meaning, column meaning, business rules ---
    lines.append("=" * 60)
    lines.append("3. SEMANTIC KNOWLEDGE (Domain meaning, Business rules)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("**Table meanings:**")
    for t, sem in TABLE_SEMANTICS.items():
        lines.append(f"  - {t}: {sem}")
    lines.append("")
    lines.append("**Country regions (use for 'European clients', 'US clients', etc.):**")
    for region, countries in COUNTRY_REGIONS.items():
        countries_str = ", ".join(f"'{c}'" for c in countries)
        lines.append(f"  - {region}: country IN ({countries_str})")
    lines.append("")

    # --- 4. Usage constraints: allowed joins, aggregation, time ---
    lines.append("=" * 60)
    lines.append("4. USAGE CONSTRAINTS (Allowed joins, Aggregation, Time semantics)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("**Allowed join paths:**")
    for j in ALLOWED_JOIN_PATHS:
        lines.append(f"  - {j}")
    lines.append("")
    lines.append("**Aggregation logic:**")

    lines.append("**Time semantics:**")
    for t in TIME_SEMANTICS:
        lines.append(f"  - {t}")
    lines.append("")

    # --- Few-shot examples (if any) ---
    if few_shot_examples:
        lines.append(_format_few_shot_section(few_shot_examples))

    # --- Chain of Thought (CoT) and ReAct ---
    lines.append("=" * 60)
    lines.append("CHAIN OF THOUGHT (CoT) & REACT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("**Chain of Thought:** Before writing SQL, reason step by step:")
    lines.append("  1. Which tables and columns are needed for the question?")
    lines.append("  2. What join path connects them (use RELATIONAL KNOWLEDGE)?")
    lines.append("  3. What filters (WHERE), aggregations (GROUP BY, SUM, etc.), and ordering (ORDER BY) apply?")
    lines.append("  4. For complex or multi-step questions, break the logic into clear steps, then write the SQL.")
    lines.append("")
    lines.append("**ReAct (Reasoning + Acting):** If a query fails validation or execution:")
    lines.append("  - Reason about the error message: syntax, wrong table/column, or schema mismatch.")
    lines.append("  - Adapt to schema variations (e.g. column names, date formats) and refine the SQL.")
    lines.append("  - Iteratively fix and retry until the query is valid.")
    lines.append("")

    # --- Final instructions ---
    lines.append("=" * 60)
    lines.append("INSTRUCTIONS")
    lines.append("=" * 60)
    lines.append("- Use SQLite syntax. Column names with spaces need double quotes.")
    lines.append("- Add LIMIT when returning many rows (default max 500 unless the question asks for a specific count).")
    lines.append("- After reasoning (CoT), return ONLY the final valid SQL, no markdown or explanation.")

    return "\n".join(lines)
