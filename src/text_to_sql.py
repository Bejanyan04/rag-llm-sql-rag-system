"""Text-to-SQL: map natural language questions to SQL queries using OpenAI or Anthropic (fallback)."""

import os
from typing import Optional, List, Dict, Any

from openai import OpenAI

from .schema import build_schema_prompt
from .llm_client import _is_anthropic_model


def generate_sql(
    question: str,
    tables: dict,
    model: str = "gpt-4o-mini",
    previous_sql: Optional[str] = None,
    validation_error: Optional[str] = None,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    table_names: Optional[List[str]] = None,
    use_retrieval: bool = False,
    schema_collection=None,
    few_shot_collection=None,
) -> str:
    """
    Use OpenAI to generate a SQL query from a natural language question.
    If previous_sql and validation_error are provided, asks the LLM to fix the query.
    Optionally pass few_shot_examples and table_names, or set use_retrieval=True with
    schema_collection/few_shot_collection to retrieve them.
    """
    if use_retrieval and (schema_collection is not None or few_shot_collection is not None):
        from .retrieval import retrieve_relevant_tables, retrieve_few_shot_examples
        if table_names is None and schema_collection is not None:
            table_names = retrieve_relevant_tables(question, tables, schema_collection=schema_collection)
        if few_shot_examples is None and few_shot_collection is not None:
            few_shot_examples = retrieve_few_shot_examples(question, few_shot_collection=few_shot_collection)

    schema_text = build_schema_prompt(
        tables,
        few_shot_examples=few_shot_examples,
        table_names=table_names,
    )

    system = f"""{schema_text}
Use Chain of Thought: before writing SQL, reason step-by-step (tables/columns, join path, filters, aggregations).
Then output ONLY one valid SQLite SQL query. No markdown, no backticks. Add LIMIT when returning many rows (default max 500 unless the question asks for a specific count)."""

    if previous_sql and validation_error:
        user_content = f"""Question: {question}

ReAct (reason then act): Your previous SQL failed validation. Error: {validation_error}

Previous SQL:
{previous_sql}

Reason about the error (syntax, wrong table/column, or schema), then output only the corrected SQL query."""
    else:
        user_content = question

    if _is_anthropic_model(model):
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        sql = (response.content[0].text if response.content else "").strip()
    else:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )
        sql = response.choices[0].message.content.strip()

    # Remove markdown code blocks if present
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )
    return sql
