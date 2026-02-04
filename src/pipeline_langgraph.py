"""
LangGraph-style Text-to-SQL pipeline: state graph with tools (get_schema, validate_sql, execute_sql, generate_answer)
and an explicit answer node that turns query results into a natural language answer.
Retry on invalid SQL up to MAX_SQL_RETRIES; then return error.
"""

import json
import re
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from .config import DEFAULT_SQL_MODEL, MAX_ROWS_FOR_ANSWER, MAX_SQL_RETRIES
from .schema import build_schema_prompt
from .sql_validator import validate_sql
from .executor import execute_query
from .llm_client import generate_answer


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], lambda a, b: a + b]
    question: str
    tables: dict
    conn: Any
    validation_retry_count: int
    final_answer: Optional[str]
    final_error: Optional[str]


# ---------------------------------------------------------------------------
# Tool definitions for the LLM (argument schemas only)
# ---------------------------------------------------------------------------

TOOL_DEFS = [
    {
        "name": "get_schema",
        "description": "Get the database schema and instructions for writing SQLite queries. Call this first.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "validate_sql",
        "description": "Validate a SQL query (syntax, safety, schema). Returns valid=true/false and error_message if invalid.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "The SQLite SQL query to validate"},
            },
            "required": ["sql"],
        },
    },
    {
        "name": "execute_sql",
        "description": "Execute a validated SQL query and return row count and a preview of the result. Only call after validate_sql returns valid=true.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "The SQLite SQL query to execute"},
            },
            "required": ["sql"],
        },
    },
    {
        "name": "generate_answer",
        "description": "Generate a natural language answer from the question and the retrieved data (after execute_sql).",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The user's original question"},
                "data_str": {"type": "string", "description": "The data string from execute_sql result (preview or full result)"},
            },
            "required": ["question", "data_str"],
        },
    },
]


def _run_tool(name: str, args: dict, state: PipelineState) -> tuple[str, Optional[dict]]:
    """
    Run one tool; returns (content_string, optional_state_update).
    State update can include validation_retry_count.
    """
    conn = state.get("conn")
    tables = state.get("tables") or {}
    update: dict = {}

    if name == "get_schema":
        schema_text = build_schema_prompt(tables)
        return schema_text, None

    if name == "validate_sql":
        sql = (args.get("sql") or "").strip()
        if not sql:
            return '{"valid": false, "error_message": "Empty SQL"}', None
        valid, error_message = validate_sql(sql, conn)
        if not valid:
            update["validation_retry_count"] = state.get("validation_retry_count", 0) + 1
        return f'{{"valid": {str(valid).lower()}, "error_message": "{error_message or ""}"}}', (update if update else None)

    if name == "execute_sql":
        sql = (args.get("sql") or "").strip()
        result = execute_query(conn, sql)
        if isinstance(result, str):
            return json.dumps({"success": False, "error": result}), None
        preview = result.head(MAX_ROWS_FOR_ANSWER).to_string() if len(result) > 0 else "(No rows returned)"
        if len(result) > MAX_ROWS_FOR_ANSWER:
            preview += f"\n(Showing first {MAX_ROWS_FOR_ANSWER} of {len(result)} rows.)"
        return json.dumps({"success": True, "row_count": len(result), "preview": preview}), None

    if name == "generate_answer":
        question = args.get("question") or state.get("question", "")
        data_str = args.get("data_str") or ""
        answer = generate_answer(question, data_str)
        return answer, None

    return f"Unknown tool: {name}", None


def _answer_node(state: PipelineState) -> dict:
    """
    Explicit answer node: find the last successful execute_sql result in messages,
    or the last validated SQL, run it, and call generate_answer(question, preview).
    Ensures we always have table data to answer from when the agent produced valid SQL.
    """
    messages = state.get("messages") or []
    question = state.get("question", "")
    conn = state.get("conn")

    data_str = None
    # 1) Look for execute_sql result (tool already ran the query)
    for m in reversed(messages):
        if not isinstance(m, ToolMessage) or not m.content:
            continue
        content = m.content.strip()
        if '"success": true' not in content or "preview" not in content:
            continue
        try:
            obj = json.loads(content)
            if obj.get("success") and "preview" in obj:
                data_str = obj["preview"]
                break
        except (json.JSONDecodeError, TypeError):
            match = re.search(r'"preview":\s*"((?:[^"\\]|\\.)*)"', content)
            if match:
                data_str = match.group(1).encode().decode("unicode_escape")
                break

    # 2) If agent never called execute_sql, find the last validated SQL and run it ourselves
    if data_str is None and conn is not None:
        pending_sql = None
        last_validated_sql = None
        for m in messages:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                for tc in m.tool_calls or []:
                    if tc.get("name") == "validate_sql":
                        pending_sql = (tc.get("args") or {}).get("sql")
            if isinstance(m, ToolMessage) and m.content and '"valid": true' in m.content:
                if pending_sql:
                    last_validated_sql = pending_sql.strip()
                pending_sql = None
            if isinstance(m, ToolMessage) and m.content and '"valid": false' in m.content:
                pending_sql = None
        if last_validated_sql:
            result = execute_query(conn, last_validated_sql)
            if isinstance(result, str):
                data_str = f"(Query failed: {result})"
            else:
                preview = result.head(MAX_ROWS_FOR_ANSWER).to_string() if len(result) > 0 else "(No rows returned)"
                if len(result) > MAX_ROWS_FOR_ANSWER:
                    preview += f"\n(Showing first {MAX_ROWS_FOR_ANSWER} of {len(result)} rows.)"
                data_str = preview

    if data_str is None:
        data_str = "(No query was executed. Try asking a question that can be answered with the database tables.)"

    answer = generate_answer(question, data_str)
    return {"final_answer": answer}


def _tools_node(state: PipelineState) -> dict:
    """Execute tool calls from the last AI message and return state update."""
    messages = state.get("messages") or []
    if not messages:
        return {"messages": []}
    last = messages[-1]
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return {"messages": []}

    tool_calls = last.tool_calls or []
    new_messages: list[BaseMessage] = []
    validation_retry_count = state.get("validation_retry_count", 0)

    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args") or {}
        tid = tc.get("id", "")
        content, state_update = _run_tool(name, args, state)
        if state_update and "validation_retry_count" in state_update:
            validation_retry_count = state_update["validation_retry_count"]
        new_messages.append(ToolMessage(content=content, tool_call_id=tid))

    out: dict = {"messages": new_messages}
    if validation_retry_count != state.get("validation_retry_count", 0):
        out["validation_retry_count"] = validation_retry_count
    return out


def _agent_node(state: PipelineState) -> dict:
    """Call LLM with messages; may return AIMessage with tool_calls."""
    messages = state.get("messages") or []
    question = state.get("question", "")

    if not any(m for m in messages if isinstance(m, SystemMessage)):
        system = (
            "You are a SQL expert using ReAct (Reasoning + Acting) and Chain of Thought (CoT).\n\n"
            "**Chain of Thought:** Before acting, reason step-by-step: which tables/columns, join path, "
            "filters, aggregations. For complex or multi-step queries, break the logic into steps.\n\n"
            "**ReAct – you MUST complete this sequence:**\n"
            "(1) Call get_schema first.\n"
            "(2) Call validate_sql with your SQL (raw SQL only, no markdown). If it returns valid=false, fix the SQL and call validate_sql again.\n"
            "(3) When validate_sql returns valid=true, you MUST call execute_sql with the same SQL so the system gets the table data.\n"
            "(4) Do not respond with text only – always call execute_sql after successful validation so the user gets an answer based on the data.\n\n"
            "Use the tools in order: get_schema, then validate_sql, then execute_sql (same SQL when valid). "
            "Respond with ONLY valid SQLite SQL in validate_sql and execute_sql."
        )
        messages = [SystemMessage(content=system)] + list(messages)

    llm = ChatOpenAI(model=DEFAULT_SQL_MODEL, temperature=0).bind_tools(TOOL_DEFS)
    response = llm.invoke(messages)
    return {"messages": [response]}


def _should_continue(state: PipelineState) -> Literal["tools", "answer"]:
    """After agent: go to tools if there are tool_calls, else go to answer node."""
    messages = state.get("messages") or []
    if not messages:
        return "answer"
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "answer"


def _after_tools_route(state: PipelineState) -> Literal["agent", "end"]:
    """After tools: if validation retries exceeded, end with error; else back to agent."""
    if state.get("validation_retry_count", 0) >= MAX_SQL_RETRIES:
        return "end"
    return "agent"


def build_graph():
    """Build and return the compiled LangGraph (for streaming step-by-step in notebooks)."""
    from langgraph.graph import END, StateGraph
    builder = StateGraph(PipelineState)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", _tools_node)
    builder.add_node("answer", _answer_node)
    builder.add_edge("__start__", "agent")
    builder.add_conditional_edges("agent", _should_continue, {"tools": "tools", "answer": "answer"})
    builder.add_conditional_edges("tools", _after_tools_route, {"agent": "agent", "end": END})
    builder.add_edge("answer", END)
    return builder.compile()


def run_pipeline_langgraph(
    question: str,
    model: str = None,
) -> dict:
    """
    Run the Text-to-SQL pipeline using a LangGraph state graph with tools.
    Returns dict with sql (if any), data (if executed), answer, error, validation_passed.
    Retries SQL generation when validation fails up to MAX_SQL_RETRIES.
    """
    # Load data and run
    from .data_loader import load_all_tables, get_sqlite_connection
    tables = load_all_tables()
    conn = get_sqlite_connection(tables)

    initial: PipelineState = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "tables": tables,
        "conn": conn,
        "validation_retry_count": 0,
    }

    graph = build_graph()
    final = graph.invoke(initial)

    conn.close()

    # Extract result from final state (prefer explicit answer node output)
    messages = final.get("messages") or []
    answer = final.get("final_answer")
    if not answer:
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                if m.content and not m.content.strip().startswith("{") and "STRUCTURAL KNOWLEDGE" not in m.content:
                    answer = m.content
                    break
            if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                answer = m.content if isinstance(m.content, str) else str(m.content)
                break

    validation_retry_count = final.get("validation_retry_count", 0)
    final_error = None
    if validation_retry_count >= MAX_SQL_RETRIES:
        final_error = f"Validation failed after {MAX_SQL_RETRIES} retries."

    # Extract last SQL from agent's execute_sql or validate_sql tool call
    sql = None
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls or []:
                if tc.get("name") in ("execute_sql", "validate_sql"):
                    args = tc.get("args") or {}
                    if args.get("sql"):
                        sql = args["sql"]
                        break
            if sql:
                break

    return {
        "sql": sql,
        "data": None,
        "answer": answer,
        "error": final_error,
        "validation_passed": validation_retry_count < MAX_SQL_RETRIES and final_error is None,
    }
