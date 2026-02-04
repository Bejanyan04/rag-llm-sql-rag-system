"""
Evaluation runner for the Text-to-SQL pipeline.
Benchmark and results live in the evaluation folder: benchmark from evaluation/benchmark.json,
results written to evaluation/eval_results.json.
Uses OpenAI embeddings (same as retrieval) for answer semantic similarity vs ground truth.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Project root; evaluation files live under ROOT/evaluation
ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation"
# Load .env from project root (script may be run from tools/)
load_dotenv(ROOT / ".env")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    DEFAULT_SQL_MODEL,
    DEFAULT_ANSWER_MODEL,
    FALLBACK_SQL_MODEL,
    FALLBACK_ANSWER_MODEL,
)
from src.data_loader import load_all_tables, get_sqlite_connection
from src.executor import execute_query
from src.pipeline import run_pipeline
from src.pipeline_langgraph import run_pipeline_langgraph

# Configurable threshold for answer similarity (agent answer vs expected_answer)
ANSWER_SIMILARITY_THRESHOLD = 0.75


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for exact match: lowercase, collapse whitespace."""
    if not sql:
        return ""
    return " ".join(sql.lower().split())


def _result_checksum(df: pd.DataFrame) -> str:
    """Stable string representation of a result for comparison (sort rows)."""
    if df is None or len(df) == 0:
        return ""
    key_cols = list(df.columns)
    try:
        sorted_df = df.sort_values(by=key_cols).fillna("").astype(str)
    except Exception:
        sorted_df = df.fillna("").astype(str)
    return sorted_df.to_csv(index=False)


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors (lists or arrays)."""
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(np.clip(n, -1.0, 1.0))


def _answer_similarity(agent_answer: str, expected_answer: str):
    """
    Semantic similarity between agent answer and ground truth using OpenAI embeddings.
    Returns float in [0, 1] or None if embedding unavailable (no API key or error).
    """
    if not agent_answer or not expected_answer:
        return None
    try:
        from chromadb.utils import embedding_functions
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )
        vecs = ef([agent_answer, expected_answer])
        if vecs is None or len(vecs) != 2:
            return None
        sim = _cosine_similarity(vecs[0], vecs[1])
        return (sim + 1) / 2.0  # map [-1,1] to [0,1] for readability
    except Exception:
        return None


def run_eval(
    benchmark_path: Path = None,
    output_path: Path = None,
    verbose: bool = True,
    use_langgraph: bool = False,
) -> dict:
    """
    Load benchmark, run pipeline for each question, compute metrics.
    Returns dict with valid_sql_rate, execution_accuracy, exact_match_rate, and per_item list.
    use_langgraph: if True, use run_pipeline_langgraph instead of run_pipeline.
    """
    benchmark_path = benchmark_path or EVAL_DIR / "benchmark.json"
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    tables = load_all_tables()
    conn = get_sqlite_connection(tables)

    run_fn = run_pipeline_langgraph if use_langgraph else run_pipeline
    pipeline_name = "langgraph" if use_langgraph else "normal"

    results = []
    valid_sql_count = 0
    execution_match_count = 0
    exact_sql_count = 0
    answer_similarity_count = 0
    answer_similarity_sum = 0.0
    answer_similarity_n = 0

    if verbose:
        print(f"Pipeline: {pipeline_name} | Questions: {len(benchmark)}")

    for i, item in enumerate(benchmark):
        question = item["question"]
        expected_sql = item.get("expected_sql") or ""
        expected_answer = item.get("expected_answer") or ""

        out = run_fn(question)
        generated_sql = out.get("sql")
        validation_passed = out.get("validation_passed", False)
        data = out.get("data")
        error = out.get("error")
        agent_answer = out.get("answer") or ""

        # LangGraph returns data=None; run generated_sql ourselves for execution comparison
        if data is None and generated_sql and validation_passed and conn:
            exec_result = execute_query(conn, generated_sql)
            if isinstance(exec_result, pd.DataFrame):
                data = exec_result

        if validation_passed:
            valid_sql_count += 1

        # Gold result: run expected_sql if provided
        gold_df = None
        if expected_sql:
            gold_result = execute_query(conn, expected_sql)
            if isinstance(gold_result, pd.DataFrame):
                gold_df = gold_result

        # Execution accuracy: compare result to gold (row count + checksum)
        execution_match = False
        if gold_df is not None and data is not None and isinstance(data, pd.DataFrame):
            execution_match = _result_checksum(data) == _result_checksum(gold_df)

        if execution_match:
            execution_match_count += 1

        # Exact SQL match (optional)
        exact_match = False
        if expected_sql and generated_sql:
            if _normalize_sql(generated_sql) == _normalize_sql(expected_sql):
                exact_sql_count += 1
                exact_match = True

        # Answer semantic similarity vs ground truth
        answer_sim = None
        if expected_answer:
            answer_sim = _answer_similarity(agent_answer, expected_answer)
            if answer_sim is not None:
                answer_similarity_n += 1
                answer_similarity_sum += answer_sim
                if answer_sim >= ANSWER_SIMILARITY_THRESHOLD:
                    answer_similarity_count += 1

        results.append({
            "index": i,
            "question": question[:60] + "..." if len(question) > 60 else question,
            "validation_passed": validation_passed,
            "execution_match": execution_match,
            "exact_sql_match": exact_match,
            "answer_similarity": answer_sim,
            "answer_similarity_pass": answer_sim is not None and answer_sim >= ANSWER_SIMILARITY_THRESHOLD,
            "error": error,
        })
        if verbose:
            sim_str = f" sim={answer_sim:.2f}" if answer_sim is not None else ""
            print(f"  [{i+1}/{len(benchmark)}] valid={validation_passed} exec_match={execution_match}{sim_str}")
            if not validation_passed and error:
                print(f"      error: {error[:200]}{'...' if len(error) > 200 else ''}")
            if not validation_passed and generated_sql:
                print(f"      sql:   {generated_sql[:150]}{'...' if len(generated_sql) > 150 else ''}")

    conn.close()

    n = len(benchmark)
    mean_answer_similarity = answer_similarity_sum / answer_similarity_n if answer_similarity_n else None
    metrics = {
        "valid_sql_rate": valid_sql_count / n if n else 0,
        "execution_accuracy": execution_match_count / n if n else 0,
        "exact_match_rate": exact_sql_count / n if n else 0,
        "answer_similarity_rate": answer_similarity_count / n if n else 0,
        "mean_answer_similarity": round(mean_answer_similarity, 4) if mean_answer_similarity is not None else None,
        "answer_similarity_threshold": ANSWER_SIMILARITY_THRESHOLD,
        "n": n,
        "pipeline": pipeline_name,
        "models": {
            "sql_model": DEFAULT_SQL_MODEL,
            "answer_model": DEFAULT_ANSWER_MODEL,
            "fallback_sql_model": FALLBACK_SQL_MODEL,
            "fallback_answer_model": FALLBACK_ANSWER_MODEL,
        },
        "per_item": results,
    }

    if verbose:
        print("\n--- Summary ---")
        print(f"Valid SQL rate:       {metrics['valid_sql_rate']:.2%} ({valid_sql_count}/{n})")
        print(f"Execution accuracy:  {metrics['execution_accuracy']:.2%} ({execution_match_count}/{n})")
        print(f"Exact SQL match:     {metrics['exact_match_rate']:.2%} ({exact_sql_count}/{n})")
        print(f"Answer similarity:   {metrics['answer_similarity_rate']:.2%} (threshold={ANSWER_SIMILARITY_THRESHOLD})")
        if mean_answer_similarity is not None:
            print(f"Mean answer sim:      {mean_answer_similarity:.4f}")
        print("Models:", metrics["models"])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        to_write = {k: v for k, v in metrics.items() if k != "per_item"}
        with open(output_path, "w") as f:
            json.dump(to_write, f, indent=2)
        if verbose:
            print(f"Metrics written to {output_path}")

    return metrics


def export_test_set_csv(
    test_set_path: Path = None,
    output_csv_path: Path = None,
    verbose: bool = True,
    use_langgraph: bool = False,
) -> Path:
    """
    Run the pipeline on the 15 test-set questions and write a CSV with columns:
    question, sql, agent_final_response.
    """
    test_set_path = test_set_path or EVAL_DIR / "test_set.json"
    output_csv_path = output_csv_path or EVAL_DIR / "test_set_results.csv"
    with open(test_set_path) as f:
        test_set = json.load(f)

    run_fn = run_pipeline_langgraph if use_langgraph else run_pipeline
    if verbose:
        print(f"Pipeline: {'langgraph' if use_langgraph else 'normal'}")

    rows = []
    for i, item in enumerate(test_set):
        question = item["question"]
        if verbose:
            print(f"  Test set [{i+1}/{len(test_set)}]: {question[:50]}...")
        out = run_fn(question)
        rows.append({
            "question": question,
            "sql": out.get("sql") or "",
            "agent_final_response": out.get("answer") or "",
        })

    df = pd.DataFrame(rows)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    if verbose:
        print(f"Wrote {len(rows)} rows to {output_csv_path}")
    return output_csv_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation or export test set CSV.")
    parser.add_argument(
        "--test-set",
        action="store_true",
        help="Run on 15 test questions (evaluation/test_set.json) instead of benchmark.json.",
    )
    parser.add_argument(
        "--langgraph",
        action="store_true",
        help="Use LangGraph pipeline instead of normal pipeline.",
    )
    parser.add_argument(
        "--export-test-csv",
        action="store_true",
        help="Run pipeline on test_set.json and write evaluation/test_set_results.csv (question, sql, agent_final_response)",
    )
    parser.add_argument("--no-verbose", action="store_true", help="Reduce output.")
    args = parser.parse_args()

    verbose = not args.no_verbose
    use_langgraph = args.langgraph

    if args.export_test_csv:
        export_test_set_csv(verbose=verbose, use_langgraph=use_langgraph)
    elif args.test_set:
        run_eval(
            benchmark_path=EVAL_DIR / "test_set.json",
            output_path=EVAL_DIR / "test_set_eval_results.json",
            verbose=verbose,
            use_langgraph=use_langgraph,
        )
    else:
        run_eval(
            output_path=EVAL_DIR / "eval_results.json",
            verbose=verbose,
            use_langgraph=use_langgraph,
        )
