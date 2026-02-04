"""
Retrieval for Text-to-SQL: schema chunks and few-shot (question, SQL) examples.
Uses vector search (Chroma + OpenAI embeddings) to return relevant schema subset and examples.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .domain_knowledge import TABLE_SEMANTICS, FEW_SHOT_EXAMPLES

# Optional Chroma; fallback to no retrieval if not installed
try:
    import chromadb
    from chromadb.utils import embedding_functions
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


def _get_embedding_function():
    """OpenAI embedding function for Chroma (requires OPENAI_API_KEY)."""
    if not _CHROMA_AVAILABLE:
        return None
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=__import__("os").environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )


def _schema_chunks(tables: dict) -> List[Dict[str, Any]]:
    """Build one text chunk per table for schema retrieval."""
    chunks = []
    for table_name, df in tables.items():
        safe = table_name.replace(" ", "_")
        cols = ", ".join(str(c) for c in df.columns)
        sem = TABLE_SEMANTICS.get(safe, "")
        text = f"Table: {safe}. Columns: {cols}. Semantics: {sem}. Row count: {len(df)}."
        chunks.append({"id": safe, "text": text, "table_name": safe})
    return chunks


def build_schema_index(tables: dict, persist_path: Optional[Path] = None):
    """
    Build Chroma collection for schema chunks. Returns collection or None if Chroma unavailable.
    """
    if not _CHROMA_AVAILABLE:
        return None
    client = chromadb.EphemeralClient() if not persist_path else chromadb.PersistentClient(path=str(persist_path))
    ef = _get_embedding_function()
    if ef is None:
        return None
    collection = client.get_or_create_collection(
        name="schema",
        embedding_function=ef,
        metadata={"description": "Schema chunks per table"},
    )
    chunks = _schema_chunks(tables)
    if not chunks:
        return collection
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"table_name": c["table_name"]} for c in chunks],
    )
    return collection


def build_few_shot_index(
    examples: Optional[List[Dict[str, Any]]] = None,
    examples_path: Optional[Path] = None,
    persist_path: Optional[Path] = None,
):
    """
    Build Chroma collection for few-shot (question, sql) examples.
    examples or examples_path (JSON list of {question, sql}) or FEW_SHOT_EXAMPLES used.
    Returns collection or None if Chroma unavailable.
    """
    if not _CHROMA_AVAILABLE:
        return None
    if examples is None and examples_path:
        with open(examples_path) as f:
            examples = json.load(f)
    if examples is None:
        examples = FEW_SHOT_EXAMPLES
    if not examples:
        return None

    client = chromadb.EphemeralClient() if not persist_path else chromadb.PersistentClient(path=str(persist_path))
    ef = _get_embedding_function()
    if ef is None:
        return None
    collection = client.get_or_create_collection(
        name="few_shot",
        embedding_function=ef,
        metadata={"description": "Few-shot question-SQL pairs"},
    )
    ids = [f"ex_{i}" for i in range(len(examples))]
    documents = [ex.get("question", "") for ex in examples]
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=[{"question": ex.get("question", ""), "sql": ex.get("sql", "")} for ex in examples],
    )
    return collection


def retrieve_relevant_tables(
    question: str,
    tables: dict,
    top_k: int = 5,
    schema_collection=None,
) -> List[str]:
    """
    Return list of relevant table names for the question.
    If schema_collection is None or Chroma unavailable, returns all table names (no retrieval).
    """
    if schema_collection is None:
        return list(tables.keys())
    try:
        results = schema_collection.query(
            query_texts=[question],
            n_results=min(top_k, len(tables)),
            include=["metadatas"],
        )
        if results and results["metadatas"] and results["metadatas"][0]:
            return [m["table_name"] for m in results["metadatas"][0]]
    except Exception:
        pass
    return list(tables.keys())


def retrieve_few_shot_examples(
    question: str,
    top_k: int = 3,
    few_shot_collection=None,
    fallback_examples: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Return top-k few-shot (question, sql) examples most similar to the question.
    If few_shot_collection is None or Chroma unavailable, returns fallback_examples or FEW_SHOT_EXAMPLES.
    """
    if few_shot_collection is None:
        return fallback_examples if fallback_examples is not None else FEW_SHOT_EXAMPLES
    try:
        results = few_shot_collection.query(
            query_texts=[question],
            n_results=min(top_k, few_shot_collection.count()),
            include=["metadatas"],
        )
        if results and results["metadatas"] and results["metadatas"][0]:
            return [
                {"question": m["question"], "sql": m["sql"]}
                for m in results["metadatas"][0]
            ]
    except Exception:
        pass
    return fallback_examples if fallback_examples is not None else FEW_SHOT_EXAMPLES


# Evaluation folder: benchmark and few-shot examples live here (same as evaluation data)
EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"


def load_few_shot_examples_from_file(examples_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load few-shot (question, sql) examples from file. Prefer evaluation/few_shot_examples.json,
    then data/few_shot_examples.json; if neither exists, return FEW_SHOT_EXAMPLES from code.
    """
    root = Path(__file__).resolve().parent.parent
    if examples_path is None:
        examples_path = root / "evaluation" / "few_shot_examples.json"
    if not examples_path.exists():
        examples_path = root / "data" / "few_shot_examples.json"
    if examples_path.exists():
        with open(examples_path) as f:
            return json.load(f)
    return FEW_SHOT_EXAMPLES


def get_retrieval_indices(tables: dict, data_dir: Optional[Path] = None):
    """
    Build and return (schema_collection, few_shot_collection) for use in retrieval.
    Few-shot examples are loaded from evaluation/few_shot_examples.json (same as evaluation data),
    then data/few_shot_examples.json, then FEW_SHOT_EXAMPLES from code.
    """
    schema_coll = build_schema_index(tables)
    examples = load_few_shot_examples_from_file()
    few_shot_coll = build_few_shot_index(examples=examples) if examples else None
    return schema_coll, few_shot_coll
