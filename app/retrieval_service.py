from typing import Any, Dict, List

import numpy as np

from app.embedding_service import create_normalized_query_embedding
from app.vector_store import load_vector_store


def search_medical_context(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    index, metadata = load_vector_store()

    if index.ntotal == 0:
        raise LookupError("FAISS index is empty.")

    query_embedding = create_normalized_query_embedding(query)

    scores, indices = index.search(
        np.asarray(query_embedding, dtype="float32"),
        top_k,
    )

    results: List[Dict[str, Any]] = []

    for score, index_id in zip(scores[0], indices[0]):
        if index_id < 0:
            continue

        if index_id >= len(metadata):
            continue

        record = metadata[index_id]

        results.append(
            {
                "chunk_id": record.get("id", index_id),
                "source": record.get("source"),
                "text": record.get("text"),
                "similarity_score": float(score),
            }
        )

    if not results:
        raise LookupError("No relevant medical context found.")

    return results


def build_context_text(results: List[Dict[str, Any]]) -> str:
    context_parts = []

    for result in results:
        chunk_id = result.get("chunk_id")
        source = result.get("source") or "unknown"
        text = result.get("text") or ""

        context_parts.append(
            f"[Chunk {chunk_id} | Source: {source}]\n{text}"
        )

    return "\n\n".join(context_parts)


def retrieve_medical_context(query: str, top_k: int = 3) -> Dict[str, Any]:
    results = search_medical_context(query=query, top_k=top_k)
    context_text = build_context_text(results)

    return {
        "query": query,
        "context": context_text,
        "matches": results,
    }
