from typing import Any, Dict, List, Tuple

import faiss

from app.config import settings
from app.metadata_store import load_metadata


def load_faiss_index(index_path: str = settings.index_path) -> faiss.Index:
    try:
        return faiss.read_index(index_path)
    except RuntimeError as error:
        raise FileNotFoundError(f"FAISS index file not found or unreadable: {index_path}") from error


def load_vector_store() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index = load_faiss_index()
    metadata = load_metadata()

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Vector store mismatch: index has {index.ntotal} vectors, "
            f"but metadata has {len(metadata)} records."
        )

    return index, metadata


def get_vector_store_stats() -> Dict[str, int]:
    index, metadata = load_vector_store()

    return {
        "index_vectors": index.ntotal,
        "metadata_count": len(metadata),
    }
