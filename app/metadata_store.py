import json
import os
from typing import Any, Dict, List

from app.config import settings
from app.pdf_service import TextChunk


def build_metadata(chunks: List[TextChunk]) -> List[Dict[str, Any]]:
    metadata = []

    for chunk in chunks:
        metadata.append(
            {
                "id": chunk.id,
                "source": chunk.source,
                "text": chunk.text,
            }
        )

    return metadata


def save_metadata(
    metadata: List[Dict[str, Any]],
    metadata_path: str = settings.metadata_path,
) -> None:
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def load_metadata(
    metadata_path: str = settings.metadata_path,
) -> List[Dict[str, Any]]:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)
