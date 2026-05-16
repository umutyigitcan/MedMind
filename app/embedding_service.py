from typing import List

import faiss
import numpy as np
from openai import OpenAI

from app.config import settings


client = OpenAI(api_key=settings.openai_api_key)


def create_embedding(text: str) -> List[float]:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    settings.validate_openai_settings()

    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=text.strip(),
    )

    return response.data[0].embedding


def create_embeddings(texts: List[str]) -> List[List[float]]:
    cleaned_texts = [text.strip() for text in texts if text and text.strip()]

    if not cleaned_texts:
        raise ValueError("Text list cannot be empty.")

    settings.validate_openai_settings()

    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=cleaned_texts,
    )

    return [item.embedding for item in response.data]


def embeddings_to_numpy(embeddings: List[List[float]]) -> np.ndarray:
    if not embeddings:
        raise ValueError("Embeddings cannot be empty.")

    return np.array(embeddings, dtype="float32")


def normalize_embedding_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        raise ValueError("Embedding matrix cannot be empty.")

    faiss.normalize_L2(matrix)
    return matrix


def create_normalized_embedding_matrix(texts: List[str]) -> np.ndarray:
    embeddings = create_embeddings(texts)
    matrix = embeddings_to_numpy(embeddings)

    return normalize_embedding_matrix(matrix)


def create_normalized_query_embedding(text: str) -> np.ndarray:
    embedding = create_embedding(text)
    matrix = embeddings_to_numpy([embedding])

    return normalize_embedding_matrix(matrix)
