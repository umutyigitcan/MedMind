import os
from dataclasses import dataclass
from typing import List

import fitz

from app.config import settings


@dataclass
class MedicalDocument:
    filename: str
    text: str


@dataclass
class TextChunk:
    id: int
    source: str
    text: str


def extract_pdf_text(file_path: str) -> str:
    text_parts: List[str] = []

    with fitz.open(file_path) as document:
        for page in document:
            page_text = page.get_text()
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts).strip()


def load_medical_pdfs(pdf_dir: str = None) -> List[MedicalDocument]:
    target_dir = pdf_dir or settings.medical_pdf_dir

    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"PDF directory not found: {target_dir}")

    documents: List[MedicalDocument] = []

    for filename in sorted(os.listdir(target_dir)):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(target_dir, filename)
        text = extract_pdf_text(file_path)

        if not text:
            continue

        documents.append(
            MedicalDocument(
                filename=filename,
                text=text,
            )
        )

    return documents


def chunk_text(text: str, chunk_size: int = None) -> List[str]:
    size = chunk_size or settings.chunk_size
    cleaned_text = text.strip()

    if not cleaned_text:
        return []

    return [
        cleaned_text[index:index + size]
        for index in range(0, len(cleaned_text), size)
    ]


def build_chunks_from_documents(documents: List[MedicalDocument]) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    chunk_id = 0

    for document in documents:
        for chunk in chunk_text(document.text):
            chunks.append(
                TextChunk(
                    id=chunk_id,
                    source=document.filename,
                    text=chunk,
                )
            )
            chunk_id += 1

    return chunks
