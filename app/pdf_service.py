import os
from dataclasses import dataclass
from typing import List

import fitz

from app.config import settings


@dataclass
class PdfDocument:
    filename: str
    text: str


@dataclass
class TextChunk:
    id: int
    source: str
    text: str


def extract_text_from_pdf(pdf_path: str) -> str:
    text_parts = []

    with fitz.open(pdf_path) as document:
        for page in document:
            page_text = page.get_text()
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts).strip()


def load_pdf_documents(pdf_dir: str = settings.pdf_dir) -> List[PdfDocument]:
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    documents = []

    for filename in sorted(os.listdir(pdf_dir)):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_dir, filename)
        text = extract_text_from_pdf(pdf_path)

        if text:
            documents.append(
                PdfDocument(
                    filename=filename,
                    text=text,
                )
            )

    return documents


def split_text_into_chunks(text: str, chunk_size: int = settings.chunk_size) -> List[str]:
    cleaned_text = " ".join(text.split())

    if not cleaned_text:
        return []

    chunks = []

    for start in range(0, len(cleaned_text), chunk_size):
        chunk = cleaned_text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def split_documents_into_chunks(
    documents: List[PdfDocument],
    chunk_size: int = settings.chunk_size,
) -> List[TextChunk]:
    chunks = []
    chunk_id = 0

    for document in documents:
        text_chunks = split_text_into_chunks(document.text, chunk_size=chunk_size)

        for chunk in text_chunks:
            chunks.append(
                TextChunk(
                    id=chunk_id,
                    source=document.filename,
                    text=chunk,
                )
            )
            chunk_id += 1

    return chunks
