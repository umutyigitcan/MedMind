import faiss

from app.config import settings
from app.embedding_service import create_normalized_embedding_matrix
from app.metadata_store import build_metadata, save_metadata
from app.pdf_service import load_pdf_documents, split_documents_into_chunks


def build_faiss_index(embedding_matrix):
    if embedding_matrix.size == 0:
        raise ValueError("Embedding matrix cannot be empty.")

    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    return index


def save_faiss_index(index, index_path: str = settings.faiss_index_path) -> None:
    faiss.write_index(index, index_path)


def build_and_save_index() -> int:
    documents = load_pdf_documents(settings.pdf_data_dir)

    if not documents:
        raise RuntimeError(
            f"No medical PDF files found. Add sample PDFs under {settings.pdf_data_dir}."
        )

    chunks = split_documents_into_chunks(
        documents=documents,
        chunk_size=settings.chunk_size,
    )

    if not chunks:
        raise RuntimeError("No text chunks could be created from the PDF files.")

    texts = [chunk.text for chunk in chunks]
    embedding_matrix = create_normalized_embedding_matrix(texts)

    index = build_faiss_index(embedding_matrix)
    save_faiss_index(index)

    metadata = build_metadata(chunks)
    save_metadata(metadata)

    return len(chunks)


if __name__ == "__main__":
    chunk_count = build_and_save_index()
    print(f"FAISS index and metadata created successfully with {chunk_count} chunks.")
