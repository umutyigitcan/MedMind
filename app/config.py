import os
from typing import List

from dotenv import load_dotenv


load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        self.postgres_db = os.getenv("POSTGRES_DB", "medmind_db")
        self.postgres_user = os.getenv("POSTGRES_USER", "medmind")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD", "")
        self.postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        self.postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))

        self.pdf_dir = os.getenv("PDF_DIR", "data/pdfs")
        self.pdf_data_dir = os.getenv("PDF_DATA_DIR", self.pdf_dir)

        self.index_path = os.getenv("INDEX_PATH", "data/rag_index.faiss")
        self.faiss_index_path = os.getenv("FAISS_INDEX_PATH", self.index_path)

        self.metadata_path = os.getenv("METADATA_PATH", "data/metadata_list.json")
        self.metadata_file_path = os.getenv("METADATA_FILE_PATH", self.metadata_path)

        self.chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
        self.rag_top_k = int(os.getenv("RAG_TOP_K", "3"))

        self.intent_model_path = os.getenv("INTENT_MODEL_PATH", "MedMind-BERT")

        self.cors_origins = self._parse_cors_origins(
            os.getenv(
                "CORS_ORIGINS",
                "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
            )
        )

    def _parse_cors_origins(self, value: str) -> List[str]:
        return [origin.strip() for origin in value.split(",") if origin.strip()]

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def validate_openai_settings(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Please set it in .env file.")


settings = Settings()
