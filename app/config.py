import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.openai_embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )

        self.postgres_db = os.getenv("POSTGRES_DB", "medmind_db")
        self.postgres_user = os.getenv("POSTGRES_USER", "medmind")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD", "")
        self.postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        self.postgres_port = os.getenv("POSTGRES_PORT", "5432")

        self.medical_pdf_dir = os.getenv("MEDICAL_PDF_DIR", "data/pdfs")
        self.faiss_index_path = os.getenv("FAISS_INDEX_PATH", "data/rag_index.faiss")
        self.metadata_path = os.getenv("METADATA_PATH", "data/metadata_list.json")
        self.bert_model_path = os.getenv("BERT_MODEL_PATH", "models/MedMind-BERT")

        self.chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        self.rag_top_k = int(os.getenv("RAG_TOP_K", "3"))

        self.cors_origins = os.getenv("CORS_ORIGINS", "*")

    def validate_openai_settings(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Please set it in .env file.")


settings = Settings()
