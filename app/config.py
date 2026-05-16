import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    """
    Central application configuration loaded from environment variables.
    """

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )

    postgres_db: str = os.getenv("POSTGRES_DB", "medmind_db")
    postgres_user: str = os.getenv("POSTGRES_USER", "medmind")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: str = os.getenv("POSTGRES_PORT", "5432")

    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "data/rag_index.faiss")
    metadata_path: str = os.getenv("METADATA_PATH", "data/metadata_list.json")
    bert_model_path: str = os.getenv("BERT_MODEL_PATH", "MedMind-BERT")

    def validate_openai_settings(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Please set it in .env file.")


settings = Settings()
