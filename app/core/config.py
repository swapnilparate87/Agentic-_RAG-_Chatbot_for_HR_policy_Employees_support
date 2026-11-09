from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    app_name: str = "HR Policy Assistant"

    # Providers
    nvidia_api_key: str = ""
    nvidia_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    nvidia_eval_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    nvidia_reasoning: bool = False  # agent model "thinking" mode; the eval judge always reasons
    nvidia_max_tokens: int = 8192  # includes reasoning tokens
    tavily_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "hr-policy-rag"
    pinecone_namespace: str = "hr-kb"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval and agent behaviour
    top_k: int = 4
    max_retries: int = 1
    web_max_results: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 120

    # Security and storage
    admin_api_key: str = "change-me"
    audit_db_path: Path = BASE_DIR / "data" / "audit.db"
    upload_dir: Path = BASE_DIR / "uploads"
    knowledge_base_dir: Path = BASE_DIR / "data" / "sample_kb"


@lru_cache
def get_settings() -> Settings:
    return Settings()
