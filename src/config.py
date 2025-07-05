"""
Configuration settings for the RAG application.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI API Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    # Embedding configuration - defaults to text-embedding-3-small for backwards compatibility
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_collection_name: str = "documents"

    # Document Processing Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Application Configuration
    upload_dir: str = "uploads"
    max_file_size: int = 50 * 1024 * 1024  # 50MB

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def validate_settings() -> bool:
    """Validate that required settings are configured."""
    if not settings.openai_api_key:
        return False
    return True


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
