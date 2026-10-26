"""Application configuration."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Mode
    use_mock: bool = True

    # AWS / Bedrock
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"

    # Mercado Libre
    meli_access_token: Optional[str] = None
    meli_site_id: str = "MLA"
    meli_client_id: Optional[str] = None
    meli_client_secret: Optional[str] = None

    # RAG
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_path: str = "./data/faiss_index"

    @property
    def has_bedrock_credentials(self) -> bool:
        return bool(self.aws_access_key_id and self.aws_secret_access_key)

    @property
    def has_meli_token(self) -> bool:
        return bool(self.meli_access_token)


@lru_cache
def get_settings() -> Settings:
    return Settings()
