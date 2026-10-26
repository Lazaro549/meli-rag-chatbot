"""Embedding model factory."""

from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import get_settings


def get_embeddings():
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
