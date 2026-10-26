"""Retriever factory."""

from functools import lru_cache

from langchain_core.retrievers import BaseRetriever

from .vectorstore import build_or_load_vectorstore


@lru_cache
def get_retriever(k: int = 4) -> BaseRetriever:
    vs = build_or_load_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})
