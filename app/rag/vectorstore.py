"""FAISS vector store helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import get_settings
from .embeddings import get_embeddings

SAMPLE_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_products.json"


def _load_documents() -> List[Document]:
    if not SAMPLE_DATA_PATH.exists():
        return []

    products = json.loads(SAMPLE_DATA_PATH.read_text(encoding="utf-8"))
    docs: List[Document] = []
    for p in products:
        content = (
            f"Title: {p.get('title')}\n"
            f"Price: {p.get('currency_id')} {p.get('price')}\n"
            f"Condition: {p.get('condition')}\n"
            f"Category: {p.get('category')}\n"
            f"Sold: {p.get('sold_quantity')}\n"
            f"Description: {p.get('description', '')}\n"
            f"Attributes: {p.get('attributes', {})}\n"
            f"Price history: {p.get('price_history', [])}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "id": p.get("id"),
                    "title": p.get("title"),
                    "price": p.get("price"),
                    "source": "sample_products",
                },
            )
        )
    return docs


def build_or_load_vectorstore() -> FAISS:
    """Build a FAISS index from sample data or load an existing one."""
    settings = get_settings()
    path = Path(settings.vector_store_path)
    embeddings = get_embeddings()

    if path.exists() and any(path.iterdir()):
        return FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    docs = _load_documents()
    if not docs:
        # Create a tiny dummy index so the system still starts
        docs = [Document(page_content="No product data available.", metadata={})]

    vs = FAISS.from_documents(docs, embeddings)
    path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(path))
    return vs
