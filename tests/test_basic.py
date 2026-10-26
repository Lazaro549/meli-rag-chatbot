"""Basic smoke tests (run with mock mode)."""

import os

# Force mock mode for tests
os.environ["USE_MOCK"] = "true"

from app.agents import run_agent
from app.meli.client import MeliClient
from app.rag import get_retriever


def test_meli_client_search():
    client = MeliClient()
    results = client.search("Samsung")
    assert len(results) > 0
    assert "title" in results[0]


def test_retriever():
    retriever = get_retriever(k=2)
    docs = retriever.invoke("Galaxy A54")
    assert len(docs) > 0


def test_agent_price_query():
    result = run_agent("What is a good price for a Samsung Galaxy A54?")
    assert "answer" in result
    assert len(result["answer"]) > 20
    # Mock should mention price-related content
    assert any(word in result["answer"].lower() for word in ["price", "ars", "recommend"])
