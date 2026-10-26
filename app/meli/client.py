"""Mercado Libre API client with mock fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from app.config import get_settings

SAMPLE_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_products.json"


class MeliClient:
    """Thin client for Mercado Libre public + authenticated endpoints."""

    BASE_URL = "https://api.mercadolibre.com"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._sample: List[Dict[str, Any]] = []
        if SAMPLE_DATA_PATH.exists():
            self._sample = json.loads(SAMPLE_DATA_PATH.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if self.settings.use_mock or not self.settings.has_meli_token:
            return self._mock_search(query, limit)
        return self._real_search(query, limit)

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        if self.settings.use_mock or not self.settings.has_meli_token:
            return self._mock_get_item(item_id)
        return self._real_get_item(item_id)

    def get_price_history(self, item_id: str) -> List[Dict[str, Any]]:
        item = self.get_item(item_id)
        if not item:
            return []
        return item.get("price_history", [])

    # ------------------------------------------------------------------
    # Mock implementations
    # ------------------------------------------------------------------
    def _mock_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        q = query.lower()
        results = []
        for p in self._sample:
            text = f"{p['title']} {p.get('description', '')} {p.get('category', '')}".lower()
            if any(term in text for term in q.split()):
                results.append(p)
        return results[:limit] or self._sample[:limit]

    def _mock_get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        for p in self._sample:
            if p["id"] == item_id:
                return p
        # fuzzy fallback
        for p in self._sample:
            if item_id.lower() in p["title"].lower():
                return p
        return self._sample[0] if self._sample else None

    # ------------------------------------------------------------------
    # Real API implementations (minimal)
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.meli_access_token:
            headers["Authorization"] = f"Bearer {self.settings.meli_access_token}"
        return headers

    def _real_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        site = self.settings.meli_site_id
        url = f"{self.BASE_URL}/sites/{site}/search"
        params = {"q": query, "limit": limit}
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, params=params, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
                return data.get("results", [])
        except Exception:
            return self._mock_search(query, limit)

    def _real_get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        url = f"{self.BASE_URL}/items/{item_id}"
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return self._mock_get_item(item_id)
