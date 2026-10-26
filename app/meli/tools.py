"""LangChain / LangGraph tools wrapping the Mercado Libre client."""

from typing import List

from langchain_core.tools import tool

from .client import MeliClient

_client = MeliClient()


@tool
def search_meli_products(query: str, limit: int = 5) -> str:
    """Search Mercado Libre products by free-text query. Returns a summary of matching items."""
    results = _client.search(query, limit=limit)
    if not results:
        return "No products found."

    lines = []
    for r in results:
        price = r.get("price", "N/A")
        currency = r.get("currency_id", "ARS")
        title = r.get("title", "Unknown")
        item_id = r.get("id", "")
        sold = r.get("sold_quantity", 0)
        free_ship = r.get("shipping", {}).get("free_shipping", False)
        lines.append(
            f"- [{item_id}] {title} | {currency} {price} | sold: {sold} | free_shipping: {free_ship}"
        )
    return "\n".join(lines)


@tool
def get_product_details(item_id: str) -> str:
    """Get detailed information about a specific Mercado Libre item by its ID."""
    item = _client.get_item(item_id)
    if not item:
        return f"Item {item_id} not found."

    history = item.get("price_history", [])
    hist_str = ", ".join(f"{h['date']}: {h['price']}" for h in history) if history else "N/A"

    return (
        f"Title: {item.get('title')}\n"
        f"Price: {item.get('currency_id')} {item.get('price')}\n"
        f"Condition: {item.get('condition')}\n"
        f"Sold quantity: {item.get('sold_quantity')}\n"
        f"Available: {item.get('available_quantity')}\n"
        f"Free shipping: {item.get('shipping', {}).get('free_shipping')}\n"
        f"Category: {item.get('category')}\n"
        f"Description: {item.get('description', '')[:300]}...\n"
        f"Price history: {hist_str}"
    )


@tool
def get_price_history(item_id: str) -> str:
    """Return the recent price history for a Mercado Libre item."""
    history = _client.get_price_history(item_id)
    if not history:
        return f"No price history available for {item_id}."
    return "\n".join(f"{h['date']}: {h['price']}" for h in history)


def get_meli_tools() -> List:
    """Return the list of tools to bind to the agent."""
    return [search_meli_products, get_product_details, get_price_history]
