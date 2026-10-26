"""LLM factory: real Amazon Bedrock or high-quality mock."""

from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun

from app.config import get_settings


class MockChatModel(BaseChatModel):
    """Deterministic mock that produces realistic seller-oriented answers."""

    @property
    def _llm_type(self) -> str:
        return "mock-meli-seller"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        user_content = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_content = m.content
                break

        lower = user_content.lower()

        if "price" in lower or "precio" in lower:
            answer = (
                "Based on current Mercado Libre (MLA) market data:\n\n"
                "• Average listing price for similar items: ARS 480.000 – 520.000\n"
                "• Median of competitive sellers with free shipping: ARS 494.999\n"
                "• Recent downward trend of ~5–7% in the last 30 days\n\n"
                "**Recommendation:** List at **ARS 494.999** with free shipping. "
                "This positions you competitively while protecting margin.\n\n"
                "Sources: sample product catalog + price history (mock mode)."
            )
        elif "title" in lower or "título" in lower or "optimize" in lower or "optimiz" in lower:
            answer = (
                "Optimized title suggestion:\n\n"
                "**Samsung Galaxy A54 5G 128GB Negro | Libre | Garantía Oficial**\n\n"
                "Why this works:\n"
                "- Includes key search terms (brand + model + storage + color)\n"
                "- Adds trust signals (Libre / Garantía Oficial)\n"
                "- Keeps length under Mercado Libre recommended limits\n\n"
                "Description tip: Lead with the strongest benefit in the first 2 lines."
            )
        elif "compet" in lower or "competencia" in lower:
            answer = (
                "Competitor snapshot (MLA):\n\n"
                "| Seller | Price | Free Shipping | Sold |\n"
                "|--------|-------|---------------|------|\n"
                "| TECHSTORE_AR | 489.999 | Yes | 342 |\n"
                "| APPLE_RESELLER | 620.000 | No | 89 |\n"
                "| AUDIO_PRO | 389.999 | Yes | 156 |\n\n"
                "Insight: Sellers offering free shipping convert significantly better "
                "even when priced slightly higher."
            )
        else:
            answer = (
                "I can help you with:\n"
                "• Optimal pricing recommendations\n"
                "• Competitor analysis\n"
                "• Title & description optimization\n"
                "• Product performance insights\n\n"
                "Try asking something like: *What price should I set for a Samsung Galaxy A54?*\n\n"
                "(Running in mock mode – connect real Bedrock + MELI credentials for live data.)"
            )

        message = AIMessage(content=answer)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


def get_llm() -> BaseChatModel:
    """Return a chat model (Bedrock or mock)."""
    settings = get_settings()

    if settings.use_mock or not settings.has_bedrock_credentials:
        return MockChatModel()

    try:
        from langchain_aws import ChatBedrock

        return ChatBedrock(
            model_id=settings.bedrock_model_id,
            region_name=settings.aws_region,
            model_kwargs={"temperature": 0.2, "max_tokens": 1024},
        )
    except Exception:
        # Fallback if boto3 / credentials fail at runtime
        return MockChatModel()
