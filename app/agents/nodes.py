"""LangGraph nodes."""

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages

from app.llm import get_llm
from app.meli.tools import get_meli_tools
from app.rag import get_retriever


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str


SYSTEM_PROMPT = """You are an expert Mercado Libre seller assistant focused on the Argentine market (MLA).
You help sellers with pricing strategy, competitor analysis, title/description optimization and product insights.

Always:
- Be concrete and data-driven.
- Prefer ARS prices and local context.
- Cite the sources you used (product IDs, price history, etc.).
- If data is limited, say so clearly and still give the best recommendation possible.
"""


def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant product documents from the vector store."""
    last_user = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m.content
            break

    retriever = get_retriever(k=4)
    docs = retriever.invoke(last_user)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return {"context": context}


def reason_node(state: AgentState) -> dict:
    """Call the LLM with retrieved context + conversation."""
    llm = get_llm()
    tools = get_meli_tools()

    # Bind tools when the model supports it (real Bedrock does; mock does not)
    try:
        llm_with_tools = llm.bind_tools(tools)
    except Exception:
        llm_with_tools = llm

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if state.get("context"):
        messages.append(
            SystemMessage(
                content=f"Relevant product context from the knowledge base:\n\n{state['context']}"
            )
        )

    messages.extend(state["messages"])

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Decide whether to call tools or finish."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"
