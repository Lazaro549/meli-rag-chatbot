"""LangGraph agent definition."""

from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.meli.tools import get_meli_tools
from .nodes import AgentState, reason_node, retrieve_node, should_continue


def build_graph():
    """Compile the multi-step RAG + tools agent."""
    tools = get_meli_tools()
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )
    graph.add_edge("tools", "reason")

    return graph.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(query: str, chat_history: list | None = None) -> Dict[str, Any]:
    """Run the agent on a user query and return the final answer + state."""
    graph = get_graph()

    messages = []
    if chat_history:
        messages.extend(chat_history)
    messages.append(HumanMessage(content=query))

    result = graph.invoke({"messages": messages, "context": ""})

    final_message = result["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)

    return {
        "answer": answer,
        "messages": result["messages"],
        "context": result.get("context", ""),
    }
