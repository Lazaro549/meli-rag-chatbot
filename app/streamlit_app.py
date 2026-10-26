"""Streamlit chat interface for the Mercado Libre RAG chatbot."""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from app.agents import run_agent
from app.config import get_settings

st.set_page_config(
    page_title="MELI RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Mercado Libre Seller Assistant")
st.caption("RAG + multi-agent chatbot powered by Amazon Bedrock & LangGraph")

settings = get_settings()
if settings.use_mock:
    st.info("Running in **mock mode** – no real API keys required. Great for demos.")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about pricing, competition, titles..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_agent(prompt, chat_history=st.session_state.langchain_history)
            answer = result["answer"]
            st.markdown(answer)

            # Optional: show retrieved context in expander
            if result.get("context"):
                with st.expander("Retrieved context"):
                    st.text(result["context"][:2000] + ("..." if len(result["context"]) > 2000 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.langchain_history.append(HumanMessage(content=prompt))
    st.session_state.langchain_history.append(AIMessage(content=answer))

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This assistant helps Mercado Libre sellers with:

        - Optimal pricing recommendations
        - Competitor analysis
        - Title & description optimization
        - Product performance insights

        Built with **LangGraph**, **Amazon Bedrock** and a **RAG** pipeline.
        """
    )
    st.divider()
    st.markdown("**Mode:** " + ("Mock" if settings.use_mock else "Live"))
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.langchain_history = []
        st.rerun()
