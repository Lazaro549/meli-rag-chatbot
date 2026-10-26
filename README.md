# meli-rag-chatbot

**RAG-powered multi-agent chatbot for Mercado Libre sellers**

A production-oriented starter that combines:
- **Retrieval-Augmented Generation (RAG)** over product data, price history and competitor insights
- **Amazon Bedrock** (Claude / Titan) for reasoning
- **LangGraph** for multi-step agent workflows
- **Mercado Libre API** client (with mock mode for local development)

Ask natural-language questions such as:
- “What price should I set for this product?”
- “How do my competitors price similar items?”
- “Optimize this title and description for better conversion”
- “Summarize the performance of my last 30 days of ads”

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-step agents** | LangGraph workflows that plan → retrieve → reason → answer |
| **RAG pipeline** | Embeddings + vector store (FAISS) over product documents |
| **Mercado Libre tools** | Search products, get item details, price history, competition |
| **Mock mode** | Fully functional without real API keys or AWS credentials |
| **Streamlit UI** | Clean chat interface with source citations |
| **CLI** | Non-interactive mode for scripting and evaluation |
| **Evaluation hooks** | Ready to plug into `ai-evaluation-platform` style metrics |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Lazaro549/meli-rag-chatbot.git
cd meli-rag-chatbot
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys (optional – mock mode works without them)
```

| Variable | Required | Description |
|----------|----------|-------------|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` | No* | For Bedrock |
| `MELI_ACCESS_TOKEN` | No* | Mercado Libre OAuth token |
| `MELI_SITE_ID` | No | Default `MLA` (Argentina) |
| `USE_MOCK` | No | `true` (default) for local demos |

\* When keys are missing the system automatically falls back to high-quality mock data.

### 3. Run the chat UI

```bash
streamlit run app/streamlit_app.py
```

### 4. Or use the CLI

```bash
python -m app.cli "What is a good price for a used iPhone 13 in Argentina?"
```

---

## Project Structure

```
meli-rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py      # Chat UI
│   ├── cli.py                # Command-line interface
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph.py          # LangGraph definition
│   │   └── nodes.py          # Agent nodes (retrieve, reason, tools)
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── vectorstore.py
│   │   └── retriever.py
│   ├── meli/
│   │   ├── __init__.py
│   │   ├── client.py         # Real + mock Mercado Libre client
│   │   └── tools.py          # LangChain/LangGraph tools
│   ├── llm/
│   │   ├── __init__.py
│   │   └── bedrock.py        # Bedrock wrapper + mock LLM
│   └── config.py
├── data/
│   └── sample_products.json  # Seed data for RAG + mocks
├── tests/
│   └── test_basic.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────┐
│  LangGraph      │
│  (plan → act)   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌──────────────┐
│  RAG  │  │ MELI Tools   │
│Retrieve│  │ (search,     │
│ docs  │  │  prices,     │
└───┬───┘  │  competition)│
    │      └──────┬───────┘
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  Bedrock    │
    │  (Claude)   │
    └──────┬──────┘
           ▼
      Final Answer
   (+ source citations)
```

---

## Example Conversation

```
You: Analyze this product and suggest an optimal price.
     Title: "Samsung Galaxy A54 128GB"

Assistant: Based on current market data from Mercado Libre (MLA):

• Average listing price: ARS 485.000 – 520.000
• Median of top 10 sellers: ARS 499.000
• Your competitors with free shipping tend to convert better around ARS 489.000–505.000

Recommendation: List at **ARS 494.999** with free shipping.
This positions you in the competitive middle while protecting margin.

Sources:
- Product search results (mock data, 2026-10)
- Price history of similar items
```

---

## Extending the Project

- Add new tools in `app/meli/tools.py`
- Improve the graph in `app/agents/graph.py` (add reflection, human-in-the-loop, etc.)
- Swap the vector store (Chroma, Pinecone, OpenSearch)
- Plug the evaluation harness from [ai-evaluation-platform](https://github.com/Lazaro549/ai-evaluation-platform)
- Connect real Mercado Libre OAuth flow for production sellers

---

## License

MIT

---

## 💸 Donations

If you'd like to support this project:

- 🇦🇷 ARS (Argentina)  
  Alias: `lazaro.503.alaba.mp`

- 🌎 USD (Argentina only, local transfers)  
  Alias: `ahogada.duras.foca`

Built by [Lazaro Gomez Vitolo](https://github.com/Lazaro549) – AI & Robotics Developer  
Portfolio: https://lazaro549.github.io/Portfolio/
