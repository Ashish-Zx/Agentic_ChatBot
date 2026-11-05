# Agentic ChatBot

A production-ready agentic chatbot built with **LangGraph**, **Streamlit**, and **Inception Labs mercury-3** (free model). Features multi-turn memory, tool use, RAG over uploaded documents, human-in-the-loop trade approval, and an on-demand **Deep Research** mode powered by a 3-agent web research pipeline.

---

## Features

| Feature | Details |
|---|---|
| 💬 **Multi-turn chat** | Persistent conversation threads backed by SQLite (LangGraph checkpointer) |
| 🧠 **Tool-calling agent** | Web search (Tavily), weather, stock prices, calculator |
| 📂 **Dynamic RAG** | Upload PDF / TXT / MD files → FAISS vector store → agent answers from your docs |
| 🔒 **Human-in-the-Loop** | Stock purchases require supervisor approval before execution |
| 🔬 **Deep Research mode** | 3-agent pipeline: Planner → Search (Tavily) → Writer — returns a full markdown report in chat |
| 🗂️ **Multi-thread sidebar** | Switch between past conversations; pending approvals are highlighted |

---

## Project structure

```
Agentic_ChatBot/
├── app.py                      # Streamlit UI (entry point)
├── agentic_chatbot_backend.py  # LangGraph graph, tools, RAG engine
├── deep_research.py            # 3-agent deep research pipeline
├── .env                        # API keys (never commit — see .env.example)
├── .env.example                # Required environment variable names
├── .gitignore
├── chatbot.db                  # SQLite conversation store (auto-created)
├── faiss_rag_db/               # FAISS vector index (auto-created on upload)
└── uploaded_documents/         # Staging folder for RAG uploads
```

---

## Quickstart

### 1. Clone & create a virtual environment

```bash
git clone <repo-url>
cd Agentic_ChatBot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install openai-agents          # OpenAI Agents SDK (deep research pipeline)
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
# Inception Labs (free model — mercury-3)
INCEPTION_API_KEY=your_key

# Tavily (web search — free tier)
TAVILY_API_KEY=your_key

# Google Generative AI (embeddings for RAG)
GOOGLE_API_KEY=your_key

# LangSmith (optional tracing)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=agentic-chatbot-project

# Stock prices (Alpha Vantage — free tier; demo prices used as fallback)
STOCK_API_KEY=your_key

# Weather (OpenWeatherMap — free tier)
OPEN_WEATHER_API_KEY=your_key

# Deep research — number of search queries the planner generates
HOW_MANY_SEARCHES=5
```

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Tools available to the agent

| Tool | Description |
|---|---|
| `search_tool` | Tavily web search (up to 5 results, advanced depth) |
| `calculator` | Safe `eval` for math expressions |
| `get_weather` | Current weather via OpenWeatherMap |
| `get_stock_price` | Live price via Alpha Vantage (demo fallback for AAPL, TSLA, etc.) |
| `buy_stocks` | Triggers a **Human-in-the-Loop** interrupt before executing any trade |
| `rag_tool` | Semantic search over your uploaded documents (FAISS + Gemini embeddings) |

---

## Deep Research mode

Enable the **Deep research** toggle in the sidebar to activate a 3-agent pipeline:

```
User query
    │
    ▼
Planner Agent  — produces N targeted search queries (HOW_MANY_SEARCHES)
    │
    ▼  (all searches run concurrently)
Search Agent × N  — Tavily search → 2-3 paragraph summary per query
    │
    ▼
Writer Agent  — synthesises all summaries into a detailed markdown report
    │
    ▼
Report rendered directly in chat
```

All three agents use **Inception Labs mercury-3** (free, no OpenAI credits needed).

---

## RAG (Knowledge base)

1. Click **Upload documents** in the sidebar and select PDF, TXT, or MD files.
2. Click **Index documents** — files are chunked (1 000 tokens, 200 overlap) and stored in a local FAISS index.
3. Ask the agent anything about your documents — it uses the `rag_tool` automatically.
4. Use the **Replace existing index** toggle to overwrite instead of append.
5. Click **Clear knowledge base** to wipe the index.

---

## Human-in-the-Loop (stock trades)

When you ask the agent to buy shares, execution is **paused** and an approval card appears:

- Shows symbol, quantity, estimated price, and total cost.
- Add an optional supervisor note, then click **Approve & Execute Trade** or **Reject Trade**.
- The agent resumes and reports the outcome.

---

## Environment variable reference

| Variable | Required | Description |
|---|---|---|
| `INCEPTION_API_KEY` | ✅ | Inception Labs API key (mercury-3 model) |
| `TAVILY_API_KEY` | ✅ | Tavily search API key |
| `GOOGLE_API_KEY` | ✅ | Google API key (Gemini embeddings for RAG) |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing |
| `LANGSMITH_TRACING` | Optional | Set `true` to enable tracing |
| `LANGSMITH_ENDPOINT` | Optional | LangSmith endpoint URL |
| `LANGSMITH_PROJECT` | Optional | LangSmith project name |
| `STOCK_API_KEY` | Optional | Alpha Vantage key (demo prices used if absent) |
| `OPEN_WEATHER_API_KEY` | Optional | OpenWeatherMap key |
| `HOW_MANY_SEARCHES` | Optional | Number of searches in deep research (default: `5`) |

---

## Tech stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — stateful agent graph with SQLite checkpointing
- **[Streamlit](https://streamlit.io)** — chat UI, file uploader, human-in-the-loop cards
- **[Inception Labs mercury-3](https://inceptionlabs.ai)** — free LLM (OpenAI-compatible API)
- **[OpenAI Agents SDK](https://github.com/openai/openai-agents-python)** — deep research multi-agent orchestration
- **[Tavily](https://tavily.com)** — web search (free tier)
- **[FAISS](https://github.com/facebookresearch/faiss)** — local vector store for RAG
- **[Google Generative AI](https://ai.google.dev)** — `gemini-embedding-001` for document embeddings

