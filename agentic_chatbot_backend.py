import os
import sqlite3
import requests
import math
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Load environment variables
load_dotenv()

# 2. Initialize LLM
llm = ChatOpenAI(
    model="mercury-3",
    temperature=0.75,
    api_key=os.getenv("INCEPTION_API_KEY"),
    base_url="https://api.inceptionlabs.ai/v1"
)

# 2.1 Initialize RAG Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

import json
import shutil
from typing import TypedDict, Annotated, List, Union, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Paths and directories for dynamic RAG
UPLOAD_DIR = "uploaded_documents"
DB_PATH = "faiss_rag_db"
METADATA_FILE = os.path.join(DB_PATH, "indexed_files.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory retriever and vector store cache
_vector_store = None
_retriever = None

def _load_metadata() -> dict:
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"files": [], "total_chunks": 0}
    return {"files": [], "total_chunks": 0}

def _save_metadata(metadata: dict) -> None:
    try:
        os.makedirs(DB_PATH, exist_ok=True)
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata: {e}")

def get_rag_status() -> dict:
    """Retrieve the current status of the knowledge base."""
    index_file = os.path.join(DB_PATH, "index.faiss")
    if not os.path.exists(index_file):
        return {"has_index": False, "files": [], "total_chunks": 0}

    metadata = _load_metadata()
    files = metadata.get("files", [])
    total_chunks = metadata.get("total_chunks", 0)

    # Fallback if metadata file does not exist yet but FAISS index is on disk
    if total_chunks == 0:
        retriever = get_retriever()
        if _vector_store and hasattr(_vector_store, "index"):
            total_chunks = _vector_store.index.ntotal
        if not files:
            if os.path.exists(UPLOAD_DIR):
                uploaded = [f for f in os.listdir(UPLOAD_DIR) if not f.startswith(".")]
                if uploaded:
                    files = uploaded
            if not files and os.path.exists("1.pdf"):
                files = ["1.pdf"]

    return {
        "has_index": True,
        "files": files,
        "total_chunks": total_chunks
    }

def clear_rag_documents() -> bool:
    """Clear the dynamic FAISS vector database and in-memory cache."""
    global _retriever, _vector_store
    _retriever = None
    _vector_store = None
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            return True
        except Exception as e:
            print(f"Error removing {DB_PATH}: {e}")
            return False
    return True

def load_single_document(file_path: str) -> list[Document]:
    """Load a single document (PDF, TXT, MD, etc.) and stamp filename metadata."""
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        return docs
    elif ext in [".txt", ".md", ".csv", ".json", ".py"]:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        except UnicodeDecodeError:
            loader = TextLoader(file_path, encoding="latin-1")
            docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        return docs
    else:
        # Fallback raw text loader
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": filename})]
        except Exception as e:
            print(f"Failed to load file {file_path}: {e}")
            return []

def ingest_rag_documents(file_paths: Union[List[str], str], replace: bool = False) -> dict:
    """Dynamically ingest one or multiple documents into the FAISS vector database.
    
    Args:
        file_paths: Path or list of paths of documents to ingest.
        replace: If True, replace the existing index; otherwise append.
    """
    global _retriever, _vector_store

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    valid_paths = [p for p in file_paths if os.path.exists(p)]
    if not valid_paths:
        return {"status": "error", "message": "No valid file paths found to ingest."}

    all_docs = []
    processed_files = []
    for path in valid_paths:
        try:
            docs = load_single_document(path)
            if docs:
                all_docs.extend(docs)
                processed_files.append(os.path.basename(path))
        except Exception as e:
            print(f"Error loading {path}: {e}")

    if not all_docs:
        return {"status": "error", "message": "Could not extract text from the provided documents."}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    if not chunks:
        return {"status": "error", "message": "No chunks generated from the provided documents."}

    existing_index_exists = os.path.exists(os.path.join(DB_PATH, "index.faiss"))
    metadata = _load_metadata() if (existing_index_exists and not replace) else {"files": [], "total_chunks": 0}

    try:
        if existing_index_exists and not replace:
            vector_store = FAISS.load_local(
                folder_path=DB_PATH,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            vector_store.add_documents(chunks)
        else:
            vector_store = FAISS.from_documents(chunks, embeddings)

        os.makedirs(DB_PATH, exist_ok=True)
        vector_store.save_local(DB_PATH)

        current_files = set(metadata.get("files", []))
        current_files.update(processed_files)
        total_chunks = vector_store.index.ntotal

        _save_metadata({
            "files": sorted(list(current_files)),
            "total_chunks": total_chunks
        })

        _vector_store = vector_store
        _retriever = vector_store.as_retriever(search_kwargs={"k": 4}, search_type="similarity")

        return {
            "status": "success",
            "files_processed": processed_files,
            "chunks_added": len(chunks),
            "total_chunks": total_chunks
        }
    except Exception as e:
        print(f"Error ingesting documents: {e}")
        return {"status": "error", "message": str(e)}

def ingest_rag_document(file_path: str):
    """Backward compatibility wrapper for ingesting a single document."""
    return ingest_rag_documents([file_path], replace=False)

def get_retriever(force_reload: bool = False):
    """Dynamically get the FAISS retriever from disk or in-memory cache."""
    global _retriever, _vector_store
    if _retriever is not None and not force_reload:
        return _retriever

    index_file = os.path.join(DB_PATH, "index.faiss")
    if not os.path.exists(index_file):
        return None

    try:
        _vector_store = FAISS.load_local(
            folder_path=DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        _retriever = _vector_store.as_retriever(search_kwargs={"k": 4}, search_type="similarity")
        return _retriever
    except Exception as e:
        print(f"Error loading retriever: {e}")
        return None
    
# 4. Define Tools
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
)

@tool
def calculator(expression: str) -> str:
    """Useful for simple math calculations.
    Input should be a valid math expression, e.g. 2 + 2 * 4.
    """
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def get_stock_price(symbol: str) -> str:
    """Useful to get the latest stock price for a given stock ticker symbol (e.g. IBM, AAPL)."""
    try:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_API_KEY") or "demo"
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}"
        response = requests.get(url)
        data = response.json()

        # Extract the latest close price from the 5-minute time series
        if "Time Series (5min)" in data:
            latest_timestamp = next(iter(data["Time Series (5min)"]))
            latest_price = data["Time Series (5min)"][latest_timestamp]["4. close"]
            return f"The latest stock price of {symbol.upper()} is ${latest_price} (as of {latest_timestamp})."

        # Handle API rate limits or invalid symbol notices
        if "Information" in data:
            return f"Alpha Vantage Notice: {data['Information']}"
        if "Error Message" in data:
            return f"Error: Could not find stock symbol '{symbol}'."

        return f"Unable to fetch price for '{symbol}': {data}"
    except Exception as e:
        return f"Error fetching stock price: {e}"

@tool
def get_weather(city: str) -> str:
    """Useful to get the current weather for a given city."""
    try:
        api_key = os.getenv("OPEN_WEATHER_API_KEY") or os.getenv("OPENWEATHERMAP_API_KEY")
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        if "main" in data and "weather" in data:
            return f"The weather in {city} is {data['weather'][0]['description']} with a temperature of {data['main']['temp']}°C."
        if "message" in data:
            return f"Error: {data['message']}"
        return f"Unable to fetch weather for {city}"
    except Exception as e:
        return f"Error fetching weather: {e}"

@tool
def rag_tool(query: str) -> str:
    """Useful to search and retrieve relevant information from uploaded documents in the knowledge base (e.g. PDFs, research papers, reports, notes)."""
    retriever = get_retriever()
    if retriever is None:
        return "No document knowledge base is available. Please upload documents in the sidebar first."
    try:
        documents = retriever.invoke(query)
        if not documents:
            return "No relevant information found in the knowledge base for this query."
        results = []
        for i, doc in enumerate(documents, 1):
            source = os.path.basename(doc.metadata.get("source", "Document"))
            page = doc.metadata.get("page")
            page_info = f", Page {page + 1}" if page is not None else ""
            results.append(f"[Excerpt {i} from {source}{page_info}]:\n{doc.page_content}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Error retrieving document info: {e}"

# Tools list
tools = [calculator, search_tool, get_weather, get_stock_price, rag_tool]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# 5. State Definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 6. Nodes
def chat_node(state: ChatState):
    """Invoke the tool-aware LLM with conversation history."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# 7. Graph Construction
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# 8. Checkpointer & Compilation
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn=conn)
chatbot = graph.compile(checkpointer=checkpoint)

# 9. Database Helper Functions
def get_all_threads():
    """Retrieve all distinct thread IDs stored in the SQLite checkpointer database."""
    try:
        cur = conn.cursor()
        cur.execute("SELECT thread_id FROM checkpoints GROUP BY thread_id ORDER BY MIN(rowid)")
        return [row[0] for row in cur.fetchall()]
    except Exception:
        return []