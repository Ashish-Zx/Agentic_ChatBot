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

# 3. Initialize RAG Vector Store from PDF
def ingest_rag_document(file_path):
    if not os.path.exists(file_path):
        return None
    DB_PATH = "faiss_rag_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)
    return vector_store

def get_retriever():
    DB_PATH = "faiss_rag_db"
    if not os.path.exists(DB_PATH):
        if os.path.exists("1.pdf"):
            ingest_rag_document("1.pdf")
        else:
            return None
    try:
        vector_store = FAISS.load_local(
            folder_path=DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store.as_retriever(search_kwargs={"k": 3}, search_type="similarity")
    except Exception as e:
        print(f"Error loading retriever: {e}")
        return None

# Auto-ingest default PDF if index does not exist yet
if not os.path.exists("faiss_rag_db") and os.path.exists("1.pdf"):
    ingest_rag_document("1.pdf")
    
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
    """Useful to search and retrieve relevant information from the uploaded PDF document (e.g., research papers, findings, summaries)."""
    retriever = get_retriever()
    if retriever is None:
        return "PDF document or vector database not found."
    try:
        documents = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in documents])
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