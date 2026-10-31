import os
from langgraph.graph import StateGraph, START,END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import  ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

# llm= ChatGroq(model="openai/gpt-oss-120b")
llm = ChatOpenAI(
    model="mercury-3",
    temperature=0.75,
    api_key=os.getenv("INCEPTION_API_KEY"),
    base_url="https://api.inceptionlabs.ai/v1"
)


from langgraph.graph.message import add_messages
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


def chat_node(state: ChatState):
 # take query from state
  messages = state['messages']
 #send to llm 
  response = llm.invoke(messages)
 # response store state
  return {'messages': [response]}  

#Connect the database for memory storage
conn=sqlite3.connect(database="chatbot.db",check_same_thread=False)

#Create the sqlite saver 
checkpoint = SqliteSaver(conn=conn)

graph= StateGraph(ChatState)
# Lets add nodes
graph.add_node('chat_node', chat_node)

# add edges

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node' ,END)


chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads():
    """Retrieve all distinct thread IDs stored in the SQLite checkpointer database."""
    try:
        cur = conn.cursor()
        cur.execute("SELECT thread_id FROM checkpoints GROUP BY thread_id ORDER BY MIN(rowid)")
        return [row[0] for row in cur.fetchall()]
    except Exception:
        return []
