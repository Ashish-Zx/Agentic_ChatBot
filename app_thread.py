from agentic_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage
import streamlit as st
import uuid

# generate unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())

# add new thread id to the conversation list
def add_thread(thread_id):
    # prevent same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# load conversation messages from LangGraph checkpointer for a specific thread
def load_thread_messages(thread_id):
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    history = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human" else "assistant"
        history.append({"role": role, "content": msg.content})
    return history

# get a preview title for a thread (uses the first user message)
def get_thread_title(thread_id, max_length=24):
    if "thread_titles" not in st.session_state:
        st.session_state["thread_titles"] = {}

    title = st.session_state["thread_titles"].get(thread_id)
    if title and title != "New Chat":
        return title

    # check if there are messages in LangGraph checkpointer
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    if messages:
        first_prompt = messages[0].content
        clean_text = " ".join(first_prompt.split())
        title = (clean_text[:max_length] + "...") if len(clean_text) > max_length else clean_text
        st.session_state["thread_titles"][thread_id] = title
        return title

    return title or "New Chat"


st.title("Agentic Chatbot With LangGraph")

# create message_history when the app runs for the first time
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# create a list for storing all conversation thread IDs
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

# store titles/previews for threads
if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}

# display sidebar
st.sidebar.title("Chat Threads")

# create a new conversation
def reset_chat():
    new_thread_id = generate_thread_id()
    st.session_state["thread_id"] = new_thread_id
    st.session_state["message_history"] = []
    st.session_state["thread_titles"][new_thread_id] = "New Chat"
    add_thread(new_thread_id)

# initialize default thread on startup
if "thread_id" not in st.session_state:
    reset_chat()

# create a button for new chat
if st.sidebar.button("New Chat", width="stretch"):
    reset_chat()
    st.rerun()

# Sidebar: display conversation threads as clickable buttons
with st.sidebar:
    st.write("### Conversation Threads")
    if st.session_state["chat_threads"]:
        for thread_id in st.session_state["chat_threads"][::-1]:
            is_active = (thread_id == st.session_state["thread_id"])
            title = get_thread_title(thread_id)
            if st.button(
                f":material/chat: {title}",
                key=f"thread_btn_{thread_id}",
                width="stretch",
                type="primary" if is_active else "secondary",
                help=f"Thread ID: {thread_id}",
            ):
                # Switch to selected conversation and load its messages
                st.session_state["thread_id"] = thread_id
                st.session_state["message_history"] = load_thread_messages(thread_id)
                st.rerun()

# render previous messages from chat history
for message in st.session_state["message_history"]:
    if isinstance(message, dict):
        role = message.get("role", "user")
        content = message.get("content", "")
    else:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        content = message.content
    with st.chat_message(role):
        st.write(content)

# create the chat input box
user_input = st.chat_input("Ask a question to the Agent")

# Run this block after the user submits a message
if user_input:
    # Set thread title based on the first user message if it's currently "New Chat"
    current_title = st.session_state["thread_titles"].get(st.session_state["thread_id"], "New Chat")
    if current_title == "New Chat":
        clean_prompt = " ".join(user_input.split())
        title_preview = (clean_prompt[:24] + "...") if len(clean_prompt) > 24 else clean_prompt
        st.session_state["thread_titles"][st.session_state["thread_id"]] = title_preview

    st.session_state["message_history"].append({
        'role': 'user',
        'content': user_input
    })

    # display the user message in the chat window
    with st.chat_message("user"): 
        st.write(user_input)

    # Pass the current thread ID to Langgraph
    # langgraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    # display assistant response with streaming
    with st.chat_message("assistant"):
        def generate_response():
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if message_chunk.content:
                    yield message_chunk.content

        full_response = st.write_stream(generate_response())

    # Store the complete response
    st.session_state["message_history"].append({"role": "assistant", "content": full_response})
