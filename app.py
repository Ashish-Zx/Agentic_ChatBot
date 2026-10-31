import os
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from agentic_chatbot_backend import (
    chatbot,
    get_all_threads,
    ingest_rag_documents,
    clear_rag_documents,
    get_rag_status,
    get_pending_interrupt,
    UPLOAD_DIR,
)

# generate unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())

# add new thread id to the conversation list
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# load conversation messages from LangGraph checkpointer for a specific thread
def load_thread_messages(thread_id):
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            history.append({"role": "user", "content": msg.content})
        elif getattr(msg, "type", "") == "tool":
            # Tool internal execution messages are not directly displayed as chat bubbles
            continue
        else:
            # Assistant messages: display only if there is text content
            if getattr(msg, "content", ""):
                history.append({"role": "assistant", "content": msg.content})
    return history

# get a preview title for a thread (uses the first user message)
def get_thread_title(thread_id, max_length=24):
    if "thread_titles" not in st.session_state:
        st.session_state["thread_titles"] = {}

    title = st.session_state["thread_titles"].get(thread_id)
    if title and title != "New Chat":
        return title

    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    if messages:
        first_prompt = messages[0].content
        clean_text = " ".join(first_prompt.split())
        title = (clean_text[:max_length] + "...") if len(clean_text) > max_length else clean_text
        st.session_state["thread_titles"][thread_id] = title
        return title

    return title or "New Chat"

# create a new conversation
def reset_chat():
    new_thread_id = generate_thread_id()
    st.session_state["thread_id"] = new_thread_id
    st.session_state["message_history"] = []
    st.session_state["thread_titles"][new_thread_id] = "New Chat"
    add_thread(new_thread_id)

# Session state initialization
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

if "thread_id" not in st.session_state:
    if st.session_state["chat_threads"]:
        latest_thread = st.session_state["chat_threads"][-1]
        st.session_state["thread_id"] = latest_thread
        st.session_state["message_history"] = load_thread_messages(latest_thread)
    else:
        reset_chat()

# Sidebar: Document Knowledge Base (Multiple Uploads & Dynamic RAG)
with st.sidebar:
    st.subheader("Knowledge base", divider="gray")
    
    rag_status = get_rag_status()
    if rag_status["has_index"]:
        file_count = len(rag_status["files"])
        st.caption(f":material/check_circle: **Active:** {file_count} file(s) ({rag_status['total_chunks']} chunks)")
        with st.expander("Indexed files", expanded=False):
            for fname in rag_status["files"]:
                st.markdown(f"- :material/description: `{fname}`")
    else:
        st.caption(":material/info: No documents indexed yet.")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF, TXT, or MD documents into the knowledge base."
    )

    replace_mode = st.toggle(
        "Replace existing index",
        value=False,
        help="Enable to replace the existing knowledge base instead of appending."
    )

    if st.button(
        "Index documents",
        icon=":material/upload_file:",
        width="stretch",
        type="primary",
        disabled=not uploaded_files
    ):
        with st.spinner("Processing & indexing documents..."):
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            saved_paths = []
            for file in uploaded_files:
                save_path = os.path.join(UPLOAD_DIR, file.name)
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
                saved_paths.append(save_path)

            result = ingest_rag_documents(saved_paths, replace=replace_mode)
            if result.get("status") == "success":
                st.toast(
                    f"Indexed {len(result['files_processed'])} file(s) ({result['chunks_added']} chunks)!",
                    icon="✅"
                )
                st.rerun()
            else:
                st.error(result.get("message", "Error indexing documents."), icon=":material/error:")

    if rag_status["has_index"]:
        if st.button("Clear knowledge base", icon=":material/delete:", width="stretch"):
            clear_rag_documents()
            st.toast("Knowledge base cleared!", icon="🗑️")
            st.rerun()

    st.divider()

    # Sidebar: Conversation threads
    st.subheader("Conversations", divider="gray")
    if st.button("New chat", icon=":material/add:", width="stretch"):
        reset_chat()
        st.rerun()

    if st.session_state["chat_threads"]:
        for thread_id in st.session_state["chat_threads"][::-1]:
            is_active = (thread_id == st.session_state["thread_id"])
            title = get_thread_title(thread_id)
            has_pending = bool(get_pending_interrupt(thread_id))
            icon = ":material/notification_important:" if has_pending else ":material/chat:"
            label = f"{icon} {title}" + (" (Action Needed)" if has_pending else "")
            if st.button(
                label,
                key=f"thread_btn_{thread_id}",
                width="stretch",
                type="primary" if is_active else "secondary",
                help=f"Thread ID: {thread_id}" + (" - Pending Approval" if has_pending else ""),
            ):
                st.session_state["thread_id"] = thread_id
                st.session_state["message_history"] = load_thread_messages(thread_id)
                st.rerun()

# Main Chat View
st.title("Agentic Chatbot With LangGraph")

# Dynamic status badge
current_rag = get_rag_status()
if current_rag["has_index"]:
    file_list = ", ".join(current_rag["files"][:3])
    if len(current_rag["files"]) > 3:
        file_list += f" +{len(current_rag['files']) - 3} more"
    st.caption(f":material/menu_book: **Knowledge base active:** {file_list} ({current_rag['total_chunks']} chunks)")
else:
    st.caption(":material/info: No documents indexed. Upload documents in the sidebar to ask questions about your files.")

# Render previous messages from chat history
for message in st.session_state["message_history"]:
    if isinstance(message, dict):
        role = message.get("role", "user")
        content = message.get("content", "")
    else:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        content = message.content
    with st.chat_message(role):
        st.write(content)

CONFIG = {
    "configurable": {
        "thread_id": st.session_state["thread_id"]
    },
    "metadata": {
        "title": st.session_state["thread_id"],
    },
    "run_name": "chat_trace"
}

# Check if there is a pending Human-in-the-Loop Interrupt
pending_action = get_pending_interrupt(st.session_state["thread_id"])

if pending_action and pending_action.get("action") == "buy_stocks":
    with st.container(border=True):
        st.markdown("#### :material/security: Human-in-the-Loop Authorization Required")
        st.caption("The agent has paused execution. A financial trade requires explicit supervisor approval before execution.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stock Symbol", pending_action.get("stock", "N/A"))
        with col2:
            st.metric("Quantity", f"{pending_action.get('quantity', 0)} shares")
        with col3:
            st.metric("Est. Unit Price", f"${pending_action.get('price_per_share', 0):.2f}")
        with col4:
            st.metric("Total Order Cost", f"${pending_action.get('total_cost', 0):,.2f}")

        st.info(pending_action.get("message", "Confirmation needed to execute stock purchase."))

        supervisor_note = st.text_input(
            "Supervisor note / reason (optional):",
            key=f"hitl_note_{st.session_state['thread_id']}",
            placeholder="e.g., Authorized by risk committee, or Declined: excessive exposure"
        )

        col_approve, col_reject = st.columns(2)
        with col_approve:
            if st.button("Approve & Execute Trade", type="primary", icon=":material/check_circle:", width="stretch", key="approve_trade_btn"):
                with st.chat_message("assistant"):
                    def resume_stream():
                        for message_chunk, metadata in chatbot.stream(
                            Command(resume={"approved": True, "reason": supervisor_note or "Approved by human supervisor"}),
                            config=CONFIG,
                            stream_mode="messages"
                        ):
                            if metadata.get("langgraph_node") == "chat_node" and message_chunk.content:
                                yield message_chunk.content

                    full_resumed = st.write_stream(resume_stream())

                if full_resumed:
                    st.session_state["message_history"].append({
                        "role": "assistant",
                        "content": full_resumed
                    })
                st.rerun()

        with col_reject:
            if st.button("Reject Trade", type="secondary", icon=":material/cancel:", width="stretch", key="reject_trade_btn"):
                with st.chat_message("assistant"):
                    def resume_stream():
                        for message_chunk, metadata in chatbot.stream(
                            Command(resume={"approved": False, "reason": supervisor_note or "Declined by human supervisor"}),
                            config=CONFIG,
                            stream_mode="messages"
                        ):
                            if metadata.get("langgraph_node") == "chat_node" and message_chunk.content:
                                yield message_chunk.content

                    full_resumed = st.write_stream(resume_stream())

                if full_resumed:
                    st.session_state["message_history"].append({
                        "role": "assistant",
                        "content": full_resumed
                    })
                st.rerun()

# Chat input box
user_input = st.chat_input(
    placeholder="Pending authorization required above. Please approve or reject before continuing..." if pending_action else "Ask a question to the Agent (e.g., 'Buy 5 shares of Apple')",
    disabled=bool(pending_action)
)

# Run when user submits a message
if user_input:
    # Set thread title based on the first user message if it's currently "New Chat"
    current_title = st.session_state["thread_titles"].get(st.session_state["thread_id"], "New Chat")
    if current_title == "New Chat":
        clean_prompt = " ".join(user_input.split())
        title_preview = (clean_prompt[:24] + "...") if len(clean_prompt) > 24 else clean_prompt
        st.session_state["thread_titles"][st.session_state["thread_id"]] = title_preview

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        def generate_response():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if metadata.get("langgraph_node") == "chat_node" and message_chunk.content:
                    yield message_chunk.content

        full_response = st.write_stream(generate_response())

    if full_response:
        st.session_state["message_history"].append({
            "role": "assistant",
            "content": full_response
        })

    # If this message triggered a Human-in-the-Loop interrupt, trigger rerun to show approval card immediately
    if get_pending_interrupt(st.session_state["thread_id"]):
        st.rerun()
