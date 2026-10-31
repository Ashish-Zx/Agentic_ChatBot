from agentic_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage
import streamlit as st

st.title("Agentic Chatbot With LangGraph")

CONFIG = {'configurable': {'thread_id': 1}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# loading the conversation history if it exists
for message in st.session_state['message_history']:
    if isinstance(message, dict):
        role = message.get("role", "user")
        content = message.get("content", "")
    else:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        content = message.content
    with st.chat_message(role):
        st.write(content)

user_input = st.chat_input("Ask a question to the Agent")

if user_input:
    with st.chat_message("user"):   
        st.write(user_input)
    # Add user message to history
    st.session_state.message_history.append(HumanMessage(content=user_input))

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
    st.session_state.message_history.append({"role": "assistant", "content": full_response})



