from agentic_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage
import streamlit as st

st.title("Agentic Chatbot With LangGraph")

CONFIG = {'configurable': {'thread_id': 2}}

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

    # display assistant response
    with st.chat_message("assistant"):
        response = chatbot.invoke({'messages':[HumanMessage(content=user_input)]},config=CONFIG)
        st.write(response['messages'][-1].content)
     
    # store assistant response in history
    st.session_state.message_history.append(response['messages'][-1])



