from agentic_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage
import streamlit as st

CONFIG = {'configurable': {'thread_id': 1}}

st.title("Agentic Chatbot With LangGraph")

user_input = st.chat_input("Ask a question to the Agent")

if user_input:
    with st.chat_message("user"):   
        st.write(user_input)
    with st.chat_message("assistant"):
        response = chatbot.invoke({'messages':[HumanMessage(content=user_input)]},config=CONFIG)
        st.write(response['messages'][-1].content)




