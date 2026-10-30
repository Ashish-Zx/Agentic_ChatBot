from agentic_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

user_message= "What is object oriented programming in one line?"
thread_id = 1

config = {'configurable': {'thread_id': thread_id}}

response = chatbot.invoke({'messages':[HumanMessage(content=user_message)]},config=config)

print(response['messages'][-1].content)