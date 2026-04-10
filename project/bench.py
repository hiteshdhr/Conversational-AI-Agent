import time
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
from agent.workflow import llm, rag_sys
from agent.intent import detect_intent

messages = [HumanMessage(content="What is the pricing?")]

start = time.time()
detect_intent(messages, llm)
t_intent = time.time() - start
print("Intent time:", t_intent)

start = time.time()
rag_sys.retrieve("What is the pricing?")
t_reg = time.time() - start
print("Retrieve time:", t_reg)

start = time.time()
llm.invoke("What is the pricing?")
t_gen = time.time() - start
print("Generation time:", t_gen)
