import os
from dotenv import load_dotenv

load_dotenv()

from agent.workflow import app
from langchain_core.messages import HumanMessage

state = {
    "messages": [HumanMessage(content="Hi, what is the pricing for the basic plan?")],
    "user_details": {"name": None, "email": None, "platform": None},
    "tool_triggered": False
}

try:
    print("Invoking graph...")
    res = app.invoke(state)
    print("\n--- RESULTS ---")
    print("Intent Detected:", res.get("intent"))
    print("Response Content:", res["messages"][-1].content)
except Exception as e:
    print("FAILED:", str(e))
    import traceback
    traceback.print_exc()
