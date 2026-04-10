from langgraph.graph import StateGraph, START, END
from agent.memory import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.intent import detect_intent
from agent.rag import RAGPipeline
from agent.tools import mock_lead_capture
from langchain_core.messages import HumanMessage, AIMessage
import json
import re
import os

try:
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0, max_output_tokens=300)
    rag_sys = RAGPipeline()
except Exception as e:
    print("Warning: GEMINI_API_KEY might be missing:", e)
    llm = None
    rag_sys = None


def get_safe_text(content):
    """Handles both string and list-format message content."""
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def get_intent_step(state: AgentState):
    print("=== intent check ===")

    # if already inside funnel, stay there
    if state.get("in_funnel", False):
        print("sticky funnel: locked in lead capture")
        return {"intent": "high_intent"}

    msgs = state.get("messages", [])
    intent_val = detect_intent(msgs, llm)
    print(f"detected intent: {intent_val}")
    return {"intent": intent_val}


def handle_greeting(state: AgentState):
    user_msg = get_safe_text(state["messages"][-1].content)
    # keep it very short - no extra context needed
    prompt_str = f"Greet the user back in max 8 words for AutoStream SaaS. User said: {user_msg}"
    res = llm.invoke(prompt_str)
    return {"messages": [res]}


def handle_rag(state: AgentState):
    user_query = get_safe_text(state["messages"][-1].content)
    context_data = rag_sys.retrieve(user_query, k=2)

    # short prompt = faster response
    prompt = f"""AutoStream SaaS assistant. Answer briefly using only the KB below.
For plans use: 🔥 Plan - $price/month ✔ feature1 ✔ feature2
KB: {context_data[:800]}
Q: {user_query}
Answer (max 80 words):"""

    result = llm.invoke(prompt)
    return {"messages": [result]}


def handle_lead_capture(state: AgentState):
    # already done - just confirm
    if state.get("tool_triggered", False):
        return {"messages": [AIMessage(content="✅ We already have your details! Our team will reach out soon.")]}

    # get current saved details
    usr_data = dict(state.get("user_details") or {})
    if not usr_data:
        usr_data = {"name": None, "email": None, "platform": None}

    # build a clean readable history for the LLM to extract from
    # only use last 6 messages to keep prompt short
    history_lines = []
    for m in list(state.get("messages", []))[-6:]:
        role = "User" if m.type == "human" else "Agent"
        history_lines.append(f"{role}: {get_safe_text(m.content)[:120]}")
    history_str = "\n".join(history_lines)

    ext_prompt = f"""Extract from chat. Current: {json.dumps(usr_data)}
Chat:
{history_str}
Return JSON only: {{"name":"...","email":"...","platform":"..."}} use null if missing."""

    res = llm.invoke(ext_prompt)
    try:
        raw = get_safe_text(res.content)  # handle list-format content
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if match:
            extracted = json.loads(match.group())
            for k in ["name", "email", "platform"]:
                val = extracted.get(k)
                if val and str(val).strip().lower() not in ("null", "none", ""):
                    usr_data[k] = str(val).strip()
                    print(f"extracted {k}: {usr_data[k]}")
    except Exception as e:
        print(f"extraction error: {e}")

    # check completion - trigger tool if all collected
    if usr_data.get("name") and usr_data.get("email") and usr_data.get("platform"):
        print(">>> calling mock_lead_capture tool!")
        mock_lead_capture(usr_data["name"], usr_data["email"], usr_data["platform"])
        return {
            "user_details": usr_data,
            "tool_triggered": True,
            "in_funnel": False,
            "messages": [AIMessage(content=f"🎉 Perfect! I've captured your details:\n\n"
                                           f"👤 **{usr_data['name']}**\n"
                                           f"📧 **{usr_data['email']}**\n"
                                           f"📱 **{usr_data['platform']}**\n\n"
                                           f"Our team will reach out very soon. Welcome to AutoStream! 🚀")]
        }

    # figure out what to ask next
    if not usr_data.get("name"):
        msg = "Great choice! 🚀 Let's get you started.\n\nWhat's your **name**?"
    elif not usr_data.get("email"):
        msg = f"Nice to meet you, **{usr_data['name']}**! 📧\n\nWhat's the best **email address** to reach you?"
    else:
        msg = f"Almost there! 📱\n\nWhich **platform** are you creating content for? (e.g. YouTube, Instagram, TikTok)"

    return {
        "user_details": usr_data,   # <- always return updated details
        "in_funnel": True,           # <- keep funnel locked
        "messages": [AIMessage(content=msg)]
    }


def route_next_step(state: AgentState):
    val = state.get("intent", "inquiry")
    if val == "greeting":
        return "do_greeting"
    elif val == "high_intent":
        return "do_capture"
    return "do_rag"


# build the graph
build_graph = StateGraph(AgentState)
build_graph.add_node("intent", get_intent_step)
build_graph.add_node("do_greeting", handle_greeting)
build_graph.add_node("do_rag", handle_rag)
build_graph.add_node("do_capture", handle_lead_capture)

build_graph.add_edge(START, "intent")
build_graph.add_conditional_edges("intent", route_next_step, {
    "do_greeting": "do_greeting",
    "do_rag": "do_rag",
    "do_capture": "do_capture"
})
build_graph.add_edge("do_greeting", END)
build_graph.add_edge("do_rag", END)
build_graph.add_edge("do_capture", END)

app = build_graph.compile()
