import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent.workflow import app

st.set_page_config(page_title="AutoStream AI Agent", page_icon="🤖", layout="wide")

def extract_text(content):
    """Safely convert any message content to a plain string."""
    try:
        if isinstance(content, list):
            return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        return str(content)
    except:
        return "Something went wrong."


# --- INIT SESSION STATE ---
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_details" not in st.session_state:
        st.session_state.user_details = {"name": None, "email": None, "platform": None}
    if "tool_triggered" not in st.session_state:
        st.session_state.tool_triggered = False
    if "current_intent" not in st.session_state:
        st.session_state.current_intent = "Browsing"
    if "in_funnel" not in st.session_state:
        st.session_state.in_funnel = False

init_state()


# --- PROCESS INPUT FIRST (so sidebar updates on same render) ---
prompt = st.chat_input("How can I help you today?")
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))

    text = prompt.lower().strip()
    if text in ["hi", "hello", "hey", "hii", "hiii"]:
        # fast-track greeting - no API call needed
        reply = "Hey! 👋 How can I help you today?"
        st.session_state.messages.append(AIMessage(content=reply))
    else:
        with st.spinner("Agent is typing..."):
            try:
                res = app.invoke({
                    "messages": st.session_state.messages,
                    "user_details": st.session_state.user_details,
                    "tool_triggered": st.session_state.tool_triggered,
                    "in_funnel": st.session_state.in_funnel,
                })

                # sync all state from graph output
                st.session_state.messages = list(res["messages"])

                if res.get("user_details"):
                    # merge any newly extracted values
                    for k in ["name", "email", "platform"]:
                        val = res["user_details"].get(k)
                        if val:
                            st.session_state.user_details[k] = val

                if "tool_triggered" in res:
                    st.session_state.tool_triggered = res["tool_triggered"]
                if "intent" in res:
                    st.session_state.current_intent = res["intent"]
                if "in_funnel" in res:
                    st.session_state.in_funnel = res["in_funnel"]

            except Exception as e:
                st.error(f"Error: {e}")
                print(f"Graph error: {e}")


# --- SIDEBAR (drawn after state is updated) ---
with st.sidebar:
    st.markdown("## 🧠 Agent State Memory")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.user_details = {"name": None, "email": None, "platform": None}
        st.session_state.tool_triggered = False
        st.session_state.current_intent = "Browsing"
        st.session_state.in_funnel = False
        st.rerun()

    st.divider()

    # Status indicator
    if st.session_state.tool_triggered:
        st.success("🔵 Status: Lead Captured!")
    elif st.session_state.in_funnel:
        st.warning("🟡 Status: Qualifying Lead")
    else:
        st.info("🟢 Status: Browsing")

    st.markdown(f"**Detected Intent:** `{st.session_state.current_intent}`")

    st.divider()
    st.markdown("### 📋 Collected Info")

    d = st.session_state.user_details

    # show name - highlight in green if filled
    name_val = d.get("name") or "Not provided"
    if d.get("name"):
        st.success(f"👤 Name: **{name_val}**")
    else:
        st.markdown(f"👤 Name: *{name_val}*")

    email_val = d.get("email") or "Not provided"
    if d.get("email"):
        st.success(f"📧 Email: **{email_val}**")
    else:
        st.markdown(f"📧 Email: *{email_val}*")

    platform_val = d.get("platform") or "Not provided"
    if d.get("platform"):
        st.success(f"📱 Platform: **{platform_val}**")
    else:
        st.markdown(f"📱 Platform: *{platform_val}*")


# --- MAIN CHAT AREA ---
st.title("🤖 AutoStream AI Agent")

if len(st.session_state.messages) == 0:
    st.info("""💡 **Try asking:**
• What are your pricing plans?
• Do you support 4K videos?
• I want to try the Pro plan for my YouTube channel""")

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(extract_text(msg.content))
