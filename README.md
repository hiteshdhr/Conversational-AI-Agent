# AutoStream Social-to-Lead Agentic Workflow

This project implements an AI agent that detects user intent, answers questions from a knowledge base using RAG, and collects information to execute a mock lead capture function.

## Architecture & Why LangGraph

**Architecture:**
- Intent detection using LLM mapping to 3 distinct behaviors: greetings, inquiries, high-intent captures.
- FAISS as an in-memory vectorstore for retrieving pricing/policy documents (RAG).
- Dynamic State passing to accumulate conversational context and lead details (`name`, `email`, `platform`).

**Why LangGraph?**
LangGraph provides a state machine model natively suited to loop logic, branching (conditional edges based on intent), and persistent memory schemas over an extended conversational timeline. It's much simpler to manage "Ask for info, pause, loop until info gathered" utilizing Node/Edge definitions compared to unmanaged loops.

## How to Run Locally

1. Setup environment variables by copying `.env.template` to `.env` and adding your GEMINI_API_KEY.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run main.py
   ```

## WhatsApp Deployment Concept (using Webhooks)

To deploy this workflow over WhatsApp Business:
1. Register for WhatsApp Business API and set up an inbound webhook URL (e.g. using FastAPI or Flask).
2. When a user messages, WhatsApp POSTs to your webhook.
3. Parse the message `Body` and sender's phone number as session ID.
4. Manage session memory across turns locally using Redis or storing the LangGraph State linked to the phone number.
5. Invoke the LangGraph workflow (`app.invoke()`).
6. Retrieve the LangGraph result state `messages[-1]`, and POST that content back to the WhatsApp Business API outgoing message endpoint.
7. Any "Tool captures" can be fired into a 
