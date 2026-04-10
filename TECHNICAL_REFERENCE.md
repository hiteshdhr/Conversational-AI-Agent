# AutoStream AI Agent — Technical Reference

## 6.1 Core Code

---

### Agent Logic — `agent/workflow.py`

The heart of the system. Built using **LangGraph** as a state machine with 4 nodes:

```
User Input
    ↓
[intent node]      → classifies the message
    ↓
Route decision:
  greeting    → [do_greeting]  → short friendly reply
  inquiry     → [do_rag]       → retrieve from knowledge base
  high_intent → [do_capture]   → collect name, email, platform
    ↓
[END]
```

**Key functions:**

| Function | What it does |
|---|---|
| `get_intent_step()` | Classifies intent. If `in_funnel=True`, stays in lead capture (sticky funnel). |
| `handle_greeting()` | Sends a short greeting reply using the LLM. |
| `handle_rag()` | Retrieves relevant KB context using FAISS, generates formatted answer. |
| `handle_lead_capture()` | Extracts name/email/platform from chat, asks for missing info step-by-step, triggers tool when all 3 collected. |
| `route_next_step()` | Conditional router — maps intent to the correct node. |

**State carried across turns (`AgentState`):**
```python
messages        # full conversation history
intent          # current detected intent
user_details    # {"name": ..., "email": ..., "platform": ...}
tool_triggered  # True once mock_lead_capture() is called
in_funnel       # True while collecting lead details (sticky lock)
```

---

### RAG Pipeline — `agent/rag.py`

Retrieves pricing, features, and policy info from the local knowledge base.

**How it works:**
1. On first run: loads `data/knowledge_base.json` → splits into chunks → generates **Gemini embeddings** → saves FAISS index to disk
2. On subsequent runs: **loads cached FAISS index instantly** (no API call needed)
3. At query time: converts user question to an embedding → finds top-k similar chunks → returns them as context to the LLM

```python
class RAGPipeline:
    embeddings   = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS  # stored locally in data/faiss_index/

    def retrieve(query, k=2):
        # returns top-2 matching KB chunks as a string
```

**Why FAISS?** Fast, local, no network call for retrieval — adds only ~0.5s latency.

---

### Intent Detection — `agent/intent.py`

Classifies every user message into one of 3 categories:

| Intent | Trigger Examples |
|---|---|
| `greeting` | "hi", "hello", "hey", "hii", "good morning" |
| `inquiry` | "price", "plan", "4k", "refund", "how", "what", etc. |
| `high_intent` | "subscribe", "buy", "want to try", "pro plan", "youtube", etc. |

**Two-tier approach for speed:**

```python
# TIER 1: Instant offline keyword matching (0ms, no API cost)
if any(k in user_msg for k in high_intent_kws): return "high_intent"
if any(k in user_msg for k in inquiry_kws):     return "inquiry"
if user_msg in greeting_kws:                    return "greeting"

# TIER 2: LLM fallback only if message is ambiguous (~2s)
chain = prompt | llm
return data.get("intent", "inquiry")
```

Tier 1 handles ~95% of real messages with zero API cost.

---

### Tool Execution — `agent/tools.py`

A mock function simulating a real CRM/lead capture API call.

```python
def mock_lead_capture(name: str, email: str, platform: str) -> None:
    """Mock API function to capture a lead."""
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    return True
```

**When it fires:** Only after the agent has confirmed all 3 fields — name, email, and platform — are extracted from the conversation. Once triggered:
- `tool_triggered = True` is saved to state and shown in the sidebar
- `in_funnel = False` releases the funnel lock
- A confirmation message with all details is sent to the user

**In production**, replace this function with a real API call to HubSpot, Salesforce, or a webhook endpoint.

---

## 6.2 Requirements

All dependencies required to run the project (`requirements.txt`):

```
langchain>=0.2.14
langchain-community>=0.2.12
langgraph>=0.2.3
langchain-google-genai>=1.0.8
faiss-cpu>=1.8.0.post1
python-dotenv>=1.0.1
streamlit>=1.37.0
```

### Dependency Breakdown

| Package | Purpose |
|---|---|
| `langchain` | Core LLM abstraction layer — prompts, chains, message types |
| `langchain-community` | Provides FAISS vectorstore integration |
| `langgraph` | State machine graph framework for the agentic workflow |
| `langchain-google-genai` | Google Gemini LLM and embedding model wrappers |
| `faiss-cpu` | Local vector similarity search for RAG retrieval |
| `python-dotenv` | Loads `GEMINI_API_KEY` from the `.env` file |
| `streamlit` | Web UI framework for the chat interface |

### Install Command

```bash
pip install -r requirements.txt
```

### Environment Variable Required

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

Get a free key at: https://aistudio.google.com/app/apikey
