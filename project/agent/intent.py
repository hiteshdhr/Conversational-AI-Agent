from langchain_core.prompts import PromptTemplate
import json

def detect_intent(messages, llm) -> str:
    """
    Detect user intent: 'greeting', 'inquiry', or 'high_intent'.
    Optimized with fast-track heuristics to avoid unnecessary API latency.
    """
    # extract text safely (handles list content from some models)
    raw_content = messages[-1].content
    if isinstance(raw_content, list):
        user_msg = " ".join([block.get("text", "") for block in raw_content if isinstance(block, dict)]).lower().strip()
    else:
        user_msg = str(raw_content).lower().strip()
    
    # OFFLINE HEURISTICS FOR INSTANT RESPONSE (0.0ms delay)
    high_intent_kws = ["buy", "purchase", "sign up", "subscribe", "want to try", "demo", "interested in", "pro plan", "youtube"]
    inquiry_kws = ["price", "cost", "plan", "policy", "refund", "resolution", "4k", "feature", "video", "how", "what", "which", "can i"]
    greeting_kws = ["hi", "hello", "hey", "hii", "good morning", "greetings"]
    
    if any(k in user_msg for k in high_intent_kws): return "high_intent"
    if any(k in user_msg for k in inquiry_kws): return "inquiry"
    if user_msg in greeting_kws: return "greeting"
    
    # fallback strictly if ambiguous
    prompt = PromptTemplate.from_template('''You are an AI analyzing user intent for a video editing SaaS called AutoStream. 
Classify the intent of the latest user message context into exactly ONE of the following categories:
- greeting (casual hello, introducing themselves)
- inquiry (asking about pricing, features, or policies)
- high_intent (ready to buy, wants a specific plan, wanting to try it for their platform)

Conversation History:
{history}

Reply ONLY with valid JSON exactly matching this format: {{"intent": "intent_name"}}''')

    history = ""
    for msg in messages:
        role = "User" if msg.type == "human" else "AI"
        history += f"{role}: {msg.content}\n"
        
    chain = prompt | llm
    res = chain.invoke({"history": history})
    try:
        content = res.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return data.get("intent", "inquiry")
    except Exception as e:
        return "inquiry" # fallback
