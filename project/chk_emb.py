import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    print("Available embedding models:")
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print("FOUND:", m.name)
except Exception as e:
    print(e)
