import json
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAGPipeline:
    def __init__(self):
        # Using Gemini embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vector_store = None
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kb_path = os.path.join(base_dir, "data", "knowledge_base.json")
        faiss_path = os.path.join(base_dir, "data", "faiss_index")
        
        # Load from disk if it already exists (instant startup!)
        if os.path.exists(faiss_path):
            print("Loading cached FAISS index...")
            self.vector_store = FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)
            return

        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        texts = []
        for category, items in data.items():
            for key, value in items.items():
                texts.append(f"Category: {category}, Feature/Policy: {key}, Details: {value}")
                
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.create_documents(texts)
        
        if chunks:
            print("Computing embeddings and saving FAISS to disk...")
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.save_local(faiss_path)

    def retrieve(self, query: str, k: int = 2):
        if not self.vector_store:
            return ""
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
