# backend/rag/retriever.py
from .vectorstore import load_vectorstore

def get_retriever(search_k: int = 5):
    vs = load_vectorstore()
    return vs.as_retriever(search_kwargs={"k": search_k})
