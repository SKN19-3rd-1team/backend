# backend/rag/embeddings.py
from langchain_openai import OpenAIEmbeddings  # 또는 HuggingFaceEmbeddings 등
from backend.config import get_settings

def get_embeddings():
    settings = get_settings()
    # 모델 이름은 .env / config.py 에서 가져오도록
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key
    )
