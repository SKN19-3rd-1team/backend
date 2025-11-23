# backend/rag/embeddings.py
import os

from langchain_openai import OpenAIEmbeddings

from backend.config import get_settings

# Embedding 모델 캐싱 (동시성 문제 해결)
_EMBEDDINGS_CACHE = None


def get_embeddings():
    """
    Return an embedding model configured via backend settings/env.

    캐싱을 사용하여 동시에 여러 tool이 실행될 때 모델 로딩 충돌을 방지합니다.
    """
    global _EMBEDDINGS_CACHE

    # 이미 로드된 모델이 있으면 재사용
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE

    settings = get_settings()
    provider = settings.embedding_provider.lower()

    # 임베딩 문맥 고려
    # - 중요한 문장이 앞에 오냐 vs 뒤에 오냐에 따라 차이가 있음
    if provider == "openai":
        # Default: OpenAI embeddings (text-embedding-3-*)
        print("Using OpenAI")
        _EMBEDDINGS_CACHE = OpenAIEmbeddings(
            model=settings.embedding_model_name,
            openai_api_key=settings.openai_api_key
        )
        return _EMBEDDINGS_CACHE

    if provider == "huggingface":
        print("Using HuggingFace")
        from langchain_huggingface import HuggingFaceEmbeddings

        # HuggingFaceEmbeddings reads auth tokens from env; ensure it is set when provided.
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if hf_token:
            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)

        # GPU 사용 설정
        model_kwargs = {"device": "cuda"}      # 또는 "cuda:0"
        # model_kwargs = {"device": "cpu"}      # 또는 "cpu"
        encode_kwargs = {"normalize_embeddings": True}

        _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return _EMBEDDINGS_CACHE

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}. "
        "Use one of ['openai', 'huggingface']."
    )
