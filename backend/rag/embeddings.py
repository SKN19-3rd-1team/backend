# backend/rag/embeddings.py
import os

from langchain_openai import OpenAIEmbeddings

from backend.config import get_settings


def get_embeddings():
    """Return an embedding model configured via backend settings/env."""
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        # Default: OpenAI embeddings (text-embedding-3-*)
        print("Using OpenAI")
        return OpenAIEmbeddings(
            model=settings.embedding_model_name,
            openai_api_key=settings.openai_api_key
        )

    if provider == "huggingface":
        print("Using HuggingFace")
        from langchain_huggingface import HuggingFaceEmbeddings

        # HuggingFaceEmbeddings reads auth tokens from env; ensure it is set when provided.
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if hf_token:
            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)

        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
        )

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}. "
        "Use one of ['openai', 'huggingface']."
    )
