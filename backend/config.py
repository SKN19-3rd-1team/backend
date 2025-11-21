"""Centralized configuration for the backend."""

import os
from dataclasses import dataclass
from pathlib import Path
from glob import glob
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables from .env file
env_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=env_path)


@dataclass
class Settings:
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    data_dir: str = os.getenv('DATA_DIR', 'backend/data')
    raw_json: str = os.getenv('RAW_JSON', 'backend/data/merged_university_courses.json')
    vector_store_path: str = os.getenv('VECTORSTORE_PATH', 'backend/data/processed/courses.parquet')
    vectorstore_dir: str = os.getenv('VECTORSTORE_DIR', 'backend/data/chroma_db')
    llm_provider: str = os.getenv('LLM_PROVIDER', 'huggingface')
    model_name: str = os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-7B-Instruct')
    embedding_model_name: str = os.getenv('EMBEDDING_MODEL_NAME', 'upskyy/bge-m3-korean')
    embedding_provider: str = os.getenv('EMBEDDING_PROVIDER', 'huggingface')


def get_settings() -> Settings:
    """Return default settings."""
    return Settings()

def get_llm():
    """
    현재 설정(Settings)에 맞춰 LangChain ChatModel 인스턴스를 생성해서 반환.

    지원 예시:
      - LLM_PROVIDER=openai    -> ChatOpenAI
      - LLM_PROVIDER=ollama    -> ChatOllama
      - LLM_PROVIDER=huggingface -> ChatHuggingFace(HuggingFaceEndpoint)
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "openai":
        # OPENAI_API_KEY는 env 에서 자동으로 읽음
        from langchain_openai import ChatOpenAI

        # OPENAI_API_BASE 지원 (vLLM, Together AI, Anyscale 등 OpenAI 호환 서버용)
        base_url = os.getenv("OPENAI_API_BASE", None)

        if base_url:
            return ChatOpenAI(
                model=settings.model_name,
                base_url=base_url,  # OpenAI 호환 API 서버 주소
                api_key=settings.openai_api_key
            )
        else:
            return ChatOpenAI(model=settings.model_name)

    elif provider == "ollama":
        # 예: MODEL_NAME=llama3.2:1b, qwen2.5:7b-instruct 등
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=settings.model_name)

    elif provider == "huggingface":
        # Hugging Face Inference Endpoint / Hub 모델 사용 예시
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

        endpoint = HuggingFaceEndpoint(
            repo_id=settings.model_name,
            huggingfacehub_api_token=hf_token or None,
        )
        return ChatHuggingFace(llm=endpoint)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {settings.llm_provider}. "
            "Use one of ['openai', 'ollama', 'huggingface']."
        )


def resolve_path(path_str: str) -> Path:
    """
    Resolve a path that might be relative to the project root.

    Args:
        path_str: Absolute or relative path string.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def expand_paths(path_pattern: str) -> list[Path]:
    """
    Expand glob patterns (or direct paths) relative to the project root.

    Args:
        path_pattern: Pattern such as "backend/data/raw/*.json".
    """
    pattern_path = resolve_path(path_pattern)
    matches = [Path(p) for p in glob(str(pattern_path))]
    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {path_pattern}")
    return matches
