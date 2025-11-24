"""
백엔드 설정 관리 모듈

이 모듈은 프로젝트 전체에서 사용되는 설정을 중앙에서 관리합니다.
.env 파일에서 환경 변수를 읽어와 Settings 데이터클래스로 제공하며,
LLM 및 임베딩 모델 인스턴스를 생성하는 팩토리 함수를 제공합니다.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from glob import glob
from dotenv import load_dotenv

# 프로젝트 루트 경로 (backend의 부모 디렉토리)
# 모든 상대 경로는 이 경로를 기준으로 해석됩니다
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# .env 파일에서 환경 변수 로드
env_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=env_path)


@dataclass
class Settings:
    """
    애플리케이션 전역 설정을 담는 데이터클래스

    모든 설정값은 .env 파일에서 읽어오며, 기본값도 제공합니다.
    """
    # API 키
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')

    # 데이터 경로 설정
    data_dir: str = os.getenv('DATA_DIR', 'backend/data')  # 데이터 디렉토리
    raw_json: str = os.getenv('RAW_JSON', 'backend/data/merged_university_courses.json')  # 원본 JSON 파일 (glob 패턴 지원)
    vector_store_path: str = os.getenv('VECTORSTORE_PATH', 'backend/data/processed/courses.parquet')  # 미사용 (레거시)
    vectorstore_dir: str = os.getenv('VECTORSTORE_DIR', 'backend/data/chroma_db')  # Chroma DB 저장 경로

    # LLM 설정
    llm_provider: str = os.getenv('LLM_PROVIDER', 'openai')  # LLM 제공자: openai, ollama, huggingface
    model_name: str = os.getenv('MODEL_NAME', 'gpt-4o-mini')  # 사용할 모델 이름

    # 임베딩 설정
    embedding_model_name: str = os.getenv('EMBEDDING_MODEL_NAME', 'text-embedding-3-small')  # 임베딩 모델 (한국어 특화)
    embedding_provider: str = os.getenv('EMBEDDING_PROVIDER', 'openai')  # 임베딩 제공자: openai, huggingface


def get_settings() -> Settings:
    """
    설정 인스턴스 반환

    Returns:
        Settings: .env 파일에서 로드된 설정값을 담은 Settings 인스턴스
    """
    return Settings()

def get_llm():
    """
    LLM(대형 언어 모델) 인스턴스를 생성하는 팩토리 함수

    .env 파일의 LLM_PROVIDER 설정에 따라 적절한 LangChain ChatModel을 생성합니다.

    지원하는 제공자:
      - openai: OpenAI API 또는 호환 서버 (vLLM, Together AI 등)
      - ollama: 로컬 Ollama 서버
      - huggingface: Hugging Face Inference API

    Returns:
        LangChain ChatModel 인스턴스 (ChatOpenAI, ChatOllama, ChatHuggingFace 중 하나)

    Raises:
        ValueError: 지원하지 않는 LLM_PROVIDER가 설정된 경우
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "openai":
        # OpenAI API 또는 호환 서버 사용
        # OPENAI_API_KEY는 환경 변수에서 자동으로 읽어옵니다
        from langchain_openai import ChatOpenAI

        # OPENAI_API_BASE 지원 (vLLM, Together AI, Anyscale 등 OpenAI 호환 서버용)
        base_url = os.getenv("OPENAI_API_BASE", None)

        if base_url:
            # 커스텀 API 서버 사용
            return ChatOpenAI(
                model=settings.model_name,
                base_url=base_url,  # OpenAI 호환 API 서버 주소
                api_key=settings.openai_api_key
            )
        else:
            # 공식 OpenAI API 사용
            return ChatOpenAI(model=settings.model_name)

    elif provider == "ollama":
        # 로컬 또는 원격 Ollama 서버 사용
        # 예: MODEL_NAME=llama3.2:1b, qwen2.5:7b-instruct 등
        try:
            # 최신 langchain_ollama 패키지 사용 시도
            from langchain_ollama import ChatOllama
        except ImportError:
            # 폴백: 구버전 langchain_community 사용
            from langchain_community.chat_models import ChatOllama

        # OLLAMA_BASE_URL 환경 변수로 원격 서버 지정 가능
        # 로컬: http://localhost:11434 (기본값)
        # RunPod: http://YOUR_RUNPOD_IP:11434
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        return ChatOllama(
            model=settings.model_name,
            temperature=0.7,  # 창의성 조절 (0.0 = 결정적, 1.0 = 창의적)
            base_url=base_url,  # .env의 OLLAMA_BASE_URL 또는 기본값
        )

    elif provider == "huggingface":
        # Hugging Face Inference Endpoint / Hub 모델 사용
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

        # HuggingFace 엔드포인트 생성
        endpoint = HuggingFaceEndpoint(
            repo_id=settings.model_name,  # 예: "Qwen/Qwen2.5-7B-Instruct"
            huggingfacehub_api_token=hf_token or None,
        )
        return ChatHuggingFace(llm=endpoint)

    else:
        # 지원하지 않는 제공자
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {settings.llm_provider}. "
            "Use one of ['openai', 'ollama', 'huggingface']."
        )


def resolve_path(path_str: str) -> Path:
    """
    경로 문자열을 프로젝트 루트 기준의 절대 경로로 변환

    상대 경로가 주어지면 프로젝트 루트(PROJECT_ROOT)를 기준으로 변환하고,
    절대 경로가 주어지면 그대로 반환합니다.

    Args:
        path_str: 절대 경로 또는 상대 경로 문자열
                  예: "backend/data" 또는 "D:/Github/project/backend/data"

    Returns:
        Path: 절대 경로 Path 객체
    """
    path = Path(path_str)
    if path.is_absolute():
        return path  # 이미 절대 경로면 그대로 반환
    return PROJECT_ROOT / path  # 상대 경로면 프로젝트 루트 기준으로 변환


def expand_paths(path_pattern: str) -> list[Path]:
    """
    Glob 패턴을 확장하여 매칭되는 모든 파일 경로를 반환

    프로젝트 루트를 기준으로 glob 패턴을 확장합니다.
    여러 데이터 파일을 한 번에 로드할 때 유용합니다.

    Args:
        path_pattern: Glob 패턴 또는 직접 경로
                      예: "backend/data/raw/*.json" → raw 폴더의 모든 JSON 파일
                          "backend/data/*.json" → data 폴더의 모든 JSON 파일

    Returns:
        list[Path]: 패턴에 매칭되는 파일들의 경로 리스트

    Raises:
        FileNotFoundError: 패턴에 매칭되는 파일이 하나도 없는 경우
    """
    pattern_path = resolve_path(path_pattern)
    matches = [Path(p) for p in glob(str(pattern_path))]
    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {path_pattern}")
    return matches
