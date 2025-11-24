"""
임베딩 모델 생성 및 관리 모듈

텍스트를 고차원 벡터로 변환하는 임베딩 모델을 제공합니다.
.env 파일의 EMBEDDING_PROVIDER 설정에 따라 OpenAI 또는 HuggingFace 임베딩을 사용합니다.

임베딩 모델은 다음 용도로 사용됩니다:
1. 과목 정보를 벡터로 변환하여 Chroma DB에 저장 (vectorstore.py)
2. 사용자 질문을 벡터로 변환하여 유사한 과목 검색 (retriever.py)
"""
# backend/rag/embeddings.py
import os

from langchain_openai import OpenAIEmbeddings

from backend.config import get_settings

# 임베딩 모델 싱글톤 캐시
# 여러 쿼리가 동시에 실행될 때 모델을 중복 로딩하지 않도록 전역 변수에 캐싱
# 특히 HuggingFace 모델은 로딩 시간이 길기 때문에 캐싱이 중요함
_EMBEDDINGS_CACHE = None


def get_embeddings():
    """
    임베딩 모델 인스턴스를 반환하는 팩토리 함수 (싱글톤 패턴)

    .env 파일의 EMBEDDING_PROVIDER 설정에 따라 적절한 임베딩 모델을 생성합니다.
    한 번 생성된 모델은 전역 캐시에 저장되어 재사용됩니다.

    ** 중요: 임베딩 모델은 반드시 일관되게 유지되어야 합니다 **
    - Vector DB를 생성할 때와 검색할 때 같은 모델을 사용해야 함
    - 모델이 바뀌면 벡터 차원이 달라져서 검색이 작동하지 않음
    - 다른 컴퓨터로 Vector DB를 옮길 때도 같은 모델 설정 필요

    지원하는 제공자:
      - openai: OpenAI의 text-embedding-3-* 모델 (예: text-embedding-3-small)
      - huggingface: HuggingFace의 임베딩 모델 (예: upskyy/bge-m3-korean)

    Returns:
        LangChain Embeddings 인스턴스 (OpenAIEmbeddings 또는 HuggingFaceEmbeddings)

    Raises:
        ValueError: 지원하지 않는 EMBEDDING_PROVIDER가 설정된 경우
    """
    global _EMBEDDINGS_CACHE

    # 이미 로드된 모델이 있으면 재사용 (싱글톤 패턴)
    # 여러 쿼리가 동시에 실행되어도 모델은 한 번만 로딩됨
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE

    settings = get_settings()
    provider = settings.embedding_provider.lower()

    # 임베딩 문맥 고려 사항
    # - 텍스트의 앞부분과 뒷부분 중 어느 쪽이 더 중요한지에 따라 임베딩 품질이 달라질 수 있음
    # - 현재는 과목 정보를 "과목명 → 설명" 순서로 구성하여 앞부분에 중요 정보 배치

    if provider == "openai":
        # OpenAI 임베딩 사용
        # 예: text-embedding-3-small (1536차원, 저렴), text-embedding-3-large (3072차원, 고품질)
        print("Using OpenAI Embeddings")
        _EMBEDDINGS_CACHE = OpenAIEmbeddings(
            model=settings.embedding_model_name,  # .env의 EMBEDDING_MODEL_NAME
            openai_api_key=settings.openai_api_key
        )
        return _EMBEDDINGS_CACHE

    if provider == "huggingface":
        # HuggingFace 임베딩 사용 (로컬 또는 Inference API)
        print("Using HuggingFace Embeddings")
        from langchain_huggingface import HuggingFaceEmbeddings

        # HuggingFace API 토큰 설정 (Inference API 사용 시 필요)
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if hf_token:
            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)

        # GPU 사용 설정 (로컬 모델 사용 시)
        # GPU가 있으면 임베딩 생성 속도가 크게 향상됨
        model_kwargs = {"device": "cuda"}      # GPU 사용 (CUDA 설치 필요)
        # model_kwargs = {"device": "cpu"}      # CPU 사용 (GPU 없을 때)

        # 임베딩 정규화 설정
        # normalize_embeddings=True: 벡터를 단위 벡터로 정규화 (코사인 유사도 계산에 유리)
        encode_kwargs = {"normalize_embeddings": True}

        _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,  # 예: "upskyy/bge-m3-korean"
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return _EMBEDDINGS_CACHE

    # 지원하지 않는 제공자
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}. "
        "Use one of ['openai', 'huggingface']."
    )
