"""
벡터 데이터베이스 (Vector Store) 관리 모듈

Pinecone을 사용하여 전공 정보를 벡터로 저장하고 검색하는 기능을 제공합니다.

** Pinecone이란? **
- 클라우드 기반 벡터 데이터베이스
- 텍스트를 벡터(임베딩)로 변환하여 저장
- 유사도 기반 검색 (Similarity Search) 지원
- Serverless 아키텍처로 확장성 제공

** 주요 기능 **
1. get_major_vectorstore(): 전공 추천을 위한 Pinecone 벡터 스토어 반환
2. index_major_docs(): 전공 문서를 Pinecone에 인덱싱
3. clear_major_index(): Pinecone 인덱스 초기화
"""
# backend/rag/vectorstore.py
from __future__ import annotations

from typing import Any
import threading

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException

from backend.config import get_settings
from .embeddings import get_embeddings
from .loader import MajorDoc

# Pinecone (majors) caches
_MAJOR_VECTORSTORE_CACHE = None
_MAJOR_VECTORSTORE_LOCK = threading.Lock()
_MAJOR_INDEX_CACHE = None


# ==================== Pinecone Vector Store for Majors ====================

def _get_pinecone_client() -> Pinecone:
    # Pinecone API 클라이언트를 초기화하고 키 누락 시 명확한 에러를 던진다
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set in environment or .env file.")
    return Pinecone(api_key=settings.pinecone_api_key)


def _list_index_names(client: Pinecone) -> list[str]:
    # 서버 버전에 따라 달라질 수 있는 list_indexes 응답을 문자열 리스트로 정규화
    response = client.list_indexes()
    if isinstance(response, dict) and "indexes" in response:
        items = response["indexes"]
        names: list[str] = []
        for item in items:
            if isinstance(item, dict):
                names.append(item.get("name"))
            else:
                names.append(getattr(item, "name", None))
        return [name for name in names if name]

    if hasattr(response, "indexes"):
        return [idx.get("name") if isinstance(idx, dict)
                else getattr(idx, "name", None)
                for idx in getattr(response, "indexes")]  # type: ignore[arg-type]

    if hasattr(response, "names") and callable(response.names):
        return list(response.names())

    if isinstance(response, list):
        names = []
        for item in response:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                names.append(item.get("name"))
            else:
                names.append(getattr(item, "name", None))
        return [name for name in names if name]

    return []


def _infer_embedding_dimension(embeddings) -> int:
    # 설정에 명시된 차원이 없으면 임베딩 모델에서 한 번 추론하여 차원을 구한다
    settings = get_settings()
    if settings.pinecone_dimension:
        return settings.pinecone_dimension
    probe_vector = embeddings.embed_query("major matching dimension probe")
    return len(probe_vector)


def _get_region_and_cloud(settings):
    # serverless 인덱스 생성을 위해 region/cloud 정보를 읽어온다
    region = settings.pinecone_region or settings.pinecone_environment
    if not region:
        raise ValueError("Set PINECONE_REGION or PINECONE_ENVIRONMENT for Pinecone.")
    cloud = settings.pinecone_cloud or "aws"
    return region, cloud


def _ensure_major_index(embeddings):
    # Pinecone 인덱스가 없으면 생성하고 있으면 핸들을 재사용
    global _MAJOR_INDEX_CACHE
    if _MAJOR_INDEX_CACHE is not None:
        return _MAJOR_INDEX_CACHE

    settings = get_settings()
    client = _get_pinecone_client()
    index_name = settings.pinecone_index_name
    existing = _list_index_names(client)
    dimension = _infer_embedding_dimension(embeddings)

    if index_name not in existing:
        region, cloud = _get_region_and_cloud(settings)
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    _MAJOR_INDEX_CACHE = client.Index(index_name)
    return _MAJOR_INDEX_CACHE


def _get_major_namespace() -> str | None:
    settings = get_settings()
    namespace = settings.pinecone_namespace
    if namespace is None:
        return None
    namespace = namespace.strip()
    return namespace or None

def get_major_index():
    # LangChain 외부에서 직접 인덱스 핸들이 필요할 때 사용
    embeddings = get_embeddings()
    return _ensure_major_index(embeddings)


def get_major_vectorstore():
    """
    전공 추천에 특화된 Pinecone 기반 LangChain VectorStore를 싱글톤으로 반환한다.

    Index 핸들, 임베딩 모델, namespace 설정을 한 번만 구성한 뒤 재사용하여
    불필요한 API 호출을 줄이고 스레드 안정성을 확보한다.
    """
    # LangChain VectorStore 인터페이스를 재사용하기 위해 싱글톤으로 구성
    global _MAJOR_VECTORSTORE_CACHE
    with _MAJOR_VECTORSTORE_LOCK:
        if _MAJOR_VECTORSTORE_CACHE is not None:
            return _MAJOR_VECTORSTORE_CACHE

        embeddings = get_embeddings()
        index = _ensure_major_index(embeddings)
        namespace = _get_major_namespace()
        _MAJOR_VECTORSTORE_CACHE = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text",
            namespace=namespace,
        )
        return _MAJOR_VECTORSTORE_CACHE


def clear_major_index(namespace: str | None = None):
    """
    Pinecone 인덱스에서 지정된 namespace를 전부 삭제한다.

    Args:
        namespace: 비우고 싶은 네임스페이스. None이면 기본값을 사용.
    """
    # 인덱스를 재구축하기 전 기존 벡터를 깨끗하게 제거
    index = get_major_index()
    delete_kwargs: dict[str, Any] = {"deleteAll": True}
    namespace = namespace if namespace is not None else _get_major_namespace()
    if namespace:
        delete_kwargs["namespace"] = namespace
    try:
        index.delete(**delete_kwargs)
    except NotFoundException:
        # 삭제 대상 네임스페이스가 없으면 무시 (이미 비어있는 상태)
        pass


def index_major_docs(docs: list[MajorDoc]) -> int:
    """
    MajorDoc 리스트를 Pinecone 인덱스에 업서트하고 실제로 업로드한 문서 수를 반환한다.

    Args:
        docs: Pinecone에 저장할 전공 문서(요약, 과목, 진로 등) 목록

    Returns:
        업서트된 문서 수 (int)
    """
    vectorstore = get_major_vectorstore()

    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for doc in docs:
        texts.append(doc.text)
        ids.append(doc.doc_id)

        meta: dict[str, Any] = {
            "major_id": doc.major_id,
            "major_name": doc.major_name,
            "doc_type": doc.doc_type,
        }

        # cluster: None이면 넣지 않기
        if doc.cluster is not None and doc.cluster != "":
            meta["cluster"] = doc.cluster

        # salary: None이 아닐 때만 숫자로 넣기
        if doc.salary is not None:
            meta["salary"] = float(doc.salary)

        # 태그 리스트: 비어있지 않을 때만 넣기 (list[str] 형태 유지)
        if getattr(doc, "relate_subject_tags", None):
            meta["relate_subject_tags"] = doc.relate_subject_tags

        if getattr(doc, "job_tags", None):
            meta["job_tags"] = doc.job_tags

        metadatas.append(meta)

    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(docs)
