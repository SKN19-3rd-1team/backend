"""
벡터 DB 검색 (Retrieval) 모듈

전공 추천을 위한 Pinecone 기반 전공 문서 검색 기능을 제공합니다.

** 주요 기능 **
1. 전공 문서 검색: 사용자 질문과 유사한 전공 문서를 검색
2. 점수 집계: 문서 타입별 가중치를 적용하여 전공별 최종 점수 산출
3. SearchHit 구조: 일관된 검색 결과 형식 제공

** 검색 과정 **
1. 질문을 벡터로 변환 (임베딩 모델 사용)
2. Pinecone 벡터 DB에서 유사한 전공 문서 검색 (코사인 유사도)
3. 문서 타입별 점수를 가중치 적용하여 전공별로 집계
"""
# backend/rag/retriever.py
from dataclasses import dataclass
from typing import Dict, List, Any

from .vectorstore import get_major_vectorstore


# Pinecone 검색 결과를 일관된 구조로 다루기 위한 헬퍼 데이터클래스
@dataclass
class SearchHit:
    doc_id: str
    major_id: str
    major_name: str
    doc_type: str
    score: float
    metadata: Dict[str, Any]
    text: str


def search_major_docs(
    query_embedding: List[float],
    top_k: int = 50,
) -> List[SearchHit]:
    """
    Pinecone 전공 인덱스에서 주어진 임베딩과 가장 유사한 문서들을 조회한다.

    Args:
        query_embedding: 사용자 질의/프로필을 임베딩한 벡터 값
        top_k: 상위 몇 개의 문서를 반환할지 결정 (기본 50개)

    Returns:
        SearchHit 객체 리스트 (문서별 점수, 메타데이터 포함)
    """
    vectorstore = get_major_vectorstore()
    try:
        results = vectorstore.similarity_search_by_vector_with_relevance_scores(
            embedding=query_embedding,
            k=top_k,
        )
    except AttributeError:
        # langchain_pinecone < 0.2.16 버전 호환: with_relevance_scores 헬퍼가 없을 때 대체 경로 사용
        results = vectorstore.similarity_search_by_vector_with_score(
            embedding=query_embedding,
            k=top_k,
        )

    hits: List[SearchHit] = []
    for doc, score in results:
        # LangChain이 반환한 Document/score 튜플을 SearchHit로 래핑
        metadata = dict(doc.metadata or {})
        hit = SearchHit(
            doc_id=metadata.get("doc_id") or metadata.get("id") or "",
            major_id=metadata.get("major_id") or "",
            major_name=metadata.get("major_name") or "",
            doc_type=metadata.get("doc_type") or "unknown",
            score=float(score),
            metadata=metadata,
            text=doc.page_content or "",
        )
        hits.append(hit)

    if not hits:
        print("[Majors] ⚠️  Pinecone returned no results")
    else:
        print(f"[Majors] ✅ Pinecone search returned {len(hits)} hits")
        for hit in hits[:5]:
            print(
                f"   - {hit.major_name} ({hit.doc_type}) "
                f"score={hit.score:.3f}, major_id={hit.major_id}"
            )

    return hits


def aggregate_major_scores(
    hits: List[SearchHit],
    doc_type_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    문서 타입별 가중치를 반영하여 전공 단위로 점수를 합산한다.

    Args:
        hits: search_major_docs 결과 목록
        doc_type_weights: doc_type → 가중치 매핑 딕셔너리

    Returns:
        major_id를 키로 하고 가중 합산 점수를 값으로 가지는 딕셔너리
    """
    # 전공별 doc_type 최고 점수를 저장
    per_major: Dict[str, Dict[str, float]] = {}
    for hit in hits:
        if not hit.major_id:
            continue
        per_type = per_major.setdefault(hit.major_id, {})
        current = per_type.get(hit.doc_type)
        if current is None or hit.score > current:
            per_type[hit.doc_type] = hit.score

    aggregated: Dict[str, float] = {}
    for major_id, type_scores in per_major.items():
        total = 0.0
        for doc_type, score in type_scores.items():
            weight = doc_type_weights.get(doc_type, 1.0)
            total += score * weight
        aggregated[major_id] = total

    return aggregated
