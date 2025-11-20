"""LangChain tools for ReAct-style agent."""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.documents import Document

from .retriever import retrieve_with_filter
from .entity_extractor import extract_filters, build_chroma_filter


@tool
def retrieve_courses(
    query: str,
    university: Optional[str] = None,
    grade: Optional[str] = None,
    semester: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    대학 과목 데이터베이스에서 관련 과목을 검색합니다.

    Args:
        query: 검색 쿼리 (예: "인공지능 관련 과목", "1학년 필수 과목")
        university: 대학교 이름 (옵션, 예: "서울대학교")
        grade: 학년 (옵션, 예: "1학년", "2학년")
        semester: 학기 (옵션, 예: "1학기", "2학기")
        top_k: 검색할 과목 수 (기본값: 5)

    Returns:
        과목 리스트 [{"id": "...", "name": "...", "university": "...", ...}, ...]
    """
    # 쿼리에서 필터 자동 추출
    extracted = extract_filters(query)

    # 파라미터로 받은 필터와 추출한 필터 병합 (파라미터가 우선)
    filters = extracted.copy() if extracted else {}
    if university:
        filters['university'] = university
    if grade:
        filters['grade'] = grade
    if semester:
        filters['semester'] = semester

    # Chroma 필터 생성
    chroma_filter = build_chroma_filter(filters) if filters else None

    # 검색 수행
    docs: List[Document] = retrieve_with_filter(
        question=query,
        search_k=top_k,
        metadata_filter=chroma_filter
    )

    # Document를 Dict 형태로 변환
    results = []
    for idx, doc in enumerate(docs):
        meta = doc.metadata
        results.append({
            "id": f"course_{idx}",
            "name": meta.get("name", "[이름 없음]"),
            "university": meta.get("university", "[정보 없음]"),
            "college": meta.get("college", "[정보 없음]"),
            "department": meta.get("department", "[정보 없음]"),
            "grade_semester": meta.get("grade_semester", "[정보 없음]"),
            "classification": meta.get("course_classification", "[정보 없음]"),
            "description": doc.page_content or "[설명 정보가 제공되지 않았습니다]"
        })

    return results


@tool
def get_course_detail(course_id: str, courses_context: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    이전에 검색된 과목 리스트에서 특정 과목의 상세 정보를 가져옵니다.

    Args:
        course_id: 과목 ID (예: "course_0", "course_1")
        courses_context: 이전에 retrieve_courses로 가져온 과목 리스트

    Returns:
        과목 상세 정보 {"id": "...", "name": "...", "description": "...", ...}
    """
    for course in courses_context:
        if course.get("id") == course_id:
            return course

    return {
        "error": f"ID '{course_id}'에 해당하는 과목을 찾을 수 없습니다.",
        "available_ids": [c["id"] for c in courses_context]
    }
