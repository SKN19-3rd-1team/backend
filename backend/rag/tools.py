"""
ReAct 스타일 에이전트를 위한 LangChain Tools 정의.

이 파일의 함수들은 @tool 데코레이터를 사용하여 LLM이 호출할 수 있는 툴로 등록됩니다.
LLM이 자율적으로 이 툴들을 선택하고 실행할 수 있습니다.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.documents import Document

from .retriever import retrieve_with_filter
from .entity_extractor import extract_filters, build_chroma_filter


@tool  # 이 데코레이터가 함수를 LangChain Tool로 변환합니다
def retrieve_courses(
    query: str,
    university: Optional[str] = None,
    grade: Optional[str] = None,
    semester: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    대학 과목 데이터베이스에서 관련 과목을 검색합니다.

    ** 중요: 이 함수는 LLM이 자율적으로 호출할 수 있는 Tool입니다 **
    LLM은 학생의 질문을 분석한 후, 필요하다고 판단되면 이 툴을 호출합니다.

    Args:
        query: 검색 쿼리 (예: "인공지능 관련 과목", "1학년 필수 과목")
        university: 대학교 이름 (옵션, 예: "서울대학교")
        grade: 학년 (옵션, 예: "1학년", "2학년")
        semester: 학기 (옵션, 예: "1학기", "2학기")
        top_k: 검색할 과목 수 (기본값: 5)

    Returns:
        과목 리스트 [{"id": "...", "name": "...", "university": "...", ...}, ...]
    """
    # 1. 쿼리에서 필터 자동 추출 (예: "서울대 1학년" → university="서울대학교", grade="1학년")
    extracted = extract_filters(query)

    # 2. 파라미터로 받은 필터와 추출한 필터 병합 (파라미터가 우선)
    filters = extracted.copy() if extracted else {}
    if university:
        filters['university'] = university
    if grade:
        filters['grade'] = grade
    if semester:
        filters['semester'] = semester

    # 3. Chroma DB 쿼리 형식으로 필터 생성
    chroma_filter = build_chroma_filter(filters) if filters else None

    # 4. 벡터 DB에서 유사도 검색 수행
    docs: List[Document] = retrieve_with_filter(
        question=query,
        search_k=top_k,
        metadata_filter=chroma_filter
    )

    # 5. LangChain Document를 LLM이 이해하기 쉬운 Dict 형태로 변환
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


@tool  # 이 함수도 LLM이 호출할 수 있는 Tool입니다
def get_course_detail(course_id: str, courses_context: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    이전에 검색된 과목 리스트에서 특정 과목의 상세 정보를 가져옵니다.

    ** 사용 시나리오 **
    1. LLM이 먼저 retrieve_courses로 과목 리스트를 가져옴
    2. 학생이 특정 과목에 대해 더 자세히 물어봄
    3. LLM이 이 툴을 사용하여 해당 과목의 상세 정보를 조회

    Args:
        course_id: 과목 ID (예: "course_0", "course_1")
        courses_context: 이전에 retrieve_courses로 가져온 과목 리스트

    Returns:
        과목 상세 정보 {"id": "...", "name": "...", "description": "...", ...}
    """
    # 주어진 course_id와 일치하는 과목을 courses_context에서 찾아 반환
    for course in courses_context:
        if course.get("id") == course_id:
            return course

    # 해당 ID가 없으면 에러 메시지와 사용 가능한 ID 목록 반환
    return {
        "error": f"ID '{course_id}'에 해당하는 과목을 찾을 수 없습니다.",
        "available_ids": [c["id"] for c in courses_context]
    }
