"""
ReAct 스타일 에이전트를 위한 LangChain Tools 정의.

이 파일의 함수들은 @tool 데코레이터를 사용하여 LLM이 호출할 수 있는 툴로 등록됩니다.
LLM이 자율적으로 이 툴들을 선택하고 실행할 수 있습니다.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.documents import Document
import numpy as np

from .retriever import retrieve_with_filter
from .entity_extractor import extract_filters, build_chroma_filter
from .vectorstore import load_vectorstore
from .embeddings import get_embeddings


@tool
def retrieve_courses(
    query: str,
    university: Optional[str] = None,
    college: Optional[str] = None,
    department: Optional[str] = None,
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
        university: 대학교 이름 (옵션, 예: "서울대학교", "홍익대학교")
        college: 단과대학 이름 (옵션, 예: "공과대학", "자연과학대학")
        department: 학과 이름 (옵션, 예: "컴퓨터공학", "전자공학")
        grade: 학년 (옵션, 예: "1학년", "2학년")
        semester: 학기 (옵션, 예: "1학기", "2학기")
        top_k: 검색할 과목 수 (기본값: 5)

    Returns:
        과목 리스트 [{"id": "...", "name": "...", "university": "...", ...}, ...]
    """
    print("Using retreive_courses tools.")
    # 1. 쿼리에서 필터 자동 추출 (예: "서울대 컴퓨터공학과 1학년" → university, department, grade)
    extracted = extract_filters(query)
    print(extracted)

    # 2. 파라미터로 받은 필터와 추출한 필터 병합 (파라미터가 우선)
    filters = extracted.copy() if extracted else {}
    if university:
        filters['university'] = university
    if college:
        filters['college'] = college
    if department:
        filters['department'] = department
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
    print(results)

    return results


@tool
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
    print("Using get_course_detail tools.")
    # 주어진 course_id와 일치하는 과목을 courses_context에서 찾아 반환
    for course in courses_context:
        if course.get("id") == course_id:
            return course

    # 해당 ID가 없으면 에러 메시지와 사용 가능한 ID 목록 반환
    return {
        "error": f"ID '{course_id}'에 해당하는 과목을 찾을 수 없습니다.",
        "available_ids": [c["id"] for c in courses_context]
    }


@tool
def list_departments(query: str) -> List[Dict[str, str]]:
    """
    Vector DB에 있는 모든 학과 목록을 조회합니다.
    학생이 "어떤 학과가 있어?", "컴퓨터 관련 학과 알려줘" 같은 질문을 할 때 사용하세요.

    ** 중요: 이 함수는 LLM이 자율적으로 호출할 수 있는 Tool입니다 **
    학생이 학과 목록, 학과 종류, 전공 리스트를 물어보면 이 툴을 호출하세요.

    ** 사용 시나리오 **
    1. "어떤 학과들이 있어?" -> query="전체" 로 호출
    2. "컴퓨터 관련 학과 알려줘" -> query="컴퓨터" 로 호출
    3. "공대에는 어떤 학과가 있어?" -> query="공학" 로 호출

    Args:
        query: 검색할 키워드 (예: "컴퓨터", "공학", "전체", "전자")
                "전체"를 입력하면 모든 학과를 반환합니다.

    Returns:
        학과 리스트 [
            {"university": "서울대학교", "college": "공과대학", "department": "컴퓨터공학"},
            {"university": "홍익대학교", "college": "공과대학", "department": "컴퓨터공학"},
            ...
        ]
    """
    print(f"Using list_departments tool with query: {query}")

    vs = load_vectorstore()
    collection = vs._collection

    # 모든 메타데이터 가져오기
    results = collection.get(include=['metadatas'])

    # 학과 정보 추출 (중복 제거)
    departments_set = set()
    for meta in results['metadatas']:
        university = meta.get('university', '')
        college = meta.get('college', '')
        department = meta.get('department', '')

        if department:
            # Tuple로 중복 제거
            departments_set.add((university, college, department))

    # 리스트로 변환
    all_departments = [
        {
            "university": univ,
            "college": college,
            "department": dept
        }
        for univ, college, dept in sorted(departments_set)
    ]

    # 쿼리 필터링
    if query.strip() == "전체" or not query.strip():
        result = all_departments
    else:
        # 키워드로 필터링 (대학, 단과대학, 학과명에서 검색)
        query_lower = query.lower()
        result = [
            dept_info for dept_info in all_departments
            if (query_lower in dept_info['university'].lower() or
                query_lower in dept_info['college'].lower() or
                query_lower in dept_info['department'].lower())
        ]

    print(f"Found {len(result)} departments matching '{query}'")
    return result


@tool
def recommend_curriculum(
    university: str,
    department: str,
    interests: Optional[str] = None,
    start_grade: int = 2,
    start_semester: int = 1,
    end_grade: int = 4,
    end_semester: int = 2
) -> List[Dict[str, Any]]:
    """
    학생의 관심사를 고려하여 학기별 맞춤 커리큘럼을 추천합니다.

    ** 중요: 이 함수는 LLM이 자율적으로 호출할 수 있는 Tool입니다 **
    학생이 "2학년부터 4학년까지 커리큘럼 추천해줘", "전체 커리큘럼 알려줘" 같은 질문을 할 때 사용하세요.

    ** 사용 시나리오 **
    1. "홍익대 컴퓨터공학과 2학년부터 4학년까지 커리큘럼 추천해줘"
       → university="홍익대학교", department="컴퓨터공학", start_grade=2, end_grade=4
    2. "인공지능에 관심있는데 커리큘럼 추천해줘"
       → interests="인공지능"으로 호출하여 관련 과목 우선 선택

    Args:
        university: 대학교 이름 (예: "홍익대학교", "서울대학교")
        department: 학과 이름 (예: "컴퓨터공학", "전자공학")
        interests: 학생의 관심 분야 키워드 (예: "인공지능", "데이터분석", "보안")
        start_grade: 시작 학년 (기본값: 2)
        start_semester: 시작 학기 (기본값: 1)
        end_grade: 종료 학년 (기본값: 4)
        end_semester: 종료 학기 (기본값: 2)

    Returns:
        학기별 추천 과목 리스트 [
            {
                "semester": "2학년 1학기",
                "course": {"name": "...", "description": "...", "classification": "..."},
                "reason": "인공지능 관심사와 관련이 높음"
            },
            ...
        ]
    """
    print(f"Using recommend_curriculum tool: {university} {department}, interests='{interests}'")

    vs = load_vectorstore()
    embeddings = get_embeddings()

    # 관심사 임베딩 생성 (있는 경우)
    interests_embedding = None
    if interests:
        interests_embedding = embeddings.embed_query(interests)

    curriculum = []

    # 학기별로 반복
    for grade in range(start_grade, end_grade + 1):
        for semester in range(1, 3):  # 1학기, 2학기
            # 종료 조건 확인
            if grade == end_grade and semester > end_semester:
                break
            if grade == start_grade and semester < start_semester:
                continue

            semester_label = f"{grade}학년 {semester}학기"

            # 해당 학기의 과목 검색
            filter_dict = {
                'university': university,
                'department': department,
                'grade': f"{grade}학년",
                'semester': f"{semester}학기"
            }

            chroma_filter = build_chroma_filter(filter_dict)

            try:
                # 해당 학기 과목 검색
                docs = retrieve_with_filter(
                    question=interests if interests else "추천 과목",
                    search_k=10,  # 후보 많이 가져오기
                    metadata_filter=chroma_filter
                )

                if not docs:
                    curriculum.append({
                        "semester": semester_label,
                        "course": None,
                        "reason": "해당 학기에 개설된 과목이 없습니다."
                    })
                    continue

                # 관심사가 있으면 유사도 기반으로 정렬
                if interests_embedding:
                    doc_scores = []
                    for doc in docs:
                        # 과목 설명 임베딩
                        desc = doc.page_content
                        doc_embedding = embeddings.embed_query(desc)

                        # 코사인 유사도 계산
                        similarity = np.dot(interests_embedding, doc_embedding) / (
                            np.linalg.norm(interests_embedding) * np.linalg.norm(doc_embedding)
                        )
                        doc_scores.append((doc, similarity))

                    # 유사도 높은 순으로 정렬
                    doc_scores.sort(key=lambda x: x[1], reverse=True)
                    best_doc, score = doc_scores[0]
                    reason = f"'{interests}' 관심사와 유사도 {score:.2f}"
                else:
                    # 관심사 없으면 첫 번째 과목 선택
                    best_doc = docs[0]
                    reason = "해당 학기 대표 과목"

                meta = best_doc.metadata
                curriculum.append({
                    "semester": semester_label,
                    "course": {
                        "name": meta.get("name", "[이름 없음]"),
                        "classification": meta.get("course_classification", "[정보 없음]"),
                        "description": best_doc.page_content
                    },
                    "reason": reason
                })

            except Exception as e:
                print(f"Error retrieving courses for {semester_label}: {e}")
                curriculum.append({
                    "semester": semester_label,
                    "course": None,
                    "reason": f"검색 중 오류 발생: {str(e)}"
                })

    print(f"Generated curriculum with {len(curriculum)} semesters")
    return curriculum
