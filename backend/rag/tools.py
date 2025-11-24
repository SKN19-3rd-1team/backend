"""
ReAct 스타일 에이전트를 위한 LangChain Tools 정의

이 파일의 함수들은 @tool 데코레이터를 사용하여 LLM이 호출할 수 있는 툴로 등록됩니다.

** ReAct 패턴에서의 툴 역할 **
LLM이 사용자 질문을 분석하고, 필요시 자율적으로 이 툴들을 호출하여 정보를 수집합니다.
예: "홍익대 컴공 과목 알려줘" → LLM이 retrieve_courses 툴 호출 결정 → 과목 정보 검색 → 답변 생성

** 제공되는 툴들 **
1. retrieve_courses: 과목 검색 (메인 툴, 가장 자주 사용됨)
2. list_departments: 학과 목록 조회 (목록만 필요할 때)
3. recommend_curriculum: 학기별 커리큘럼 추천 (여러 학기 계획)
4. get_search_help: 검색 실패 시 사용 가이드 제공
5. get_course_detail: 특정 과목 상세 정보 (현재 미사용)

** 작동 방식 **
1. LLM이 사용자 질문 분석
2. LLM이 필요한 툴 선택 및 파라미터 결정
3. 툴 실행 (이 파일의 함수 호출)
4. 툴 결과를 LLM에게 전달
5. LLM이 결과를 바탕으로 최종 답변 생성
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.documents import Document
import numpy as np

from .retriever import retrieve_with_filter
from .entity_extractor import extract_filters, build_chroma_filter
from .vectorstore import load_vectorstore
from .embeddings import get_embeddings


def _get_tool_usage_guide() -> str:
    """
    사용자에게 제공할 툴 사용 가이드 메시지를 생성합니다.
    """
    return """
검색 가능한 방법들:

1. **특정 과목 검색**
   - 예시: "인공지능 관련 과목 추천해줘", "1학년 필수 과목 알려줘"
   - 검색어에 과목명, 학년, 학기, 대학명 등을 포함할 수 있습니다

2. **학과 목록 조회**
   - 예시: "어떤 학과들이 있어?", "컴퓨터 관련 학과 알려줘", "공대에는 어떤 학과가 있어?"
   - 전체 학과 목록 또는 키워드로 필터링된 학과를 확인할 수 있습니다

3. **커리큘럼 추천**
   - 예시: "홍익대 컴퓨터공학과 2학년부터 4학년까지 커리큘럼 추천해줘"
   - 예시: "인공지능에 관심있는데 전체 커리큘럼 알려줘"
   - 학기별로 맞춤 과목을 추천받을 수 있습니다

더 구체적인 질문을 해주시면 더 정확한 정보를 제공해드릴 수 있습니다!
"""


@tool
def retrieve_courses(
    query: Optional[str] = None,
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
    ** 학생이 특정 대학, 학과, 과목에 대해 질문하면 반드시 이 툴을 먼저 호출해야 합니다! **

    ** 필수 사용 상황 **
    - 학생이 특정 대학/학과를 언급할 때 (예: "홍익대학교 컴퓨터공학", "서울대 전자공학과")
    - 학생이 과목 추천을 요청할 때 (예: "인공지능 과목 추천해줘", "1학년 필수 과목")
    - 학생이 특정 분야 과목을 물어볼 때 (예: "데이터분석 과목", "네트워크 관련 수업")

    ** 호출 방법 **
    1. query만 사용: retrieve_courses(query="홍익대학교 컴퓨터공학")
    2. 파라미터만 사용: retrieve_courses(university="홍익대학교", department="컴퓨터공학")
    3. 혼합 사용: retrieve_courses(query="인공지능", university="홍익대학교")

    Args:
        query: 검색 쿼리 (옵션, 예: "인공지능 관련 과목", "1학년 필수 과목")
               query가 없으면 다른 파라미터들로 자동 생성됩니다.
        university: 대학교 이름 (옵션, 예: "서울대학교", "홍익대학교")
        college: 단과대학 이름 (옵션, 예: "공과대학", "자연과학대학")
        department: 학과 이름 (옵션, 예: "컴퓨터공학", "전자공학")
        grade: 학년 (옵션, 예: "1학년", "2학년")
        semester: 학기 (옵션, 예: "1학기", "2학기")
        top_k: 검색할 과목 수 (기본값: 5)

    Returns:
        과목 리스트 [{"id": "...", "name": "...", "university": "...", ...}, ...]
    """
    # query가 없으면 다른 파라미터들로부터 자동 생성
    auto_generated = False
    if not query:
        query_parts = []
        if university:
            query_parts.append(university)
        if college:
            query_parts.append(college)
        if department:
            query_parts.append(department)
        if grade:
            query_parts.append(grade)
        if semester:
            query_parts.append(semester)

        if query_parts:
            query = " ".join(query_parts)
            auto_generated = True
        else:
            # 아무 파라미터도 없으면 기본 쿼리
            query = "추천 과목"
            auto_generated = True

    if auto_generated:
        print(f"✅ Using retrieve_courses tool (auto-generated query: '{query}')")
        print(f"   Params: university={university}, college={college}, department={department}, grade={grade}, semester={semester}")
    else:
        print(f"✅ Using retrieve_courses tool with query: '{query}'")
    # 1. 쿼리에서 필터 자동 추출 (예: "서울대 컴퓨터공학과 1학년" → university, department, grade)
    extracted = extract_filters(query)
    print(f"   Extracted filters: {extracted}")

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

    # 5. 검색 결과가 없을 때 예외처리
    if not docs:
        print(f"⚠️  WARNING: No courses found for query='{query}', filters={chroma_filter}")
        return [{
            "error": "no_results",
            "message": "사용자 질문에 대한 정보를 가져올 수 없었습니다.",
            "suggestion": "get_search_help 툴을 사용하여 검색 가능한 방법을 안내하세요."
        }]

    # 6. LangChain Document를 LLM이 이해하기 쉬운 Dict 형태로 변환
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

    print(f"✅ Found {len(results)} courses")
    for r in results[:3]:  # 처음 3개만 출력
        print(f"   - {r['name']} ({r['university']} {r['department']})")

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
    print(f"✅ Using get_course_detail tool for course_id: {course_id}")
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
    Vector DB에 있는 학과 목록을 조회합니다.

    ** 중요: 이 툴은 학과 **목록 조회**에만 사용하세요! **
    ** ⚠️ 특정 대학/학과의 과목 정보가 필요하면 retrieve_courses를 사용하세요! **

    ** 올바른 사용 시나리오 (목록 조회) **
    ✅ "어떤 학과들이 있어?" -> query="전체" 로 호출
    ✅ "컴퓨터 관련 학과 목록 알려줘" -> query="컴퓨터" 로 호출
    ✅ "공대에는 어떤 학과가 있어?" -> query="공학" 로 호출

    ** 잘못된 사용 시나리오 (과목 정보 필요) **
    ❌ "홍익대학교 컴퓨터공학" -> retrieve_courses를 사용해야 함
    ❌ "서울대 전자공학과" -> retrieve_courses를 사용해야 함
    ❌ "컴퓨터공학 과목 추천해줘" -> retrieve_courses를 사용해야 함

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
    print(f"✅ Using list_departments tool with query: '{query}'")

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

    print(f"✅ Found {len(result)} departments matching '{query}'")

    # 검색 결과가 없을 때 예외처리
    if not result:
        print(f"⚠️  WARNING: No departments found matching '{query}'")
        return [{
            "error": "no_results",
            "message": "사용자 질문에 대한 정보를 가져올 수 없었습니다.",
            "suggestion": "get_search_help 툴을 사용하여 검색 가능한 방법을 안내하세요."
        }]

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
    print(f"✅ Using recommend_curriculum tool: {university} {department}, interests='{interests}'")

    vs = load_vectorstore()
    embeddings = get_embeddings()

    # 관심사 임베딩 생성 (있는 경우)
    interests_embedding = None
    if interests:
        interests_embedding = embeddings.embed_query(interests)

    curriculum = []
    selected_course_names = set()  # 중복 과목 방지용

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
            print(f"   [{semester_label}] Searching with filter: {filter_dict}")

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

                # 이미 선택된 과목 제외
                available_docs = [
                    doc for doc in docs
                    if doc.metadata.get("name", "") not in selected_course_names
                ]

                if not available_docs:
                    print(f"   ⚠️  [{semester_label}] 모든 과목이 이미 선택됨")
                    curriculum.append({
                        "semester": semester_label,
                        "course": None,
                        "reason": "해당 학기의 과목이 이미 다른 학기에 선택되었습니다."
                    })
                    continue

                # 관심사가 있으면 유사도 기반으로 정렬
                if interests_embedding and interests:
                    # 성능 개선: Vector Store의 유사도 검색 활용
                    # 이미 검색된 docs는 유사도 순으로 정렬되어 있음
                    # 관심사로 한 번 더 검색하는 대신, 검색 결과 순서 활용
                    print(f"   [Optimization] Using vector store similarity scores (skipping re-embedding)")

                    # available_docs는 이미 similarity_search로 정렬된 상태
                    # 관심사와 가장 유사한 과목이 앞에 있을 가능성이 높음
                    best_doc = available_docs[0]
                    reason = f"'{interests}' 관심사 관련 과목 (검색 결과 기준)"
                else:
                    # 관심사 없으면 첫 번째 과목 선택
                    best_doc = available_docs[0]
                    reason = "해당 학기 대표 과목"

                meta = best_doc.metadata
                course_name = meta.get("name", "[이름 없음]")

                # 선택된 과목 추가
                selected_course_names.add(course_name)

                # 실제 메타데이터 로깅 (디버깅용)
                actual_univ = meta.get("university", "[정보 없음]")
                actual_dept = meta.get("department", "[정보 없음]")
                actual_grade_sem = meta.get("grade_semester", "[정보 없음]")
                print(f"   ✅ [{semester_label}] Selected: {course_name}")
                print(f"      Source: {actual_univ} / {actual_dept} / {actual_grade_sem}")

                curriculum.append({
                    "semester": semester_label,
                    "course": {
                        "name": course_name,
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

    # 커리큘럼 전체가 비어있거나 모든 항목이 오류인 경우 예외처리
    valid_items = [item for item in curriculum if item.get("course") is not None]
    if not valid_items:
        print(f"⚠️  WARNING: No valid curriculum generated for {university} {department}")
        return [{
            "error": "no_results",
            "message": "사용자 질문에 대한 정보를 가져올 수 없었습니다.",
            "suggestion": "get_search_help 툴을 사용하여 검색 가능한 방법을 안내하세요.",
            "details": f"대학: {university}, 학과: {department}에 대한 커리큘럼을 찾을 수 없습니다."
        }]

    print(f"✅ Generated curriculum with {len(curriculum)} semesters ({len(valid_items)} valid)")

    return curriculum




@tool
def match_department_name(department_query: str) -> dict:
    """
    학과명을 임베딩 기반으로 표준 학과명으로 매핑한다.
    ex) '컴공' → '컴퓨터공학과'
        '컴퓨터과' → '컴퓨터공학과'
        '소프트웨어' → '소프트웨어학과'
    """

    vs = load_vectorstore()
    embeddings = get_embeddings()

    # Load all department names from metadata
    collection = vs._collection
    results = collection.get(include=["metadatas"])

    # unique department list
    departments = list({meta["department"] for meta in results["metadatas"] if meta.get("department")})

    # embed user query
    query_vec = embeddings.embed_query(department_query)

    best_match = None
    best_score = -999

    for dept in departments:
        dept_vec = embeddings.embed_query(dept)
        sim = np.dot(query_vec, dept_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(dept_vec))
        if sim > best_score:
            best_score = sim
            best_match = dept

    return {
        "input": department_query,
        "matched_department": best_match,
        "similarity": best_score
    }
  
@tool
def get_search_help() -> str:
    """
    사용자 질문에 대한 정보를 가져올 수 없었을 때 사용하는 툴입니다.
    검색 가능한 방법들(각 툴을 호출할 수 있는 방법들)을 안내합니다.

    ** 언제 사용하나요? **
    1. 다른 툴(retrieve_courses, list_departments, recommend_curriculum)의 결과가 비어있을 때
    2. 사용자의 질문이 너무 모호하거나 데이터베이스에 없는 정보를 요청할 때
    3. 검색 결과가 없어서 사용자에게 다른 검색 방법을 안내해야 할 때

    Returns:
        검색 가능한 방법들을 설명하는 가이드 메시지
    """
    print("ℹ️  Using get_search_help tool - providing usage guide to user")
    return _get_tool_usage_guide()
