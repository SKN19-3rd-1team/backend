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

# 학과 임베딩 캐싱 함수
_DEPT_EMBEDDINGS_CACHE = None
_DEPT_NAMES_CACHE = None

def _load_department_embeddings():
    global _DEPT_EMBEDDINGS_CACHE, _DEPT_NAMES_CACHE
    if _DEPT_EMBEDDINGS_CACHE is not None:
        return _DEPT_NAMES_CACHE, _DEPT_EMBEDDINGS_CACHE

    vs = load_vectorstore()
    embeddings = get_embeddings()

    collection = vs._collection
    results = collection.get(include=["metadatas"])

    departments = sorted({meta["department"]
                          for meta in results["metadatas"]
                          if meta.get("department")})

    # 🔹 한 번에 배치 임베딩 (OpenAI는 내부에서 알아서 배치 처리)
    dept_vecs = embeddings.embed_documents(departments)

    _DEPT_NAMES_CACHE = departments
    _DEPT_EMBEDDINGS_CACHE = np.array(dept_vecs)
    return _DEPT_NAMES_CACHE, _DEPT_EMBEDDINGS_CACHE

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
def list_departments(query: str) -> List[str]:
    """
    Vector DB에 있는 학과 목록을 조회합니다. (학과명만 반환, 대학명 제외)

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
        학과명 리스트 (중복 제거됨): ["컴퓨터공학과", "소프트웨어학부", "전자공학과", ...]
    """
    print(f"✅ Using list_departments tool with query: '{query}'")

    vs = load_vectorstore()
    collection = vs._collection

    # 모든 메타데이터 가져오기
    results = collection.get(include=['metadatas'])

    # 학과명만 추출 (중복 제거)
    departments_set = set()
    all_departments_with_info = []  # 필터링을 위해 전체 정보 보관

    for meta in results['metadatas']:
        university = meta.get('university', '')
        college = meta.get('college', '')
        department = meta.get('department', '')

        if department:
            departments_set.add(department)
            all_departments_with_info.append({
                "university": university,
                "college": college,
                "department": department
            })

    # 쿼리 필터링
    if query.strip() == "전체" or not query.strip():
        # 전체 학과명 반환
        result = sorted(list(departments_set))
    else:
        # 키워드로 필터링 (대학, 단과대학, 학과명에서 검색)
        query_lower = query.lower()
        matching_departments = set()

        for dept_info in all_departments_with_info:
            if (query_lower in dept_info['university'].lower() or
                query_lower in dept_info['college'].lower() or
                query_lower in dept_info['department'].lower()):
                matching_departments.add(dept_info['department'])

        result = sorted(list(matching_departments))

    print(f"✅ Found {len(result)} unique departments matching '{query}'")

    # 검색 결과가 없을 때 예외처리
    if not result:
        print(f"⚠️  WARNING: No departments found matching '{query}'")
        return ["검색 결과가 없습니다. 다른 키워드로 검색해보세요."]

    return result


@tool
def get_universities_by_department(department_name: str) -> List[Dict[str, str]]:
    """
    특정 학과가 있는 대학 목록을 조회합니다.

    ** 사용 시나리오 **
    - 학생이 특정 학과를 선택한 후, 해당 학과가 있는 대학들을 보여줄 때 사용
    - 예: "컴퓨터공학과"를 선택하면 → 서울대, 연세대, 고려대 등 목록 제공

    Args:
        department_name: 학과명 (예: "컴퓨터공학과", "소프트웨어학부")

    Returns:
        대학 정보 리스트 [
            {"university": "서울대학교", "college": "공과대학", "department": "컴퓨터공학과"},
            {"university": "연세대학교", "college": "공과대학", "department": "컴퓨터공학과"},
            ...
        ]
    """
    print(f"✅ Using get_universities_by_department tool for: '{department_name}'")

    vs = load_vectorstore()
    collection = vs._collection

    # 모든 메타데이터 가져오기
    results = collection.get(include=['metadatas'])

    # 해당 학과가 있는 대학 찾기
    universities_set = set()
    for meta in results['metadatas']:
        university = meta.get('university', '')
        college = meta.get('college', '')
        department = meta.get('department', '')

        # 정확한 매칭 또는 부분 매칭
        if department and (department == department_name or department_name in department):
            universities_set.add((university, college, department))

    # 리스트로 변환
    result = [
        {
            "university": univ,
            "college": college,
            "department": dept
        }
        for univ, college, dept in sorted(universities_set)
    ]

    print(f"✅ Found {len(result)} universities offering '{department_name}'")

    if not result:
        print(f"⚠️  WARNING: No universities found offering '{department_name}'")
        return [{
            "error": "no_results",
            "message": f"'{department_name}' 학과를 개설한 대학을 찾을 수 없습니다.",
            "suggestion": "학과명을 정확히 확인하거나 list_departments로 사용 가능한 학과 목록을 먼저 조회하세요."
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
    end_semester: int = 2,
    courses_per_semester: int = 5
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
        courses_per_semester: 학기당 추천 과목 수 (기본값: 5)

    Returns:
        학기별 추천 과목 리스트 [
            {
                "semester": "2학년 1학기",
                "courses": [
                    {"name": "...", "description": "...", "classification": "..."},
                    {"name": "...", "description": "...", "classification": "..."},
                    ...
                ],
                "count": 5
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
                # 해당 학기 과목 검색 (더 많은 후보 가져오기)
                docs = retrieve_with_filter(
                    question=interests if interests else "추천 과목",
                    search_k=20,  # 학기당 5개 선택하므로 더 많은 후보 필요
                    metadata_filter=chroma_filter
                )

                if not docs:
                    curriculum.append({
                        "semester": semester_label,
                        "courses": [],
                        "count": 0,
                        "message": "해당 학기에 개설된 과목이 없습니다."
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
                        "courses": [],
                        "count": 0,
                        "message": "해당 학기의 과목이 이미 다른 학기에 선택되었습니다."
                    })
                    continue

                # 학기당 최대 courses_per_semester개 과목 선택
                selected_courses = []
                for i, doc in enumerate(available_docs[:courses_per_semester]):
                    meta = doc.metadata
                    course_name = meta.get("name", "[이름 없음]")

                    # 중복 체크
                    if course_name in selected_course_names:
                        continue

                    # 선택된 과목 추가
                    selected_course_names.add(course_name)

                    # 실제 메타데이터 로깅 (디버깅용)
                    actual_univ = meta.get("university", "[정보 없음]")
                    actual_dept = meta.get("department", "[정보 없음]")
                    actual_grade_sem = meta.get("grade_semester", "[정보 없음]")
                    print(f"   ✅ [{semester_label}] Selected ({i+1}/{courses_per_semester}): {course_name}")
                    print(f"      Source: {actual_univ} / {actual_dept} / {actual_grade_sem}")

                    selected_courses.append({
                        "name": course_name,
                        "classification": meta.get("course_classification", "[정보 없음]"),
                        "description": doc.page_content
                    })

                    # 원하는 개수만큼 선택했으면 중단
                    if len(selected_courses) >= courses_per_semester:
                        break

                curriculum.append({
                    "semester": semester_label,
                    "courses": selected_courses,
                    "count": len(selected_courses)
                })

            except Exception as e:
                print(f"Error retrieving courses for {semester_label}: {e}")
                curriculum.append({
                    "semester": semester_label,
                    "courses": [],
                    "count": 0,
                    "message": f"검색 중 오류 발생: {str(e)}"
                })

    # 커리큘럼 전체가 비어있거나 모든 항목이 오류인 경우 예외처리
    valid_items = [item for item in curriculum if item.get("count", 0) > 0]
    if not valid_items:
        print(f"⚠️  WARNING: No valid curriculum generated for {university} {department}")
        return [{
            "error": "no_results",
            "message": "사용자 질문에 대한 정보를 가져올 수 없었습니다.",
            "suggestion": "get_search_help 툴을 사용하여 검색 가능한 방법을 안내하세요.",
            "details": f"대학: {university}, 학과: {department}에 대한 커리큘럼을 찾을 수 없습니다."
        }]

    total_courses = sum(item.get("count", 0) for item in curriculum)
    print(f"✅ Generated curriculum with {len(curriculum)} semesters ({total_courses} total courses)")

    return curriculum




@tool
def match_department_name(department_query: str) -> dict:
    """
    학과명을 임베딩 기반으로 표준 학과명으로 매핑합니다.

    대학명과 학과명이 섞여 있는 경우 자동으로 분리하여 처리합니다.
    대학명 정규화는 univ_mapping.json을 사용합니다.

    Examples:
        '컴공' → '컴퓨터공학과'
        '컴퓨터과' → '컴퓨터공학과'
        '소프트웨어' → '소프트웨어학부'
        '홍대 컴공' → university='홍익대학교', department='컴퓨터공학과'
        '서울대 전전' → university='서울대학교', department='전자공학과'
        '설대 컴공' → university='서울대학교', department='컴퓨터공학과' (은어 지원)

    Args:
        department_query: 학과명 또는 "대학명 + 학과명" 형태 (예: "컴공", "홍대 컴공")

    Returns:
        {
            "input": "원본 쿼리",
            "university": "추출된 대학명 (있는 경우)",
            "matched_department": "매칭된 표준 학과명",
            "similarity": "유사도 점수"
        }
    """
    from backend.rag.entity_extractor import normalize_university_name
    import re

    print(f"✅ Using match_department_name with query: '{department_query}'")

    # 대학명 추출 시도
    extracted_university = None
    dept_only_query = department_query

    # 1단계: 공백으로 분리하여 대학명 체크
    tokens = department_query.split()
    if len(tokens) >= 2:
        first_token = tokens[0]

        # entity_extractor의 normalize_university_name 사용
        # 정규화 시도 (홍대 → 홍익대학교, 설대 → 서울대학교 등)
        normalized = normalize_university_name(first_token)

        # 정규화가 성공했는지 확인 (원본과 다르면 성공)
        if normalized != first_token or normalized.endswith('대학교'):
            extracted_university = normalized
            # "대학교"로 끝나지 않으면 추가
            if not extracted_university.endswith('대학교'):
                extracted_university += '대학교'

            dept_only_query = ' '.join(tokens[1:])  # 나머지를 학과명으로
            print(f"   Extracted university: {extracted_university} (from '{first_token}')")
            print(f"   Department query: {dept_only_query}")

    # 2단계: 공백 없이 붙어있는 경우 처리 (예: "홍대컴공")
    # 정규식으로 대학명 패턴 찾기
    if not extracted_university:
        # "~대학교", "~대" 패턴 찾기
        univ_pattern = r'^([가-힣]+대학교|[가-힣]+대)'
        univ_match = re.match(univ_pattern, department_query)

        if univ_match:
            univ_token = univ_match.group(1)
            normalized = normalize_university_name(univ_token)

            if normalized != univ_token or normalized.endswith('대학교'):
                extracted_university = normalized
                if not extracted_university.endswith('대학교'):
                    extracted_university += '대학교'

                # 대학명 부분을 제거한 나머지를 학과명으로
                dept_only_query = department_query[len(univ_match.group(0)):].strip()
                print(f"   Extracted university: {extracted_university} (from '{univ_token}')")
                print(f"   Department query: {dept_only_query}")

    embeddings = get_embeddings()

    # 1) 캐시된 학과명 + 임베딩 불러오기
    departments, dept_matrix = _load_department_embeddings()

    # 2) 학과명만 임베딩하여 매칭
    query_vec = np.array(embeddings.embed_query(dept_only_query))

    # 3) 전체 학과와의 코사인 유사도 계산
    norms = np.linalg.norm(dept_matrix, axis=1) * np.linalg.norm(query_vec)
    # 0으로 나누는 것 방지
    norms = np.where(norms == 0, 1e-10, norms)
    sims = (dept_matrix @ query_vec) / norms

    best_idx = int(np.argmax(sims))
    best_match = departments[best_idx]
    best_score = float(sims[best_idx])

    print(f"   Best match: {best_match} (similarity: {best_score:.3f})")

    result = {
        "input": department_query,
        "matched_department": best_match,
        "similarity": best_score,
    }

    if extracted_university:
        result["university"] = extracted_university

    return result
  
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
