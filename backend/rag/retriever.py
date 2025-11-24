"""
벡터 DB 검색 (Retrieval) 모듈

Chroma DB에서 사용자 질문과 유사한 과목을 검색하는 기능을 제공합니다.

** 주요 기능 **
1. 메타데이터 필터링: 대학, 학과, 학년 등으로 검색 범위 제한
2. Fuzzy 매칭: "컴퓨터공학" 검색 시 "컴퓨터공학부", "컴퓨터공학과"도 검색
3. 자동 폴백: 필터가 너무 제한적일 때 단계적으로 완화

** 검색 과정 **
1. 질문을 벡터로 변환 (임베딩 모델 사용)
2. 벡터 DB에서 유사한 벡터 검색 (코사인 유사도)
3. 메타데이터 필터 적용 (선택적)
4. 결과가 없으면 필터 완화 후 재시도
"""
# backend/rag/retriever.py
from typing import Dict, Optional, List
from langchain_core.documents import Document
from .vectorstore import load_vectorstore
from .entity_extractor import normalize_department_name


def get_retriever(search_k: int = 5, metadata_filter: Optional[Dict] = None):
    """
    메타데이터 필터를 적용한 검색기(Retriever) 인스턴스 반환

    LangChain의 Retriever 인터페이스를 반환하여 LangChain 파이프라인에 통합할 수 있습니다.

    Args:
        search_k: 검색할 문서 개수 (기본값: 5개)
        metadata_filter: Chroma DB 메타데이터 필터
                        예: {"university": {"$eq": "서울대학교"}}
                            {"$and": [{"university": ...}, {"department": ...}]}

    Returns:
        LangChain Retriever: 설정된 검색기 인스턴스
    """
    # 디스크에서 Chroma DB 로드
    vs = load_vectorstore()

    # 검색 파라미터 설정
    search_kwargs = {"k": search_k}  # 반환할 문서 개수
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter  # 메타데이터 필터 추가

    # VectorStore를 Retriever로 변환하여 반환
    return vs.as_retriever(search_kwargs=search_kwargs)


def _relax_filter(metadata_filter: Optional[Dict], relax_field: str) -> Optional[Dict]:
    """
    필터에서 특정 필드를 제거하여 검색 조건 완화

    검색 결과가 없을 때 필터를 단계적으로 완화하기 위해 사용됩니다.
    예: "서울대 + 컴공" 필터 → "서울대만" 필터

    Args:
        metadata_filter: 원본 메타데이터 필터
        relax_field: 제거할 필드명 (예: "department", "college")

    Returns:
        Optional[Dict]: 완화된 필터 또는 None (조건이 모두 제거된 경우)

    ** 예시 **
    - 입력: {"department": {"$eq": "컴공"}}, "department"
      출력: None (단일 조건이므로 제거 시 필터 없음)

    - 입력: {"$and": [{"university": ...}, {"department": ...}]}, "department"
      출력: {"university": ...} (department 조건만 제거)
    """
    if not metadata_filter:
        return None

    # 단일 조건 처리: 해당 필드만 있으면 None 반환 (필터 전체 제거)
    if relax_field in metadata_filter:
        return None

    # $and 조건 처리: 여러 조건 중 하나만 제거
    if "$and" in metadata_filter:
        # relax_field를 포함하지 않는 조건만 남김
        remaining_conditions = [
            cond for cond in metadata_filter["$and"]
            if relax_field not in cond
        ]

        # 남은 조건이 없으면 None
        if len(remaining_conditions) == 0:
            return None
        # 남은 조건이 1개면 $and 없이 직접 반환
        elif len(remaining_conditions) == 1:
            return remaining_conditions[0]
        # 남은 조건이 2개 이상이면 $and로 묶어서 반환
        else:
            return {"$and": remaining_conditions}

    # 다른 형태의 필터는 그대로 반환 (변경 없음)
    return metadata_filter


def _build_fuzzy_department_filter(
    base_filter: Optional[Dict],
    department_base: str
) -> Optional[Dict]:
    """
    학과명의 다양한 변형을 모두 매칭하는 Fuzzy 필터 생성

    한국 대학의 학과명은 "부", "과" 등의 접미사가 붙거나 안 붙을 수 있습니다.
    이 함수는 접미사 유무와 관계없이 모두 검색되도록 필터를 생성합니다.

    Args:
        base_filter: department 필드가 제외된 기본 필터
        department_base: 정규화된 학과명 (예: "컴퓨터공학")

    Returns:
        Optional[Dict]: Fuzzy 매칭을 위한 $in 연산자가 적용된 필터

    ** 예시 **
    - 입력: "컴퓨터공학"
      출력: {"department": {"$in": ["컴퓨터공학", "컴퓨터공학부", "컴퓨터공학과"]}}
      → 3가지 변형 모두 검색됨
    """
    # 학과명 변형 생성: 접미사 유무에 따른 3가지 패턴
    dept_variations = [
        department_base,           # 접미사 없음: "컴퓨터공학"
        department_base + "부",    # 학부: "컴퓨터공학부"
        department_base + "과"     # 학과: "컴퓨터공학과"
    ]

    # $in 연산자로 여러 변형 모두 매칭
    dept_filter = {"department": {"$in": dept_variations}}

    # 기본 필터가 없으면 department 필터만 반환
    if base_filter is None:
        return dept_filter

    # 기본 필터와 결합
    if "$and" in base_filter:
        # 기존 $and 조건에 department 필터 추가
        return {"$and": base_filter["$and"] + [dept_filter]}
    else:
        # 기본 필터를 $and로 묶어서 department 필터와 결합
        return {"$and": [base_filter, dept_filter]}


def retrieve_with_filter(
    question: str,
    search_k: int = 5,
    metadata_filter: Optional[Dict] = None,
    warn_on_fallback: bool = False
) -> List[Document]:
    """
    메타데이터 필터링과 자동 폴백을 지원하는 스마트 검색 함수

    검색 결과가 없을 때 필터를 단계적으로 완화하여 사용자에게 항상 유용한 결과를 제공합니다.

    ** 폴백 전략 (순서대로 시도, 결과를 찾을 때까지) **
    1단계: 정확한 필터 매칭 시도
    2단계: Fuzzy 학과명 매칭 ("컴공" → "컴공부", "컴공과" 모두 매칭)
    3단계: 학과 필터 제거 (대학, 학년만으로 검색)
    4단계: 단과대 필터 제거 (대학, 학년만으로 검색)
    5단계: 모든 필터 제거 (순수 의미 기반 검색)

    ** 사용 예시 **
    ```python
    # 필터 없이 검색
    docs = retrieve_with_filter("인공지능 관련 과목", search_k=5)

    # 대학 필터 적용
    filter = {"university": {"$eq": "서울대학교"}}
    docs = retrieve_with_filter("인공지능", search_k=5, metadata_filter=filter)

    # 여러 조건 결합
    filter = {"$and": [
        {"university": {"$eq": "서울대학교"}},
        {"department": {"$eq": "컴퓨터공학"}}
    ]}
    docs = retrieve_with_filter("머신러닝", search_k=5, metadata_filter=filter)
    ```

    Args:
        question: 사용자 질문 (예: "인공지능 관련 과목 추천해줘")
        search_k: 검색할 문서 개수 (기본값: 5개)
        metadata_filter: Chroma DB 메타데이터 필터 (선택적)
        warn_on_fallback: True시 폴백 발생 시 경고 메시지 출력

    Returns:
        List[Document]: 검색된 과목 Document 리스트
    """
    # Chroma DB 로드
    vs = load_vectorstore()

    # 필터가 없으면 순수 의미 기반 검색만 수행
    if not metadata_filter:
        return vs.similarity_search(query=question, k=search_k)

    # ==================== 1단계: 정확한 필터 매칭 시도 ====================
    # 사용자가 지정한 필터를 그대로 적용하여 검색
    try:
        results = vs.similarity_search(
            query=question,
            k=search_k,
            filter=metadata_filter  # 원본 필터 그대로 사용
        )
        if results:
            print(f"[Retriever] ✅ Found {len(results)} results with exact filter")
            return results
    except Exception as e:
        # 필터 형식 오류 등으로 실패할 수 있음
        print(f"[Retriever] ❌ Exact filter failed: {e}")

    # ==================== 2단계: Fuzzy 학과명 매칭 시도 ====================
    # "컴퓨터공학" → "컴퓨터공학부", "컴퓨터공학과" 변형도 함께 검색
    # 학과명 접미사 차이로 인한 검색 실패를 방지
    department_value = None

    # 필터에서 department 값 추출
    if "department" in metadata_filter:
        # 단일 조건: {"department": {"$eq": "컴퓨터공학"}}
        department_value = metadata_filter["department"].get("$eq")
    elif "$and" in metadata_filter:
        # 복합 조건: {"$and": [{...}, {"department": ...}]}
        for cond in metadata_filter["$and"]:
            if "department" in cond:
                department_value = cond["department"].get("$eq")
                break

    # department 값이 있으면 Fuzzy 매칭 시도
    if department_value:
        # 학과명 정규화 (예: "컴공과" → "컴퓨터공학")
        dept_base = normalize_department_name(department_value)

        # department 필드를 제거한 기본 필터 생성
        base_filter = _relax_filter(metadata_filter, "department")

        # Fuzzy 필터 생성 (부, 과 변형 모두 포함)
        fuzzy_filter = _build_fuzzy_department_filter(base_filter, dept_base)

        try:
            results = vs.similarity_search(
                query=question,
                k=search_k,
                filter=fuzzy_filter
            )
            if results:
                if warn_on_fallback:
                    print(f"⚠️  [Fallback] Exact filter failed, using fuzzy department matching")
                print(f"[Retriever] ✅ Found {len(results)} results with fuzzy department matching")
                return results
        except Exception as e:
            print(f"[Retriever] ❌ Fuzzy department matching failed: {e}")

    # ==================== 3단계: 학과 필터 제거 ====================
    # "서울대 + 컴공" 검색 실패 → "서울대"만으로 검색
    # 다른 학과의 관련 과목도 찾을 수 있도록 완화
    relaxed_filter = _relax_filter(metadata_filter, "department")
    if relaxed_filter:
        try:
            results = vs.similarity_search(
                query=question,
                k=search_k,
                filter=relaxed_filter
            )
            if results:
                if warn_on_fallback:
                    print(f"⚠️  [Fallback] Department filter removed - searching without department constraint")
                print(f"[Retriever] ✅ Found {len(results)} results without department filter")
                return results
        except Exception as e:
            print(f"[Retriever] ❌ Relaxed filter (no department) failed: {e}")

    # ==================== 4단계: 단과대 필터 제거 ====================
    # "서울대 + 공대" 검색 실패 → "서울대"만으로 검색
    # 다른 단과대의 관련 과목도 찾을 수 있도록 추가 완화
    relaxed_filter2 = _relax_filter(relaxed_filter, "college")
    if relaxed_filter2:
        try:
            results = vs.similarity_search(
                query=question,
                k=search_k,
                filter=relaxed_filter2
            )
            if results:
                if warn_on_fallback:
                    print(f"⚠️  [Fallback] College filter also removed - searching with minimal constraints")
                print(f"[Retriever] ✅ Found {len(results)} results without college filter")
                return results
        except Exception as e:
            print(f"[Retriever] ❌ Relaxed filter (no college) failed: {e}")

    # ==================== 5단계: 최종 폴백 - 순수 의미 기반 검색 ====================
    # 모든 필터를 제거하고 질문의 의미만으로 검색
    # 다른 대학/학과의 과목이 반환될 수 있지만, 빈 결과보다는 유용함
    if warn_on_fallback:
        print(f"🚨 [CRITICAL FALLBACK] All filters failed! Using pure semantic search.")
        print(f"   This may return courses from different universities/departments!")
    print("[Retriever] ⚠️  Falling back to pure semantic search (no filters)")
    return vs.similarity_search(query=question, k=search_k)
