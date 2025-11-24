# backend/rag/entity_extractor.py
"""
엔티티 추출 (Entity Extraction) 모듈

사용자 질문에서 대학, 학과, 학년, 학기 등의 엔티티를 자동으로 추출합니다.
추출된 엔티티는 메타데이터 필터로 변환되어 검색 범위를 제한하는 데 사용됩니다.

** 주요 기능 **
1. extract_filters(): 질문에서 대학, 학과, 학년 등 추출
2. normalize_university_name(): 대학 별칭 정규화 ("홍대" → "홍익대학교")
3. normalize_department_name(): 학과명 정규화 ("컴공과" → "컴퓨터공학")
4. build_chroma_filter(): 추출된 엔티티를 Chroma DB 필터로 변환

** 예시 **
질문: "홍익대 컴공과 1학년 필수 과목"
추출: {university: "홍익대학교", department: "컴퓨터공학", grade: "1학년"}
필터: {"$and": [{"university": {"$eq": "홍익대학교"}}, ...]}
"""

import re
import json
from pathlib import Path
from typing import Dict, Optional, List


# 대학명 매핑 데이터 싱글톤 캐시
# univ_mapping.json 파일을 한 번만 로드하여 메모리에 캐싱
_UNIVERSITY_MAPPING = None

# 학과명 매핑 데이터 싱글톤 캐시
# department_mapping.json 파일을 한 번만 로드하여 메모리에 캐싱
_DEPARTMENT_MAPPING = None

def _load_university_mapping() -> List[Dict]:
    """
    대학 별칭 매핑 JSON 파일 로드 (싱글톤 패턴)

    univ_mapping.json 파일에서 대학별 공식 이름, 별칭, 은어를 읽어옵니다.
    예: "홍대", "홍익대" → "홍익대학교"
        "설대", "샤대" → "서울대학교"

    Returns:
        List[Dict]: 대학 매핑 정보 리스트
    """
    global _UNIVERSITY_MAPPING
    if _UNIVERSITY_MAPPING is None:
        # 프로젝트 루트의 univ_mapping.json 파일 경로
        mapping_path = Path(__file__).parent.parent.parent / "univ_mapping.json"
        if mapping_path.exists():
            _UNIVERSITY_MAPPING = json.loads(mapping_path.read_text(encoding="utf-8"))
        else:
            # 파일이 없으면 빈 리스트 반환 (폴백)
            _UNIVERSITY_MAPPING = []
    return _UNIVERSITY_MAPPING


def _load_department_mapping() -> Dict[str, Dict]:
    """
    학과명 매핑 JSON 파일 로드 (싱글톤 패턴)

    department_mapping.json 파일에서 학과 계열별 별칭과 키워드를 읽어옵니다.
    예: "소프트웨어학부", "컴퓨터공학과", "컴공" → "컴퓨터·소프트웨어·인공지능 계열"

    Returns:
        Dict[str, Dict]: 학과 매핑 정보 딕셔너리
                        키: 계열 ID (예: "cs_software_ai")
                        값: {canonical_kor, alias_departments, alias_keywords}
    """
    global _DEPARTMENT_MAPPING
    if _DEPARTMENT_MAPPING is None:
        # 프로젝트 루트의 department_mapping.json 파일 경로
        mapping_path = Path(__file__).parent.parent.parent / "department_mapping.json"
        if mapping_path.exists():
            _DEPARTMENT_MAPPING = json.loads(mapping_path.read_text(encoding="utf-8"))
        else:
            # 파일이 없으면 빈 딕셔너리 반환 (폴백)
            _DEPARTMENT_MAPPING = {}
    return _DEPARTMENT_MAPPING


def normalize_university_name(university_query: str) -> str:
    """
    대학 별칭을 공식 이름으로 정규화

    사용자가 사용한 대학 별칭이나 은어를 univ_mapping.json을 참고하여 공식 이름으로 변환합니다.
    이를 통해 다양한 표현으로 검색해도 정확한 대학을 찾을 수 있습니다.

    ** 변환 예시 **
    - "홍대" → "홍익대학교"
    - "서울대" → "서울대학교"
    - "설대" (은어) → "서울대학교"
    - "카이스트" → "KAIST"

    Args:
        university_query: 질문에서 추출한 대학명 (예: "홍대", "서울대")

    Returns:
        str: 정규화된 공식 대학명 (예: "홍익대학교", "서울대학교")
              매핑이 없으면 원본 그대로 반환
    """
    # univ_mapping.json 로드
    mapping = _load_university_mapping()

    for univ in mapping:
        # 1순위: 별칭 확인 (예: "홍익대", "홍대")
        if university_query in univ.get("aliases_ko", []):
            return univ["official_name_ko"]

        # 2순위: 은어 확인 (예: "설대", "샤대")
        if university_query in univ.get("slang_ko", []):
            return univ["official_name_ko"]

        # 3순위: 공식 이름 확인 (이미 정확하게 입력된 경우)
        if university_query == univ["official_name_ko"]:
            return univ["official_name_ko"]

    # 매핑에 없으면 원본 그대로 반환 (폴백)
    return university_query


def get_all_department_variants(department_query: str) -> List[str]:
    """
    학과명이 속한 계열의 모든 학과명 변형 반환 (검색 범위 확대용)

    하나의 학과명을 입력받아, 같은 계열에 속하는 모든 학과명을 반환합니다.
    이를 통해 대학마다 학과명이 다르더라도 모두 검색할 수 있습니다.

    ** 사용 목적 **
    - "컴퓨터공학"으로 검색해도 "소프트웨어학부", "인공지능학과" 모두 찾기
    - 한양대처럼 "소프트웨어학부"로 저장된 경우에도 "컴퓨터공학" 검색으로 찾기

    ** 예시 **
    - 입력: "컴퓨터공학과"
      출력: ["컴퓨터공학과", "컴퓨터공학부", "소프트웨어학과", "소프트웨어학부",
             "인공지능학과", "인공지능학부", "데이터사이언스학과", ...]

    - 입력: "전기공학과"
      출력: ["전기공학과", "전기공학부", "전자공학과", "전자공학부",
             "전기전자공학과", ...]

    Args:
        department_query: 사용자가 입력한 학과명 (예: "컴퓨터공학", "소프트웨어학부")

    Returns:
        List[str]: 같은 계열에 속하는 모든 학과명 리스트
                   매핑되지 않으면 기본 변형(접미사 유무)만 반환
    """
    # department_mapping.json 로드
    mapping = _load_department_mapping()

    # ==================== 1단계: 정확한 학과명 매칭으로 계열 찾기 ====================
    for category_id, category_data in mapping.items():
        alias_depts = category_data.get("alias_departments", [])
        if department_query in alias_depts:
            # 같은 계열의 모든 학과명 반환
            print(f"[Department Variants] '{department_query}' → {len(alias_depts)} variants from {category_data['canonical_kor']}")
            return alias_depts

    # ==================== 2단계: 키워드 매칭으로 계열 찾기 ====================
    for category_id, category_data in mapping.items():
        alias_keywords = category_data.get("alias_keywords", [])
        for keyword in alias_keywords:
            if keyword in department_query or department_query in keyword:
                alias_depts = category_data.get("alias_departments", [])
                print(f"[Department Variants] '{department_query}' → {len(alias_depts)} variants (keyword: '{keyword}')")
                return alias_depts

    # ==================== 3단계: 폴백 - 기본 변형만 반환 ====================
    # 매핑되지 않으면 접미사 유무만 처리
    base_name = department_query[:-1] if department_query.endswith(('과', '부')) else department_query
    variants = [base_name, base_name + "과", base_name + "부"]
    print(f"[Department Variants] '{department_query}' → {len(variants)} basic variants (no mapping)")
    return variants


def normalize_department_name(department_query: str) -> Optional[str]:
    """
    학과명 정규화: 매핑 기반 정규화 + 접미사 제거

    department_mapping.json을 사용하여 다양한 학과 별칭을 대표 학과명으로 정규화합니다.
    이를 통해 "소프트웨어학부", "컴공", "인공지능학과" 등을 모두 같은 계열로 인식할 수 있습니다.

    ** 중요: 여러 학과가 섞이지 않도록 보장 **
    - 하나의 입력은 하나의 대표 학과명으로만 매핑됩니다
    - 매핑되지 않으면 기존 방식대로 접미사만 제거

    ** 정규화 전략 **
    1. department_mapping.json의 alias_departments에서 정확히 일치하는 학과명 검색
    2. 일치하면 해당 계열의 **가장 대표적인 학과명** 반환 (alias_departments[0])
    3. 일치하지 않으면 alias_keywords에서 부분 매칭 시도
    4. 그래도 없으면 기존 방식(접미사 제거)으로 폴백

    ** 예시 **
    - "소프트웨어학과" → "컴퓨터공학과" (cs_software_ai 계열의 대표 학과)
    - "컴공" → "컴퓨터공학과" (키워드 매칭)
    - "인공지능학부" → "컴퓨터공학과" (같은 계열로 통합)
    - "전기공학과" → "전기공학과" (ee_electronics_ict 계열의 대표 학과)
    - "알수없는학과" → "알수없는학" (매핑 없으면 접미사 제거)

    Args:
        department_query: 질문에서 추출한 학과명 (예: "소프트웨어학부", "컴공")

    Returns:
        Optional[str]: 정규화된 대표 학과명 (예: "컴퓨터공학과")
                      매핑되지 않으면 접미사를 제거한 이름 반환
    """
    # department_mapping.json 로드
    mapping = _load_department_mapping()

    # ==================== 1단계: 정확한 학과명 매칭 ====================
    # alias_departments에서 정확히 일치하는 학과명 검색
    for category_id, category_data in mapping.items():
        alias_depts = category_data.get("alias_departments", [])
        if department_query in alias_depts:
            # 해당 계열의 **첫 번째 학과명**을 대표 학과명으로 반환
            # (첫 번째 학과가 가장 대표적인 학과로 간주)
            representative_dept = alias_depts[0]
            print(f"[Department Normalization] '{department_query}' → '{representative_dept}' ({category_data['canonical_kor']})")
            # 접미사 제거하여 반환
            return representative_dept[:-1] if representative_dept.endswith(('과', '부')) else representative_dept

    # ==================== 2단계: 키워드 부분 매칭 ====================
    # "컴공", "소웨" 같은 축약어를 처리
    for category_id, category_data in mapping.items():
        alias_keywords = category_data.get("alias_keywords", [])
        for keyword in alias_keywords:
            if keyword in department_query or department_query in keyword:
                alias_depts = category_data.get("alias_departments", [])
                if alias_depts:
                    representative_dept = alias_depts[0]
                    print(f"[Department Normalization] '{department_query}' → '{representative_dept}' (keyword match: '{keyword}')")
                    # 접미사 제거하여 반환
                    return representative_dept[:-1] if representative_dept.endswith(('과', '부')) else representative_dept

    # ==================== 3단계: 폴백 - 기존 방식 (접미사 제거) ====================
    # 매핑되지 않으면 기존 방식대로 접미사만 제거
    if department_query.endswith('과') or department_query.endswith('부'):
        normalized = department_query[:-1]
        print(f"[Department Normalization] '{department_query}' → '{normalized}' (suffix removal, no mapping)")
        return normalized

    # 접미사도 없고 매핑도 없으면 원본 그대로 반환
    print(f"[Department Normalization] '{department_query}' → '{department_query}' (no change)")
    return department_query


def extract_filters(question: str) -> Dict[str, str]:
    """
    사용자 질문에서 대학, 학과, 학년, 학기 정보 자동 추출

    정규 표현식(regex)를 사용하여 질문 텍스트를 파싱하고 엔티티를 추출합니다.
    추출 순서: 단과대학 → 대학교 → 학과 → 학년 → 학기

    ** 추출 예시 **
    - "홍익대학교 컴퓨터공학과 1학년 필수 과목"
      → {university: "홍익대학교", department: "컴퓨터공학", grade: "1학년"}

    - "서울대 공과대학 전자공학부 2학기"
      → {university: "서울대학교", college: "공과대학", department: "전자공학", semester: "2학기"}

    Args:
        question: 사용자의 질문 문자열

    Returns:
        Dict[str, str]: 추출된 필터 정보
                        키: university, college, department, grade, semester
    """
    filters = {}

    # ==================== 1단계: 단과대학 추출 ====================
    # 예: "공과대학", "자연과학대학" 등
    # 주의: 대학교 추출 **전에** 먼저 수행해야 "공과대학교" 같은 오탐지 방지
    college_pattern = r'([가-힣]+대학)(?!교)'  # "대학"으로 끝나지만 "대학교"는 제외
    college_match = re.search(college_pattern, question)
    if college_match:
        college_name = college_match.group(1)
        filters['college'] = college_name

    # ==================== 2단계: 대학교 추출 및 정규화 ====================
    # 예: "홍익대학교", "서울대학교", "홍대", "설대"(은어) 등
    # 단과대학과 겹치면 스킵 (공과대학을 대학으로 오인하지 않기 위해)
    university_pattern = r'([가-힣]+대학교|[가-힣]+대)'
    university_match = re.search(university_pattern, question)

    if university_match and not college_match:
        university_raw = university_match.group(1)
        # univ_mapping.json을 사용하여 별칭/은어를 공식 이름으로 변환
        university_name = normalize_university_name(university_raw)
        # "대학교"로 끝나지 않으면 추가
        if not university_name.endswith('대학교'):
            university_name += '학교'
        filters['university'] = university_name

    # ==================== 3단계: 학과 추출 전처리 ====================
    # 대학교/단과대학 이름을 질문에서 제거하여 학과 추출 시 혼동 방지
    # 예: "홍익대학교 컴퓨터공학과" → " 컴퓨터공학과" (홍익대학교 제거)
    question_for_dept = question
    if university_match and not college_match:
        question_for_dept = question.replace(university_match.group(0), '')
    if college_match:
        question_for_dept = question_for_dept.replace(college_match.group(0), '')

    # Extract department name (e.g., 컴퓨터공학과, 컴퓨터 공학과, 전자공학부, etc.)
    # 전체 학과명을 캡처한 후, "과"/"부"만 제거 (DB에 "컴퓨터공학"으로 저장되어 있음)
    # 우선순위: "과"/"부" 있는 경우 → "공학"으로 끝나는 경우 → "학과"/"학부"
    department_patterns = [
        r'([가-힣\s]+공학)과',      # 1순위: 컴퓨터공학과 → 컴퓨터공학
        r'([가-힣\s]+공학)부',      # 1순위: 컴퓨터공학부 → 컴퓨터공학
        r'([가-힣\s]+)학과',        # 2순위: 정보시스템학과 → 정보시스템
        r'([가-힣\s]+)학부',        # 2순위: 전자전기학부 → 전자전기
        r'([가-힣\s]+공학)(?![과부학])',  # 3순위: 컴퓨터공학 (뒤에 과/부/학이 없는 경우)
    ]

    for pattern in department_patterns:
        dept_match = re.search(pattern, question_for_dept)
        if dept_match:
            dept_raw = dept_match.group(1)
            # 띄어쓰기 제거
            dept_raw = dept_raw.strip().replace(' ', '')
            # 너무 짧거나 "에", "에게" 같은 조사만 매칭된 경우 스킵
            if len(dept_raw) < 2:
                continue
            # 너무 일반적인 단어는 제외 (오탐지 방지)
            if dept_raw in ['대학', '학교', '과목', '수업']:
                continue

            # ==================== 학과명 정규화 적용 ====================
            # department_mapping.json을 사용하여 정규화
            # 예: "소프트웨어학과" → "컴퓨터공학", "컴공" → "컴퓨터공학"
            normalized_dept = normalize_department_name(dept_raw)
            if normalized_dept:
                filters['department'] = normalized_dept
            else:
                filters['department'] = dept_raw
            break

    # Extract grade level (e.g., 1학년, 2학년, 3학년, 4학년)
    grade_pattern = r'([1-4])학년'
    grade_match = re.search(grade_pattern, question)
    if grade_match:
        grade_num = grade_match.group(1)
        filters['grade'] = f"{grade_num}학년"

    semester_pattern = r'([1-2])학기'
    semester_match = re.search(semester_pattern, question)
    if semester_match:
        semester_num = semester_match.group(1)
        filters['semester'] = f"{semester_num}학기"

    return filters


def build_chroma_filter(filters: Dict[str, str]) -> Optional[Dict]:
    """
    추출된 엔티티를 Chroma DB 메타데이터 필터로 변환

    extract_filters()가 추출한 딕셔너리를 Chroma DB가 이해할 수 있는
    쿼리 형식으로 변환합니다.

    ** Chroma 필터 형식 **
    - 단일 조건: {"university": {"$eq": "홍익대학교"}}
    - 복수 조건: {"$and": [{"university": ...}, {"department": ...}]}

    ** 변환 예시 **
    입력: {university: "홍익대학교", department: "컴퓨터공학", grade: "1학년"}
    출력: {"$and": [
            {"university": {"$eq": "홍익대학교"}},
            {"department": {"$eq": "컴퓨터공학"}},
            {"grade": {"$eq": "1학년"}}
          ]}

    Args:
        filters: extract_filters()가 반환한 엔티티 딕셔너리
                 키: university, college, department, grade, semester

    Returns:
        Optional[Dict]: Chroma DB 쿼리 필터 또는 None (필터가 없는 경우)
    """
    if not filters:
        return None

    # 각 필터 조건을 Chroma 쿼리 형식으로 변환
    conditions = []

    if 'university' in filters:
        # 대학 필터 (예: {"university": {"$eq": "홍익대학교"}})
        conditions.append({"university": {"$eq": filters['university']}})

    if 'college' in filters and filters['college']:
        # 단과대학 필터 (예: {"college": {"$eq": "공과대학"}})
        conditions.append({"college": {"$eq": filters['college']}})

    if 'department' in filters and filters['department']:
        # 학과 필터 (예: {"department": {"$eq": "컴퓨터공학"}})
        # entity_extractor가 이미 정규화했으므로 (예: "컴퓨터공학과" → "컴퓨터공학")
        # DB의 department 필드와 정확히 매칭 ($eq 사용)
        conditions.append({"department": {"$eq": filters['department']}})

    if 'grade' in filters:
        # 학년 필터 (예: {"grade": {"$eq": "2학년"}})
        conditions.append({"grade": {"$eq": filters['grade']}})

    if 'semester' in filters:
        # 학기 필터 (예: {"semester": {"$eq": "1학기"}})
        conditions.append({"semester": {"$eq": filters['semester']}})

    # 조건이 없으면 None 반환 (필터링 없음)
    if len(conditions) == 0:
        return None
    # 조건이 1개면 그대로 반환
    elif len(conditions) == 1:
        return conditions[0]
    # 조건이 2개 이상이면 $and로 결합
    else:
        return {"$and": conditions}
