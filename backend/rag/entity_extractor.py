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


def normalize_department_name(department_query: str) -> str:
    """
    학과명 정규화: 접미사 제거

    한국 대학의 학과명은 "과", "부" 접미사가 붙거나 안 붙을 수 있습니다.
    DB에는 두 형태가 모두 존재할 수 있으므로 Fuzzy 매칭을 위해 접미사를 제거합니다.

    ** 예시 **
    - "컴퓨터공학과" → "컴퓨터공학"
    - "컴퓨터공학부" → "컴퓨터공학"
    - "컴퓨터공학" → "컴퓨터공학" (이미 정규화됨)

    이렇게 정규화된 이름은 retriever.py의 Fuzzy 필터에서
    ["컴퓨터공학", "컴퓨터공학부", "컴퓨터공학과"] 3가지 변형으로 확장되어 검색됩니다.

    Args:
        department_query: 질문에서 추출한 학과명 (예: "컴퓨터공학", "컴퓨터공학과")

    Returns:
        str: 정규화된 학과 기본 이름 (예: "컴퓨터공학")
    """
    # 이미 정규화된 경우 (접미사 없음)
    if not (department_query.endswith('과') or department_query.endswith('부')):
        return department_query

    # 접미사 제거: "컴퓨터공학과" → "컴퓨터공학"
    return department_query[:-1]


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
            # JSON에 들어있는 값(예: "건설환경공학", "컴퓨터공학")과 일치
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
