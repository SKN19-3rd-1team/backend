# backend/rag/entity_extractor.py
"""Extract entities (university, department, grade) from user questions."""

import re
import json
from pathlib import Path
from typing import Dict, Optional, List


# 대학명 매핑 데이터 로드 (캐싱)
_UNIVERSITY_MAPPING = None

def _load_university_mapping() -> List[Dict]:
    """Load university mapping from JSON file (cached)."""
    global _UNIVERSITY_MAPPING
    if _UNIVERSITY_MAPPING is None:
        mapping_path = Path(__file__).parent.parent.parent / "univ_mapping.json"
        if mapping_path.exists():
            _UNIVERSITY_MAPPING = json.loads(mapping_path.read_text(encoding="utf-8"))
        else:
            _UNIVERSITY_MAPPING = []
    return _UNIVERSITY_MAPPING


def normalize_university_name(university_query: str) -> str:
    """
    Normalize university name using mapping file.

    Examples:
        "홍대" -> "홍익대학교"
        "서울대" -> "서울대학교"
        "설대" (slang) -> "서울대학교"

    Args:
        university_query: Raw university name from query

    Returns:
        Normalized official university name
    """
    mapping = _load_university_mapping()

    for univ in mapping:
        # Check aliases (홍익대, 홍대, etc.)
        if university_query in univ.get("aliases_ko", []):
            return univ["official_name_ko"]

        # Check slang (설대, 샤대, etc.)
        if university_query in univ.get("slang_ko", []):
            return univ["official_name_ko"]

        # Check official name
        if university_query == univ["official_name_ko"]:
            return univ["official_name_ko"]

    # Fallback: return as-is
    return university_query


def normalize_department_name(department_query: str) -> str:
    """
    Normalize department name by removing suffixes (과, 부).

    This is used for fuzzy matching since DB contains both:
    - "컴퓨터공학" and "컴퓨터공학부"
    - "건설환경공학" and "건설환경공학부"

    Args:
        department_query: Raw department name (e.g., "컴퓨터공학", "컴퓨터공학과")

    Returns:
        Normalized department base name (e.g., "컴퓨터공학")
    """
    # Already normalized (no suffix)
    if not (department_query.endswith('과') or department_query.endswith('부')):
        return department_query

    # Remove suffix: 컴퓨터공학과 -> 컴퓨터공학
    return department_query[:-1]


def extract_filters(question: str) -> Dict[str, str]:
    """
    Extract university, college, department, and grade information from question.

    Args:
        question: User's question string

    Returns:
        Dictionary with extracted filters (university, college, department, grade, semester)
    """
    filters = {}

    # Extract college name first (단과대학, e.g., 공과대학, 자연과학대학, etc.)
    # Must be done BEFORE university extraction to avoid "공과대학교" false positives
    college_pattern = r'([가-힣]+대학)(?!교)'  # "대학"으로 끝나지만 "대학교"는 제외
    college_match = re.search(college_pattern, question)
    if college_match:
        college_name = college_match.group(1)
        filters['college'] = college_name

    # Extract university name (e.g., 홍익대학교, 서울대학교, 홍대, 설대, etc.)
    # Skip if already matched as college
    university_pattern = r'([가-힣]+대학교|[가-힣]+대)'
    university_match = re.search(university_pattern, question)

    if university_match and not college_match:
        university_raw = university_match.group(1)
        # Normalize using mapping file (handles aliases and slang)
        university_name = normalize_university_name(university_raw)
        # Ensure it ends with 대학교
        if not university_name.endswith('대학교'):
            university_name += '학교'
        filters['university'] = university_name

    # 대학교/단과대학 이름을 질문에서 제거하여 학과 추출 시 혼동 방지
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
    Build Chroma-compatible metadata filter from extracted entities.

    Args:
        filters: Dictionary with university, college, department, grade, semester keys

    Returns:
        Chroma filter dictionary or None if no filters
    """
    if not filters:
        return None

    conditions = []

    if 'university' in filters:
        conditions.append({"university": {"$eq": filters['university']}})

    if 'college' in filters and filters['college']:
        # 단과대학 필터 (예: "공과대학", "자연과학대학")
        conditions.append({"college": {"$eq": filters['college']}})

    if 'department' in filters and filters['department']:
        # 정확한 매칭 사용 ($eq)
        # entity_extractor가 이미 정규화했으므로 (예: "컴퓨터공학")
        # DB의 department 필드와 정확히 일치함
        conditions.append({"department": {"$eq": filters['department']}})

    if 'grade' in filters:
        # Use $eq to match grade field (e.g., "2학년")
        conditions.append({"grade": {"$eq": filters['grade']}})

    if 'semester' in filters:
        conditions.append({"semester": {"$eq": filters['semester']}})

    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}
