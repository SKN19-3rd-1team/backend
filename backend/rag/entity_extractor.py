# backend/rag/entity_extractor.py
"""Extract entities (university, department, grade) from user questions."""

import re
from typing import Dict, Optional


def extract_filters(question: str) -> Dict[str, str]:
    """
    Extract university, department, and grade information from question.

    Args:
        question: User's question string

    Returns:
        Dictionary with extracted filters (university, department, grade_semester)
    """
    filters = {}

    # Extract university name (e.g., 홍익대학교, 서울대학교, etc.)
    university_pattern = r'([가-힣]+대학교|[가-힣]+대)'
    university_match = re.search(university_pattern, question)
    if university_match:
        university_name = university_match.group(1)
        # Normalize: ensure it ends with 대학교
        if not university_name.endswith('대학교'):
            university_name += '학교'
        filters['university'] = university_name

    # 대학교 이름을 질문에서 제거하여 학과 추출 시 혼동 방지
    question_for_dept = question
    if university_match:
        question_for_dept = question.replace(university_match.group(0), '')

    # Extract department name (e.g., 컴퓨터공학과, 컴퓨터 공학과, 전자공학부, etc.)
    # 전체 학과명을 캡처한 후, "과"/"부"만 제거 (DB에 "컴퓨터공학"으로 저장되어 있음)
    department_patterns = [
        r'([가-힣\s]+공학)과',  # 컴퓨터공학과 → 컴퓨터공학
        r'([가-힣\s]+공학)부',  # 컴퓨터공학부 → 컴퓨터공학
        r'([가-힣\s]+)학과',    # 정보시스템학과 → 정보시스템
        r'([가-힣\s]+)학부',    # 전자전기학부 → 전자전기
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
        filters: Dictionary with university, department, grade keys

    Returns:
        Chroma filter dictionary or None if no filters
    """
    if not filters:
        return None

    conditions = []

    if 'university' in filters:
        conditions.append({"university": {"$eq": filters['university']}})

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
