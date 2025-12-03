"""
ReAct 스타일 에이전트를 위한 LangChain Tools 정의

이 파일의 함수들은 @tool 데코레이터를 사용하여 LLM이 호출할 수 있는 툴로 등록됩니다.

** ReAct 패턴에서의 툴 역할 **
LLM이 사용자 질문을 분석하고, 필요시 자율적으로 이 툴들을 호출하여 정보를 수집합니다.

** 제공되는 툴들 **
1. list_departments: 학과 목록 조회
2. get_universities_by_department: 특정 학과가 있는 대학 조회
3. get_major_career_info: 전공별 진출 직업/분야 조회
4. get_search_help: 검색 실패 시 사용 가이드 제공

** 작동 방식 **
1. LLM이 사용자 질문 분석
2. LLM이 필요한 툴 선택 및 파라미터 결정
3. 툴 실행 (이 파일의 함수 호출)
4. 툴 결과를 LLM에게 전달
5. LLM이 결과를 바탕으로 최종 답변 생성
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.tools import tool
import re
import json
from pathlib import Path
from backend.config import get_settings

from .vectorstore import get_major_vectorstore
from .loader import load_major_detail
from .university_lookup import lookup_university_url, search_universities

# ==================== 상수 정의 ====================

# 검색 결과 제한
DEFAULT_SEARCH_LIMIT = 10
MAX_UNIVERSITY_RESULTS = 50
UNIVERSITY_PREVIEW_COUNT = 5
VECTOR_SEARCH_MULTIPLIER = 3

# 파일 경로
MAJOR_CATEGORIES_FILE = "major_categories.json"

# 출력 포맷
SEPARATOR_LINE = "=" * 80


# ==================== 로깅 유틸리티 ====================


def _log_tool_start(tool_name: str, description: str) -> None:
    """
    툴 실행 시작 로그 출력
    
    Args:
        tool_name: 툴 이름
        description: 실행 목적 설명
    """
    print(f"[Tool:{tool_name}] 시작 - {description}")


def _log_tool_result(tool_name: str, outcome: str) -> None:
    """
    툴 실행 결과 로그 출력
    
    Args:
        tool_name: 툴 이름
        outcome: 실행 결과 요약
    """
    print(f"[Tool:{tool_name}] 결과 - {outcome}")


# ==================== 사용자 가이드 ====================


def _get_tool_usage_guide() -> str:
    """
    사용자에게 제공할 툴 사용 가이드 메시지를 생성합니다.
    
    Returns:
        검색 가능한 방법들을 설명하는 가이드 메시지
    """
    return """
🤖 **Major Mentor 검색 가이드**

저희는 **전국 대학의 전공 정보, 개설 대학, 그리고 졸업 후 진로 데이터**를 보유하고 있습니다! 
궁금한 점을 아래와 같이 물어보시면 자세히 답변해 드릴 수 있어요.

### 1️⃣ **전공 탐색**
관심 있는 분야나 키워드로 어떤 학과들이 있는지 찾아보세요.
- "인공지능 관련 학과 알려줘"
- "공학 계열에는 어떤 전공이 있어?"
- "경영학과랑 비슷한 학과 추천해줘"

### 2️⃣ **개설 대학 찾기**
특정 학과가 어느 대학에 개설되어 있는지 알려드립니다.
- "컴퓨터공학과가 있는 대학 어디야?"
- "서울에 있는 심리학과 알려줘"
- "간호학과 개설 대학 목록 보여줘"

### 3️⃣ **진로 및 상세 정보**
졸업 후 어떤 직업을 갖게 되는지, 연봉이나 필요한 자격증은 무엇인지 확인해보세요.
- "컴퓨터공학과 나오면 무슨 일 해?"
- "기계공학과 졸업 후 연봉은 얼마야?"
- "사회복지학과 가려면 어떤 자격증이 필요해?"
- "경영학과에서는 주로 뭘 배워?"

💡 **팁**: 질문이 구체적일수록 더 정확한 정보를 드릴 수 있습니다!
"""


# ==================== 텍스트 처리 유틸리티 ====================


def _strip_html(value: str) -> str:
    """
    HTML 태그를 제거하고 텍스트만 반환
    
    Args:
        value: HTML이 포함된 문자열
        
    Returns:
        HTML 태그가 제거된 순수 텍스트
    """
    return re.sub(r"<[^>]+>", " ", value or "")


def _normalize_major_key(value: str) -> str:
    """
    전공명을 정규화하여 비교 가능한 형태로 변환
    공백 제거 및 소문자 변환
    
    Args:
        value: 원본 전공명
        
    Returns:
        정규화된 전공명 (공백 제거, 소문자)
    """
    return re.sub(r"\s+", "", (value or "").lower())


def _dedup_preserve_order(items: List[str]) -> List[str]:
    """
    리스트에서 중복을 제거하되 순서는 유지
    
    Args:
        items: 중복이 포함된 문자열 리스트
        
    Returns:
        중복이 제거된 문자열 리스트 (순서 유지)
    """
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


# ==================== 전공 카테고리 관리 ====================


def _load_major_categories() -> Dict[str, List[str]]:
    """
    backend/data/major_categories.json 파일에서 전공 분류 정보를 로드합니다.
    
    파일 구조:
    {
        "공학계열": ["컴퓨터 / 소프트웨어 / 인공지능", "전기 / 전자 / 통신", ...],
        "자연계열": ["수학 / 통계", "물리 / 화학", ...],
        ...
    }
    
    Returns:
        전공 카테고리 딕셔너리 (대분류 -> 세부분류 리스트)
    """
    try:
        # 현재 파일(tools.py)의 위치: backend/rag/
        # 데이터 파일 위치: backend/data/major_categories.json
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent  # backend/
        json_path = project_root / "data" / MAJOR_CATEGORIES_FILE
        
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))
        
        print(f"⚠️ Major categories file not found at: {json_path}")
        return {}
        
    except Exception as e:
        # 파일 로드 실패 시 에러 메시지 출력 및 빈 딕셔너리 반환
        print(f"⚠️ Failed to load major categories: {e}")
        return {}


# 전공 카테고리 전역 변수 (모듈 로드 시 1회만 실행)
MAIN_CATEGORIES = _load_major_categories()


def _expand_category_query(query: str) -> Tuple[List[str], str]:
    """
    list_departments용 쿼리 확장 함수
    
    사용자 입력을 분석하여 검색 토큰과 임베딩용 텍스트로 변환:
    - 대분류(key)를 넣으면: 해당 key에 속한 모든 세부 value들을 풀어서 키워드로 사용
    - 세부 분류(value)를 넣으면: "컴퓨터 / 소프트웨어 / 인공지능" → ["컴퓨터","소프트웨어","인공지능"]
    - 그 외 일반 텍스트: "/", "," 기준으로 토큰 나눈 뒤 사용
    
    Args:
        query: 사용자 입력 쿼리
        
    Returns:
        (tokens, embed_text) 튜플
        - tokens: 검색에 사용할 키워드 리스트
        - embed_text: 벡터 임베딩에 사용할 텍스트
    """
    raw = query.strip()
    if not raw:
        return [], ""

    tokens: List[str] = []

    # 1) 대분류(key) 입력인 경우 → 해당 key의 모든 세부 value를 한꺼번에 풀어서 사용
    if raw in MAIN_CATEGORIES:
        details = MAIN_CATEGORIES[raw]
        for item in details:
            # "컴퓨터 / 소프트웨어 / 인공지능" 형태를 개별 토큰으로 분리
            parts = [p.strip() for p in re.split(r"[\/,()]", item) if p.strip()]
            tokens.extend(parts)

    # 2) 세부 분류(value) 그대로 들어온 경우
    elif any(raw in v for values in MAIN_CATEGORIES.values() for v in values):
        parts = [p.strip() for p in re.split(r"[\/,()]", raw) if p.strip()]
        tokens.extend(parts)

    # 3) 일반 텍스트 쿼리 (예: "컴퓨터 / 소프트웨어 / 인공지능", "AI, 데이터")
    else:
        parts = [p.strip() for p in re.split(r"[\/,]", raw) if p.strip()]
        if parts:
            tokens.extend(parts)
        else:
            tokens.append(raw)

    # 중복 제거 (순서 유지)
    dedup_tokens = _dedup_preserve_order(tokens)
    
    # 임베딩용 텍스트 생성
    embed_text = " ".join(dedup_tokens) if dedup_tokens else raw
    
    return dedup_tokens, embed_text


# ==================== 전공 레코드 캐시 관리 ====================

# 전역 캐시 변수 (첫 호출 시 major_detail.json을 로드하여 메모리에 보관)
_MAJOR_RECORDS_CACHE: Optional[List[Any]] = None
_MAJOR_ID_MAP: Dict[str, Any] = {}      # major_id로 빠른 조회
_MAJOR_NAME_MAP: Dict[str, Any] = {}    # 정규화된 전공명으로 빠른 조회
_MAJOR_ALIAS_MAP: Dict[str, Any] = {}   # 별칭으로 빠른 조회


def _ensure_major_records() -> None:
    """
    전공 레코드를 메모리에 캐싱하고 인덱스 맵을 생성
    
    첫 호출 시에만 major_detail.json을 로드하여:
    1. _MAJOR_RECORDS_CACHE에 전체 레코드 저장
    2. _MAJOR_ID_MAP에 major_id 기반 인덱스 생성
    3. _MAJOR_NAME_MAP에 전공명 기반 인덱스 생성
    4. _MAJOR_ALIAS_MAP에 별칭 기반 인덱스 생성
    
    이후 호출 시에는 캐시된 데이터를 재사용
    """
    global _MAJOR_RECORDS_CACHE, _MAJOR_ID_MAP, _MAJOR_NAME_MAP, _MAJOR_ALIAS_MAP
    
    # 이미 캐시되어 있으면 스킵
    if _MAJOR_RECORDS_CACHE is not None:
        return

    # major_detail.json 로드
    records = load_major_detail()
    _MAJOR_RECORDS_CACHE = records
    
    # 인덱스 맵 초기화
    id_map: Dict[str, Any] = {}
    name_map: Dict[str, Any] = {}
    alias_map: Dict[str, Any] = {}

    # 각 레코드를 순회하며 인덱스 생성
    for record in records:
        # major_id 기반 인덱스
        if record.major_id:
            id_map[record.major_id] = record

        # 전공명 기반 인덱스
        if record.major_name:
            norm_name = _normalize_major_key(record.major_name)
            if norm_name:
                name_map[norm_name] = record
                alias_map.setdefault(norm_name, record)

        # 별칭 기반 인덱스
        for alias in getattr(record, "department_aliases", []) or []:
            norm_alias = _normalize_major_key(alias)
            if norm_alias and norm_alias not in alias_map:
                alias_map[norm_alias] = record

    # 전역 변수에 할당
    _MAJOR_ID_MAP = id_map
    _MAJOR_NAME_MAP = name_map
    _MAJOR_ALIAS_MAP = alias_map


def _get_major_records() -> List[Any]:
    """
    캐시된 전공 레코드 리스트 반환
    
    Returns:
        전체 MajorRecord 객체 리스트
    """
    _ensure_major_records()
    return _MAJOR_RECORDS_CACHE or []


def _lookup_major_by_name(name: str) -> Optional[Any]:
    """
    전공명 또는 별칭으로 전공 레코드 조회
    
    Args:
        name: 전공명 또는 별칭
        
    Returns:
        매칭되는 MajorRecord 객체 또는 None
    """
    if not name:
        return None
    
    _ensure_major_records()
    key = _normalize_major_key(name)
    
    # 정확한 전공명 매칭 우선, 없으면 별칭 매칭
    return _MAJOR_NAME_MAP.get(key) or _MAJOR_ALIAS_MAP.get(key)


# ==================== 벡터 검색 ====================


def _search_major_records_by_vector(query_text: str, limit: int) -> List[Any]:
    """
    Pinecone 벡터 데이터베이스를 사용한 전공 검색
    
    Args:
        query_text: 검색 쿼리 텍스트
        limit: 반환할 최대 결과 수
        
    Returns:
        유사도가 높은 순으로 정렬된 MajorRecord 리스트
    """
    if not query_text.strip():
        return []

    _ensure_major_records()
    
    # 벡터스토어 로드
    try:
        vectorstore = get_major_vectorstore()
    except Exception as exc:
        print(f"⚠️  Unable to load major vectorstore for query '{query_text}': {exc}")
        return []

    # 유사도 검색 실행
    try:
        docs = vectorstore.similarity_search(query_text, k=max(limit, 5))
    except Exception as exc:
        print(f"⚠️  Vector search failed for majors query '{query_text}': {exc}")
        return []

    # 검색 결과를 MajorRecord로 변환 (중복 제거)
    matches: List[Any] = []
    seen_ids: set[str] = set()
    
    for doc in docs:
        meta = doc.metadata or {}
        major_id = meta.get("major_id")
        
        # major_id가 없거나 이미 추가된 경우 스킵
        if not major_id or major_id in seen_ids:
            continue
            
        # 캐시에서 레코드 조회
        record = _MAJOR_ID_MAP.get(major_id)
        if record is None:
            continue
            
        seen_ids.add(major_id)
        matches.append(record)
        
        # 제한 수에 도달하면 중단
        if len(matches) >= limit:
            break
            
    return matches


def _filter_records_by_tokens(tokens: List[str], limit: int) -> List[Any]:
    """
    토큰 포함 여부로 전공 레코드 필터링
    
    모든 토큰이 전공명에 포함되어 있는 레코드만 반환
    
    Args:
        tokens: 검색 토큰 리스트
        limit: 반환할 최대 결과 수
        
    Returns:
        필터링된 MajorRecord 리스트
    """
    if not tokens:
        return []
        
    # 토큰을 소문자로 정규화
    normalized = [t.lower() for t in tokens if t]
    if not normalized:
        return []

    results: List[Any] = []
    seen_ids: set[str] = set()
    
    # 전체 레코드를 순회하며 필터링
    for record in _get_major_records():
        target = _normalize_major_key(record.major_name)
        
        # 모든 토큰이 전공명에 포함되어 있는지 확인
        if all(tok in target for tok in normalized):
            # 중복 제거
            if record.major_id and record.major_id in seen_ids:
                continue
                
            if record.major_id:
                seen_ids.add(record.major_id)
                
            results.append(record)
            
            # 제한 수에 도달하면 중단
            if len(results) >= limit:
                break
                
    return results


def _find_majors(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List[Any]:
    """
    통합 전공 검색 함수 (4단계 검색 전략)
    
    검색 우선순위:
    1. 정확히 일치하는 전공명 확인
    2. 토큰 별칭 확인 (정확 일치 없을 시)
    3. 벡터 유사도 검색 (항상 수행하여 연관 전공 포함)
    4. 토큰 포함 여부 필터링 (결과 없을 시 최후의 수단)
    
    Args:
        query: 검색 쿼리
        limit: 반환할 최대 결과 수
        
    Returns:
        검색된 MajorRecord 리스트 (최대 limit개)
    """
    _ensure_major_records()
    matches: List[Any] = []
    seen_ids: set[str] = set()

    # 1단계: 정확한 전공명 매칭
    direct = _lookup_major_by_name(query)
    if direct:
        matches.append(direct)
        if direct.major_id:
            seen_ids.add(direct.major_id)

    # 쿼리 확장 (카테고리 → 토큰 변환)
    tokens, embed_text = _expand_category_query(query)

    # 2단계: 별칭 검색 (정확한 매칭이 없을 경우)
    if not matches and tokens:
        for token in tokens:
            alias_match = _lookup_major_by_name(token)
            if alias_match and alias_match not in matches:
                matches.append(alias_match)
                if alias_match.major_id:
                    seen_ids.add(alias_match.major_id)

    # 3단계: 벡터 유사도 검색 (항상 수행하여 연관 전공 포함)
    search_text = embed_text or query
    vector_matches = _search_major_records_by_vector(
        search_text, 
        limit=max(limit * VECTOR_SEARCH_MULTIPLIER, DEFAULT_SEARCH_LIMIT)
    )
    
    for record in vector_matches:
        if record.major_id and record.major_id in seen_ids:
            continue
        matches.append(record)
        if record.major_id:
            seen_ids.add(record.major_id)
        if len(matches) >= max(limit, DEFAULT_SEARCH_LIMIT):
            break

    # 4단계: 토큰 필터링 (검색 결과가 없을 경우 최후의 수단)
    if not matches and tokens:
        token_matches = _filter_records_by_tokens(tokens, limit=max(limit, DEFAULT_SEARCH_LIMIT))
        for record in token_matches:
            if record.major_id and record.major_id in seen_ids:
                continue
            matches.append(record)
            if record.major_id:
                seen_ids.add(record.major_id)
            if len(matches) >= limit:
                break

    return matches[:limit]


# ==================== 대학 정보 추출 ====================


def _extract_university_entries(record: Any) -> List[Dict[str, str]]:
    """
    MajorRecord에서 대학 정보 추출
    
    Args:
        record: MajorRecord 객체
        
    Returns:
        대학 정보 딕셔너리 리스트
        [
            {
                "university": "서울대학교",
                "college": "공과대학",
                "department": "컴퓨터공학과",
                "area": "서울",
                "campus": "본교",
                "url": "https://...",
                "standard_major_name": "컴퓨터공학"
            },
            ...
        ]
    """
    entries: List[Dict[str, str]] = []
    raw_list = getattr(record, "university", None)
    
    if not isinstance(raw_list, list):
        return entries

    seen: set[Tuple[str, str, str]] = set()
    
    for item in raw_list:
        # 필드 추출 (다양한 키 이름 지원)
        school = (item.get("schoolName") or "").strip()
        campus = (item.get("campus_nm") or item.get("campusNm") or "").strip()
        major_name = (item.get("majorName") or "").strip()
        area = (item.get("area") or "").strip()
        url = (item.get("schoolURL") or "").strip()

        # 학과명 결정 (majorName이 있으면 우선 사용, 없으면 record.major_name 사용)
        dept_label = major_name or record.major_name
        
        # 대학명이 없으면 스킵
        if not school:
            continue

        # 중복 제거 (대학명, 학과명, 캠퍼스 조합)
        dedup_key = (school, dept_label, campus)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # 엔트리 생성
        entry: Dict[str, str] = {
            "university": school,
            "college": campus or area or "",
            "department": dept_label,
        }
        
        # 선택적 필드 추가
        if area:
            entry["area"] = area
        if campus:
            entry["campus"] = campus
        if url:
            entry["url"] = url
        if record.major_name and record.major_name != dept_label:
            entry["standard_major_name"] = record.major_name

        entries.append(entry)

    return entries


def _collect_university_pairs(record: Any, limit: int = 3) -> List[str]:
    """
    전공 레코드에서 "대학명 학과명" 형태의 문자열 리스트 생성
    
    Args:
        record: MajorRecord 객체
        limit: 반환할 최대 개수
        
    Returns:
        "대학명 학과명" 형태의 문자열 리스트
        예: ["서울대학교 컴퓨터공학과", "연세대학교 컴퓨터공학과", ...]
    """
    entries = _extract_university_entries(record)
    pairs: List[str] = []
    
    for entry in entries[:limit]:
        university = entry.get("university", "").strip()
        department = entry.get("department", "").strip()
        
        # 대학명과 학과명을 공백으로 연결
        label = " ".join(token for token in [university, department] if token)
        
        if label and label not in pairs:
            pairs.append(label)
            
    return pairs


# ==================== 진로 정보 추출 ====================


def _extract_job_list(job_text: str) -> List[str]:
    """
    진출 직업 텍스트를 개별 직업명 리스트로 분리
    
    Args:
        job_text: 쉼표/슬래시/줄바꿈으로 구분된 직업명 문자열
        
    Returns:
        중복이 제거된 직업명 리스트
    """
    if not job_text:
        return []
        
    # 구분자로 분리
    parts = re.split(r"[,/\n]", job_text)
    
    # 공백 제거 및 너무 짧은 항목 제외
    cleaned = [part.strip() for part in parts if len(part.strip()) > 1]
    
    # 중복 제거 (순서 유지)
    return _dedup_preserve_order(cleaned)


def _format_enter_field(record: Any) -> List[Dict[str, str]]:
    """
    major_detail.json의 enter_field 구조를 사용자에게 보여주기 쉬운 형태로 정리
    
    Args:
        record: MajorRecord 객체
        
    Returns:
        진출 분야 정보 리스트
        [
            {"category": "기업 및 산업체", "description": "..."},
            {"category": "연구소", "description": "..."},
            ...
        ]
    """
    formatted: List[Dict[str, str]] = []
    raw_list = getattr(record, "enter_field", None)
    
    if not isinstance(raw_list, list):
        return formatted

    for item in raw_list:
        if not isinstance(item, dict):
            continue
            
        # 카테고리 추출 (오타 대응: gradeuate/graduate)
        category = (item.get("gradeuate") or item.get("graduate") or "").strip()
        description = _strip_html(item.get("description") or "").strip()
        
        # 카테고리와 설명이 모두 없으면 스킵
        if not category and not description:
            continue
            
        entry: Dict[str, str] = {}
        if category:
            entry["category"] = category
        if description:
            entry["description"] = description
            
        formatted.append(entry)

    return formatted


def _format_career_activities(record: Any) -> List[Dict[str, str]]:
    """
    학과 준비 활동(career_act)을 act_name/description 짝으로 정리
    
    Args:
        record: MajorRecord 객체
        
    Returns:
        추천 활동 정보 리스트
        [
            {"act_name": "건축박람회", "act_description": "..."},
            {"act_name": "코딩대회", "act_description": "..."},
            ...
        ]
    """
    activities: List[Dict[str, str]] = []
    raw_list = getattr(record, "career_act", None)
    
    if not isinstance(raw_list, list):
        return activities

    for item in raw_list:
        if not isinstance(item, dict):
            continue
            
        name = (item.get("act_name") or "").strip()
        description = _strip_html(item.get("act_description") or "").strip()
        
        # 이름과 설명이 모두 없으면 스킵
        if not name and not description:
            continue
            
        entry: Dict[str, str] = {}
        if name:
            entry["act_name"] = name
        if description:
            entry["act_description"] = description
            
        activities.append(entry)

    return activities


def _parse_qualifications(record: Any) -> Tuple[str, List[str]]:
    """
    qualifications 필드를 문자열/리스트 여부에 관계없이 일관된 형태로 변환
    
    Args:
        record: MajorRecord 객체
        
    Returns:
        (joined_text, list) 튜플
        - joined_text: 쉼표로 연결된 자격증 문자열
        - list: 개별 자격증 리스트
    """
    raw_value = getattr(record, "qualifications", None)
    
    if raw_value is None:
        return "", []

    tokens: List[str] = []
    
    # 리스트 타입 처리
    if isinstance(raw_value, list):
        tokens = [str(item).strip() for item in raw_value if str(item).strip()]
    # 문자열 타입 처리
    else:
        text = str(raw_value).strip()
        if text:
            parts = [p.strip() for p in re.split(r"[,/\n]", text) if p.strip()]
            tokens = parts

    # 중복 제거
    deduped = _dedup_preserve_order(tokens)
    
    # 쉼표로 연결
    joined = ", ".join(deduped)
    
    return joined, deduped


def _format_main_subjects(record: Any) -> List[Dict[str, str]]:
    """
    main_subject 배열에서 과목명과 요약을 추출하여 정리
    
    Args:
        record: MajorRecord 객체
        
    Returns:
        주요 과목 정보 리스트
        [
            {"SBJECT_NM": "건축구조시스템", "SBJECT_SUMRY": "..."},
            {"SBJECT_NM": "건축설계", "SBJECT_SUMRY": "..."},
            ...
        ]
    """
    subjects: List[Dict[str, str]] = []
    raw_list = getattr(record, "main_subject", None)
    
    if not isinstance(raw_list, list):
        return subjects

    for item in raw_list:
        if not isinstance(item, dict):
            continue
            
        # 과목명 추출 (다양한 키 이름 지원)
        name = (item.get("SBJECT_NM") or item.get("subject_name") or "").strip()
        summary = _strip_html(
            item.get("SBJECT_SUMRY") or item.get("subject_description") or ""
        ).strip()
        
        # 과목명과 요약이 모두 없으면 스킵
        if not name and not summary:
            continue
            
        entry: Dict[str, str] = {}
        if name:
            entry["SBJECT_NM"] = name
        if summary:
            entry["SBJECT_SUMRY"] = summary
            
        subjects.append(entry)

    return subjects


def _resolve_major_for_career(query: str) -> Optional[Any]:
    """
    진로 정보 조회를 위한 전공 레코드 검색
    
    Args:
        query: 전공명 또는 별칭
        
    Returns:
        가장 관련성 높은 MajorRecord 객체 또는 None
    """
    if not query:
        return None

    # _find_majors를 사용하여 가장 관련성 높은 전공 1개 반환
    matches = _find_majors(query, limit=1)
    return matches[0] if matches else None


# ==================== 출력 포맷팅 ====================


def _format_department_output(
    query: str,
    departments: List[str],
    total_available: Optional[int] = None,
    dept_univ_map: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    학과 목록을 사용자 친화적인 형태로 포맷팅
    
    Args:
        query: 검색 쿼리
        departments: 학과명 리스트
        total_available: 전체 학과 수 (선택)
        dept_univ_map: 학과별 개설 대학 매핑 (선택)
        
    Returns:
        포맷팅된 학과 목록 문자열
    """
    lines = []
    
    # 헤더
    lines.append(SEPARATOR_LINE)
    lines.append(f"🎯 검색 결과: '{query}'에 대한 학과 {len(departments)}개")
    if total_available is not None:
        lines.append(f"(총 {total_available}개 중 상위 {len(departments)}개 표시)")
    lines.append(SEPARATOR_LINE)
    lines.append("")
    lines.append("📋 **정확한 학과명 목록** (아래 백틱 안의 이름을 그대로 복사하세요):")
    lines.append("")

    # 학과 목록
    for i, dept in enumerate(departments, 1):
        lines.append(f"{i}. `{dept}`")
        
        # 개설 대학 예시 추가
        if dept_univ_map:
            universities = dept_univ_map.get(dept)
            if universities:
                lines.append(f"   - 개설 대학 예시: {', '.join(universities)}")

    # 사용 가이드
    lines.append("")
    lines.append(SEPARATOR_LINE)
    lines.append("🚨 **중요 - 답변 작성 규칙**:")
    lines.append("   1. 백틱(`) 안의 학과명을 **한 글자도 바꾸지 말고** 복사하세요")
    lines.append("   2. 위 목록에 없는 학과명을 절대 만들지 마세요")
    lines.append("   3. '과', '부', '전공' 등을 추가/제거하지 마세요")
    lines.append("")
    lines.append("   올바른 예시:")
    lines.append("   - 목록에 `지능로봇`이 있으면 → 답변: **지능로봇** ✅")
    lines.append("   - 목록에 `화공학부`가 있으면 → 답변: **화공학부** ✅")
    lines.append("")
    lines.append("   잘못된 예시:")
    lines.append("   - 목록에 `지능로봇`인데 → 답변: **지능로봇공학과** ❌ (단어 추가)")
    lines.append("   - 목록에 `화공학부`인데 → 답변: **화공학과** ❌ (학부→학과 변경)")
    lines.append(SEPARATOR_LINE)

    return "\n".join(lines)


# ==================== LangChain Tools ====================


@tool
def list_departments(query: str, top_k: int = DEFAULT_SEARCH_LIMIT) -> str:
    """
    Pinecone majors vector DB를 기반으로 학과 목록을 조회하고 추천하는 툴입니다.

    이 툴을 호출해야 하는 상황 (LLM용 가이드):
    - 사용자가
      - "어떤 학과들이 있어?", "컴퓨터 관련 학과 알려줘"
      - "나의 관심사는 ~인데 어떤 전공이 좋을까?" (관심사 키워드로 검색)
      - "~와 비슷한 전공 추천해줘"
      와 같이 **전공/학과 목록 탐색**을 요청할 때 사용하세요.
    - 특정 학과의 상세 정보(진로, 연봉 등)나 개설 대학을 묻는 질문에는 이 툴이 아니라
      `get_major_career_info`나 `get_universities_by_department`를 사용해야 합니다.

    파라미터 설명:
    - query:
        검색하고 싶은 전공 분야, 관심사 키워드, 또는 "전체".
        예: "인공지능", "로봇", "경영", "전체"
    - top_k:
        반환할 학과 개수. 기본값은 10입니다.
    """
    raw_query = (query or "").strip()
    _log_tool_start("list_departments", f"학과 목록 조회 - query='{raw_query or '전체'}', top_k={top_k}")
    print(f"✅ Using list_departments tool with query: '{raw_query}'")

    _ensure_major_records()

    # 전체 목록 요청 처리
    if raw_query == "전체" or not raw_query:
        dept_univ_map: Dict[str, List[str]] = {}
        all_names = []
        
        # 모든 전공 레코드에서 전공명과 대학 정보 수집
        for record in _get_major_records():
            if not record.major_name:
                continue
                
            all_names.append(record.major_name)
            
            # 개설 대학 정보 수집
            pairs = _collect_university_pairs(record)
            if pairs:
                bucket = dept_univ_map.setdefault(record.major_name, [])
                for pair in pairs:
                    if pair not in bucket:
                        bucket.append(pair)
        
        # 정렬 및 제한
        all_names = sorted(set(all_names))
        limited = all_names[:top_k] if top_k else all_names
        
        print(f"✅ Returning {len(limited)} majors out of {len(all_names)} total")
        
        result_text = _format_department_output(
            raw_query or "전체",
            limited,
            total_available=len(all_names),
            dept_univ_map=dept_univ_map,
        )
        
        _log_tool_result("list_departments", f"총 {len(all_names)}개 중 {len(limited)}개 목록 반환")
        return result_text

    # 키워드 검색 처리
    tokens, embed_text = _expand_category_query(raw_query)
    print(f"   ℹ️ Expanded query tokens: {tokens}")
    print(f"   ℹ️ Embedding text: '{embed_text}'")

    # 통합 검색 실행
    matches = _find_majors(raw_query, limit=max(top_k, DEFAULT_SEARCH_LIMIT))
    dept_univ_map: Dict[str, List[str]] = {}

    # 각 매칭된 전공의 개설 대학 정보 수집
    for record in matches:
        pairs = _collect_university_pairs(record)
        if pairs:
            bucket = dept_univ_map.setdefault(record.major_name, [])
            for pair in pairs:
                if pair not in bucket:
                    bucket.append(pair)

    # 학과명 리스트 생성
    department_names = [record.major_name for record in matches if record.major_name]
    
    # 검색 결과가 없는 경우
    if not department_names:
        print("⚠️  WARNING: No majors found for the given query")
        _log_tool_result("list_departments", "검색 결과 없음")
        return "검색 결과가 없습니다. 다른 키워드로 검색해보세요."

    # 결과 제한 및 포맷팅
    result = department_names[:top_k]
    print(f"✅ Returning {len(result)} majors from major_detail vector DB")
    
    _log_tool_result("list_departments", f"{len(result)}개 학과 정보 반환")
    return _format_department_output(raw_query, result, dept_univ_map=dept_univ_map)


@tool
def get_major_career_info(major_name: str) -> Dict[str, Any]:
    """
    특정 전공(major)에 대한 상세 정보(진로, 연봉, 자격증, 주요 과목 등)를 조회하는 툴입니다.

    이 툴을 호출해야 하는 상황 (LLM용 가이드):
    - 사용자가
      - "컴퓨터공학과에 대해 알려줘" (단일 학과명 질문의 첫 단계)
      - "이 학과 나오면 무슨 일 해?", "졸업 후 진로가 어떻게 돼?"
      - "연봉은 얼마나 받아?"
      - "어떤 자격증이 필요해?", "무엇을 배워?"
      와 같이 **특정 학과의 상세 정보**를 물을 때 사용하세요.

    파라미터 설명:
    - major_name:
        정보를 조회할 학과명.
        예: "컴퓨터공학과", "경영학과"
    """
    query = (major_name or "").strip()
    _log_tool_start("get_major_career_info", f"전공 진로 정보 조회 - major='{query}'")
    print(f"✅ Using get_major_career_info tool for: '{query}'")

    # 입력 검증
    if not query:
        result = {
            "error": "invalid_query",
            "message": "전공명을 입력해 주세요.",
            "suggestion": "예: '컴퓨터공학과', '소프트웨어공학과'"
        }
        _log_tool_result("get_major_career_info", "전공명 누락 - 오류 반환")
        return result

    # 전공 레코드 검색
    record = _resolve_major_for_career(query)
    if record is None:
        print(f"⚠️  WARNING: No career data found for '{query}'")
        result = {
            "error": "no_results",
            "message": f"'{query}' 전공의 진출 직업 정보를 찾을 수 없습니다.",
            "suggestion": "학과명을 정확히 입력하거나 list_departments 툴로 전공명을 먼저 확인하세요."
        }
        _log_tool_result("get_major_career_info", "전공 데이터 미발견 - 오류 반환")
        return result

    # 진로 정보 추출
    job_text = (getattr(record, "job", "") or "").strip()
    job_list = _extract_job_list(job_text)
    enter_field = _format_enter_field(record)
    career_activities = _format_career_activities(record)
    qualifications_text, qualifications_list = _parse_qualifications(record)
    main_subjects = _format_main_subjects(record)

    # 연봉 정보 계산 (월평균 * 12)
    annual_salary = None
    if record.salary:
        try:
            annual_salary = float(record.salary) * 12
        except (ValueError, TypeError):
            pass

    # 응답 구성
    response: Dict[str, Any] = {
        "major": record.major_name,
        "jobs": job_list,
        "job_summary": job_text,
        "enter_field": enter_field,
        "source": "backend/data/major_detail.json",
        # 추가 통계 정보
        "gender_ratio": record.gender,
        "satisfaction": record.satisfaction,
        "employment_rate": record.employment_rate,
        "acceptance_rate": record.acceptance_rate,
        "annual_salary": annual_salary,  # 연봉 정보 추가
    }

    # 선택적 필드 추가
    if career_activities:
        response["career_act"] = career_activities
    if qualifications_text:
        response["qualifications"] = qualifications_text
    if qualifications_list:
        response["qualifications_list"] = qualifications_list
    if main_subjects:
        response["main_subject"] = main_subjects

    # 경고 메시지 추가 (직업 목록이 없는 경우)
    if not job_list:
        response["warning"] = "데이터에 등록된 직업 목록이 없습니다."
    else:
        print(f"✅ Retrieved {len(job_list)} jobs for '{record.major_name}'")

    # 진출 분야 정보 로깅
    if enter_field:
        print(f"   ℹ️ Enter field categories: {[item.get('category') for item in enter_field]}")
        
    # 통계 정보 로깅
    if record.acceptance_rate:
        print(f"   ℹ️ Acceptance rate: {record.acceptance_rate}%")

    # 결과 로깅
    activity_info = f"활동 {len(career_activities)}건" if career_activities else "활동 정보 없음"
    subject_info = f"주요 과목 {len(main_subjects)}건" if main_subjects else "주요 과목 정보 없음"
    stats_info = []
    if record.acceptance_rate: stats_info.append(f"합격률 {record.acceptance_rate}%")
    if record.employment_rate: stats_info.append("취업률 정보 있음")
    stats_str = ", ".join(stats_info) if stats_info else "통계 정보 없음"
    
    _log_tool_result(
        "get_major_career_info",
        f"{record.major_name} - 직업 {len(job_list)}건, {activity_info}, {subject_info}, {stats_str} 반환",
    )
    
    return response


@tool
def get_universities_by_department(department_name: str) -> List[Dict[str, str]]:
    """
    특정 학과를 개설한 대학 목록을 조회하는 툴입니다.

    이 툴을 호출해야 하는 상황 (LLM용 가이드):
    - 사용자가
      - "컴퓨터공학과는 어느 대학에 있어?"
      - "서울에 있는 심리학과 대학 알려줘"
      - "고분자공학과 개설 대학 보여줘"
      와 같이 **특정 학과의 개설 대학 정보**를 요청할 때 사용하세요.
    - 단일 학과명 질문("컴퓨터공학과")이 들어왔을 때, `get_major_career_info` 호출 후
      이 툴을 연달아 호출하여 대학 정보도 함께 제공하면 좋습니다.

    파라미터 설명:
    - department_name:
        대학 목록을 찾고 싶은 학과명.
        예: "컴퓨터공학과", "심리학과"
    """
    query = (department_name or "").strip()
    _log_tool_start("get_universities_by_department", f"학과별 대학 조회 - department='{query}'")
    print(f"✅ Using get_universities_by_department tool for: '{query}'")

    # 입력 검증
    if not query:
        result = [{
            "error": "invalid_query",
            "message": "학과명을 입력해 주세요.",
            "suggestion": "예: '컴퓨터공학과', '소프트웨어학부'"
        }]
        _log_tool_result("get_universities_by_department", "학과명 누락 - 오류 반환")
        return result

    _ensure_major_records()

    # 전공 검색 (정확한 매칭 우선)
    matches: List[Any] = []
    direct = _lookup_major_by_name(query)
    
    if direct:
        matches.append(direct)
    else:
        # 정확히 일치하는 학과가 없으면 유사 학과 검색
        matches = _find_majors(query, limit=5)

    # 대학 정보 수집
    aggregated: List[Dict[str, str]] = []
    for record in matches:
        entries = _extract_university_entries(record)
        if entries:
            aggregated.extend(entries)
        # 최대 50개까지만 수집
        if len(aggregated) >= MAX_UNIVERSITY_RESULTS:
            break

    # 검색 결과가 없는 경우
    if not aggregated:
        print(f"⚠️  WARNING: No universities found offering '{query}' in major_detail.json")
        result = [{
            "error": "no_results",
            "message": f"'{query}' 학과를 개설한 대학 정보를 major_detail 데이터에서 찾을 수 없습니다.",
            "suggestion": "학과명을 정확히 입력하거나 list_departments 툴로 사용 가능한 전공명을 먼저 확인하세요."
        }]
        _log_tool_result("get_universities_by_department", "검색 결과 없음 - 오류 반환")
        return result

    # 결과 로깅
    print(f"✅ Found {len(aggregated)} university rows for '{query}'")
    for entry in aggregated[:UNIVERSITY_PREVIEW_COUNT]:
        print(
            f"   - {entry.get('university')} / {entry.get('college')} / "
            f"{entry.get('department')}"
        )
    
    _log_tool_result("get_universities_by_department", f"총 {len(aggregated)}건 대학 정보 반환")
    return aggregated


@tool
def get_search_help() -> str:
    """
    사용자의 질문을 처리할 적절한 툴을 찾지 못했거나, 검색 결과가 없을 때 도움말을 제공하는 툴입니다.

    이 툴을 호출해야 하는 상황 (LLM용 가이드):
    - 사용자의 질문이 너무 모호하여 어떤 툴을 써야 할지 판단이 서지 않을 때
    - 다른 툴을 호출했으나 결과가 비어있어("검색 결과 없음"), 사용자에게 검색 팁을 줘야 할 때
    - 사용자가 "어떻게 검색해야 해?", "도움말 보여줘"라고 직접 요청할 때

    이 툴은 별도의 파라미터 없이 호출하면 됩니다.
    """
    _log_tool_start("get_search_help", "검색 가이드 안내")
    print("ℹ️  Using get_search_help tool - providing usage guide to user")
    
    message = _get_tool_usage_guide()
    
    _log_tool_result("get_search_help", "사용자 가이드 메시지 반환")
    return message


@tool
def get_university_admission_info(university_name: str, department_name: str = "") -> Dict[str, Any]:
    """
    특정 대학의 입시 정보(정시컷, 수시컷 등)를 조회하는 툴입니다.
    
    이 툴을 호출해야 하는 상황 (LLM용 가이드):
    - 사용자가
      - "서울대학교 컴퓨터공학과 정시컷 알려줘"
      - "연세대학교 수시컷이 궁금해"
      - "고려대학교 입시 결과 보여줘"
      - "OO대학교 OO학과 입시 정보 알려줘"
      와 같이 **특정 대학의 입시 정보**를 요청할 때 사용하세요.
    
    파라미터 설명:
    - university_name:
        입시 정보를 조회할 대학명.
        예: "서울대학교", "연세대학교", "고려대학교"
    - department_name:
        (선택) 학과명. 제공되면 응답 메시지에 포함됩니다.
        예: "컴퓨터공학과", "경영학과"
    """
    query = (university_name or "").strip()
    dept = (department_name or "").strip()
    
    _log_tool_start(
        "get_university_admission_info", 
        f"대학 입시 정보 조회 - university='{query}', department='{dept}'"
    )
    print(f"✅ Using get_university_admission_info tool for: '{query}' / '{dept}'")
    
    # 입력 검증
    if not query:
        result = {
            "error": "invalid_query",
            "message": "대학명을 입력해 주세요.",
            "suggestion": "예: '서울대학교', '연세대학교', '고려대학교'"
        }
        _log_tool_result("get_university_admission_info", "대학명 누락 - 오류 반환")
        return result
    
    # 대학 정보 조회
    university_info = lookup_university_url(query)
    
    if university_info is None:
        print(f"⚠️  WARNING: No admission data found for '{query}'")
        
        # 유사한 대학명 검색
        similar_universities = search_universities(query)
        
        if similar_universities:
            similar_names = [u["university"] for u in similar_universities[:5]]
            result = {
                "error": "no_exact_match",
                "message": f"'{query}' 대학의 입시 정보를 찾을 수 없습니다.",
                "suggestion": f"다음 대학명 중 하나를 선택해주세요: {', '.join(similar_names)}",
                "similar_universities": similar_names
            }
        else:
            result = {
                "error": "no_results",
                "message": f"'{query}' 대학의 입시 정보를 찾을 수 없습니다.",
                "suggestion": "대학명을 정확히 입력해주세요. 예: '서울대학교', '연세대학교[본교]'"
            }
        
        _log_tool_result("get_university_admission_info", "대학 데이터 미발견 - 오류 반환")
        return result
    
    # 성공 응답 구성
    response: Dict[str, Any] = {
        "university": university_info["university"],
        "code": university_info["code"],
        "url": university_info["url"],
        "source": "KCUE (한국대학교육협의회)",
        "message": f"{university_info['university']}의 입시 정보를 찾았습니다."
    }
    
    # 학과명이 제공된 경우 메시지에 포함
    if dept:
        response["department"] = dept
        response["message"] = f"{university_info['university']} {dept}의 입시 정보를 찾았습니다."
    
    # 안내 메시지 추가
    response["guide"] = "입시제도에 대해서는 해당 URL 좌측 메뉴의 평가기준 및 입시결과를 참고해주세요!"
    
    print(f"✅ Found admission info for '{university_info['university']}'")
    print(f"   URL: {university_info['url']}")
    
    _log_tool_result(
        "get_university_admission_info",
        f"{university_info['university']} 입시 정보 URL 반환"
    )
    
    return response
