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

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
import re
import json
from pathlib import Path
from backend.config import get_settings

from .vectorstore import get_major_vectorstore
from .loader import load_major_detail


def _log_tool_start(tool_name: str, description: str) -> None:
    # 각 LangChain Tool이 어떤 목적을 가지는지 콘솔에 명확히 남긴다
    print(f"[Tool:{tool_name}] 시작 - {description}")


def _log_tool_result(tool_name: str, outcome: str) -> None:
    # 툴 실행 결과(반환 건수, 상태 메시지 등)를 요약 출력
    print(f"[Tool:{tool_name}] 결과 - {outcome}")


def _get_tool_usage_guide() -> str:
    """
    사용자에게 제공할 툴 사용 가이드 메시지를 생성합니다.
    """
    return """
검색 가능한 방법들:

1. **학과 목록 조회**
   - 예시: "어떤 학과들이 있어?", "컴퓨터 관련 학과 알려줘", "공대에는 어떤 학과가 있어?"
   - 전체 학과 목록 또는 키워드로 필터링된 학과를 확인할 수 있습니다

2. **특정 학과가 있는 대학 조회**
   - 예시: "컴퓨터공학과가 있는 대학 알려줘", "소프트웨어학부 개설 대학"
   - 특정 학과를 개설한 대학 목록을 확인할 수 있습니다

3. **전공별 진출 직업/분야 조회**
   - 예시: "컴공 졸업하면 어떤 직업?", "OO학과 진로 알려줘"
   - get_major_career_info 툴을 호출하여 major_detail.json의 `job`/`enter_field` 데이터를 그대로 확인할 수 있습니다

더 구체적인 질문을 해주시면 더 정확한 정보를 제공해드릴 수 있습니다!
"""


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value or "")

# ===== 전공 대분류/세부분류 카테고리 =====
# ===== 전공 대분류/세부분류 카테고리 =====
def _load_major_categories() -> dict[str, list[str]]:
    """
    backend/data/major_categories.json 파일에서 전공 분류 정보를 로드합니다.
    """
    try:
        settings = get_settings()
        # Assuming major_categories.json is in the same directory as major_detail.json
        # or we can construct the path relative to this file or project root.
        # Let's try to use a fixed path or derive it.
        # Since we just created it in backend/data/major_categories.json:
        json_path = Path("/home/maroco/major_mentor/backend/data/major_categories.json")
        if not json_path.exists():
             # Fallback or try relative path if absolute fails in different envs (though we are in a specific env)
             base_dir = Path(__file__).parent.parent / "data"
             json_path = base_dir / "major_categories.json"
        
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))
        return {}
    except Exception as e:
        # 파일 로드 실패 시 에러 메시지 출력 및 빈 딕셔너리 반환
        print(f"⚠️ Failed to load major categories: {e}")
        return {}

MAIN_CATEGORIES = _load_major_categories()

# list_departments 쿼리 확장 함수
def _expand_category_query(query: str) -> tuple[list[str], str]:
    """
    list_departments용 쿼리 확장:
    - 대분류(key)를 넣으면: 해당 key에 속한 모든 세부 value들을 풀어서 키워드로 사용
    - 세부 분류(value)를 넣으면: "컴퓨터 / 소프트웨어 / 인공지능" → ["컴퓨터","소프트웨어","인공지능"]
    - 그 외 일반 텍스트: "/", "," 기준으로 토큰 나눈 뒤 사용

    Returns:
        tokens: ["컴퓨터", "소프트웨어", "인공지능", ...]
        embed_text: "컴퓨터 소프트웨어 인공지능 ..." (임베딩에 넣을 문자열)
    """
    raw = query.strip()
    if not raw:
        return [], ""

    tokens: list[str] = []

    # 1) 대분류(key) 입력인 경우 → 해당 key의 모든 세부 value를 한꺼번에 풀어서 사용
    if raw in MAIN_CATEGORIES:
        details = MAIN_CATEGORIES[raw]
        for item in details:
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

    # 중복 제거(순서 유지)
    seen = set()
    dedup_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            dedup_tokens.append(t)

    embed_text = " ".join(dedup_tokens) if dedup_tokens else raw
    return dedup_tokens, embed_text


# ==================== Major detail helpers ====================
_MAJOR_RECORDS_CACHE = None
_MAJOR_ID_MAP: dict[str, Any] = {}
_MAJOR_NAME_MAP: dict[str, Any] = {}
_MAJOR_ALIAS_MAP: dict[str, Any] = {}


def _normalize_major_key(value: str) -> str:
    return re.sub(r"\s+", "", (value or "").lower())


def _ensure_major_records():
    global _MAJOR_RECORDS_CACHE, _MAJOR_ID_MAP, _MAJOR_NAME_MAP, _MAJOR_ALIAS_MAP
    if _MAJOR_RECORDS_CACHE is not None:
        return

    records = load_major_detail()
    _MAJOR_RECORDS_CACHE = records
    id_map: dict[str, Any] = {}
    name_map: dict[str, Any] = {}
    alias_map: dict[str, Any] = {}

    for record in records:
        if record.major_id:
            id_map[record.major_id] = record

        if record.major_name:
            norm_name = _normalize_major_key(record.major_name)
            if norm_name:
                name_map[norm_name] = record
                alias_map.setdefault(norm_name, record)

        for alias in getattr(record, "department_aliases", []) or []:
            norm_alias = _normalize_major_key(alias)
            if norm_alias and norm_alias not in alias_map:
                alias_map[norm_alias] = record

    _MAJOR_ID_MAP = id_map
    _MAJOR_NAME_MAP = name_map
    _MAJOR_ALIAS_MAP = alias_map


def _get_major_records() -> list[Any]:
    _ensure_major_records()
    return _MAJOR_RECORDS_CACHE or []


def _lookup_major_by_name(name: str) -> Any | None:
    if not name:
        return None
    _ensure_major_records()
    key = _normalize_major_key(name)
    return _MAJOR_NAME_MAP.get(key) or _MAJOR_ALIAS_MAP.get(key)


def _search_major_records_by_vector(query_text: str, limit: int) -> list[Any]:
    if not query_text.strip():
        return []

    _ensure_major_records()
    try:
        vectorstore = get_major_vectorstore()
    except Exception as exc:
        print(f"⚠️  Unable to load major vectorstore for query '{query_text}': {exc}")
        return []

    try:
        docs = vectorstore.similarity_search(query_text, k=max(limit, 5))
    except Exception as exc:
        print(f"⚠️  Vector search failed for majors query '{query_text}': {exc}")
        return []

    matches: list[Any] = []
    seen_ids: set[str] = set()
    for doc in docs:
        meta = doc.metadata or {}
        major_id = meta.get("major_id")
        if not major_id or major_id in seen_ids:
            continue
        record = _MAJOR_ID_MAP.get(major_id)
        if record is None:
            continue
        seen_ids.add(major_id)
        matches.append(record)
        if len(matches) >= limit:
            break
    return matches


def _filter_records_by_tokens(tokens: list[str], limit: int) -> list[Any]:
    if not tokens:
        return []
    normalized = [t.lower() for t in tokens if t]
    if not normalized:
        return []

    results: list[Any] = []
    seen_ids: set[str] = set()
    for record in _get_major_records():
        target = _normalize_major_key(record.major_name)
        if all(tok in target for tok in normalized):
            if record.major_id and record.major_id in seen_ids:
                continue
            if record.major_id:
                seen_ids.add(record.major_id)
            results.append(record)
            if len(results) >= limit:
                break
    return results


def _find_majors(query: str, limit: int = 10) -> list[Any]:
    """
    통합 전공 검색 함수:
    1. 정확히 일치하는 전공명 확인
    2. (정확 일치 없을 시) 토큰 별칭 확인
    3. 벡터 유사도 검색 (항상 수행하여 연관 전공 포함)
    4. (결과 없을 시) 토큰 포함 여부 필터링
    """
    _ensure_major_records()
    matches: list[Any] = []
    seen_ids: set[str] = set()

    # 1. Direct Match (정확히 일치하는 전공명 검색)
    direct = _lookup_major_by_name(query)
    if direct:
        matches.append(direct)
        if direct.major_id:
            seen_ids.add(direct.major_id)

    tokens, embed_text = _expand_category_query(query)

    # 2. Alias Match (only if no direct match)
    # 2. 별칭 검색 (정확한 매칭이 없을 경우, 토큰별로 별칭 확인)
    if not matches and tokens:
        for token in tokens:
            alias_match = _lookup_major_by_name(token)
            if alias_match and alias_match not in matches:
                matches.append(alias_match)
                if alias_match.major_id:
                    seen_ids.add(alias_match.major_id)

    # 3. Vector Search (벡터 유사도 검색 - 항상 수행하여 연관 전공 포함)
    search_text = embed_text or query
    vector_matches = _search_major_records_by_vector(search_text, limit=max(limit * 3, 10))
    for record in vector_matches:
        if record.major_id and record.major_id in seen_ids:
            continue
        matches.append(record)
        if record.major_id:
            seen_ids.add(record.major_id)
        if len(matches) >= max(limit, 10):
            break

    # 4. Fallback Token Filter (if no matches yet)
    # 4. 토큰 필터링 (검색 결과가 없을 경우 최후의 수단)
    if not matches and tokens:
        token_matches = _filter_records_by_tokens(tokens, limit=max(limit, 10))
        for record in token_matches:
            if record.major_id and record.major_id in seen_ids:
                continue
            matches.append(record)
            if record.major_id:
                seen_ids.add(record.major_id)
            if len(matches) >= limit:
                break

    return matches[:limit]


def _format_department_output(
    query: str,
    departments: list[str],
    total_available: int | None = None,
    dept_univ_map: Optional[dict[str, list[str]]] = None,
) -> str:
    formatted_output = "=" * 80 + "\n"
    formatted_output += f"🎯 검색 결과: '{query}'에 대한 학과 {len(departments)}개\n"
    if total_available is not None:
        formatted_output += f"(총 {total_available}개 중 상위 {len(departments)}개 표시)\n"
    formatted_output += "=" * 80 + "\n\n"
    formatted_output += "📋 **정확한 학과명 목록** (아래 백틱 안의 이름을 그대로 복사하세요):\n\n"

    for i, dept in enumerate(departments, 1):
        formatted_output += f"{i}. `{dept}`\n"
        if dept_univ_map:
            universities = dept_univ_map.get(dept)
            if universities:
                formatted_output += f"   - 개설 대학 예시: {', '.join(universities)}\n"

    formatted_output += "\n" + "=" * 80 + "\n"
    formatted_output += "🚨 **중요 - 답변 작성 규칙**:\n"
    formatted_output += "   1. 백틱(`) 안의 학과명을 **한 글자도 바꾸지 말고** 복사하세요\n"
    formatted_output += "   2. 위 목록에 없는 학과명을 절대 만들지 마세요\n"
    formatted_output += "   3. '과', '부', '전공' 등을 추가/제거하지 마세요\n\n"
    formatted_output += "   올바른 예시:\n"
    formatted_output += "   - 목록에 `지능로봇`이 있으면 → 답변: **지능로봇** ✅\n"
    formatted_output += "   - 목록에 `화공학부`가 있으면 → 답변: **화공학부** ✅\n\n"
    formatted_output += "   잘못된 예시:\n"
    formatted_output += "   - 목록에 `지능로봇`인데 → 답변: **지능로봇공학과** ❌ (단어 추가)\n"
    formatted_output += "   - 목록에 `화공학부`인데 → 답변: **화공학과** ❌ (학부→학과 변경)\n"
    formatted_output += "=" * 80
    return formatted_output


def _extract_university_entries(record: Any) -> list[Dict[str, str]]:
    entries: list[Dict[str, str]] = []
    raw_list = getattr(record, "university", None)
    if not isinstance(raw_list, list):
        return entries

    seen: set[tuple[str, str, str]] = set()
    for item in raw_list:
        school = (item.get("schoolName") or "").strip()
        campus = (item.get("campus_nm") or item.get("campusNm") or "").strip()
        major_name = (item.get("majorName") or "").strip()
        area = (item.get("area") or "").strip()
        url = (item.get("schoolURL") or "").strip()

        dept_label = major_name or record.major_name
        if not school:
            continue

        dedup_key = (school, dept_label, campus)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        entry: Dict[str, str] = {
            "university": school,
            "college": campus or area or "",
            "department": dept_label,
        }
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


def _collect_university_pairs(record: Any, limit: int = 3) -> list[str]:
    entries = _extract_university_entries(record)
    pairs: list[str] = []
    for entry in entries[:limit]:
        university = entry.get("university", "").strip()
        department = entry.get("department", "").strip()
        label = " ".join(token for token in [university, department] if token)
        if label and label not in pairs:
            pairs.append(label)
    return pairs


def _dedup_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _extract_job_list(job_text: str) -> list[str]:
    if not job_text:
        return []
    parts = re.split(r"[,/\n]", job_text)
    cleaned = [part.strip() for part in parts if len(part.strip()) > 1]
    return _dedup_preserve_order(cleaned)


def _format_enter_field(record: Any) -> list[Dict[str, str]]:
    """
    major_detail.json의 enter_field 구조를 사용자에게 보여주기 쉬운 형태로 정리한다.
    """
    formatted: list[Dict[str, str]] = []
    raw_list = getattr(record, "enter_field", None)
    if not isinstance(raw_list, list):
        return formatted

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        category = (item.get("gradeuate") or item.get("graduate") or "").strip()
        description = _strip_html(item.get("description") or "").strip()
        if not category and not description:
            continue
        entry: Dict[str, str] = {}
        if category:
            entry["category"] = category
        if description:
            entry["description"] = description
        formatted.append(entry)

    return formatted


def _format_career_activities(record: Any) -> list[Dict[str, str]]:
    """
    학과 준비 활동(career_act)을 act_name/description 짝으로 정리해 LLM이 바로 읽도록 반환한다.
    """
    activities: list[Dict[str, str]] = []
    raw_list = getattr(record, "career_act", None)
    if not isinstance(raw_list, list):
        return activities

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        name = (item.get("act_name") or "").strip()
        description = _strip_html(item.get("act_description") or "").strip()
        if not name and not description:
            continue
        entry: Dict[str, str] = {}
        if name:
            entry["act_name"] = name
        if description:
            entry["act_description"] = description
        activities.append(entry)

    return activities


def _parse_qualifications(record: Any) -> tuple[str, list[str]]:
    """
    qualifications 필드를 문자열/리스트 여부에 관계없이 일관된 리스트와 문자열로 변환한다.
    """
    raw_value = getattr(record, "qualifications", None)
    if raw_value is None:
        return "", []

    tokens: list[str] = []
    if isinstance(raw_value, list):
        tokens = [str(item).strip() for item in raw_value if str(item).strip()]
    else:
        text = str(raw_value).strip()
        if text:
            parts = [p.strip() for p in re.split(r"[,/\n]", text) if p.strip()]
            tokens = parts

    deduped = _dedup_preserve_order(tokens)
    joined = ", ".join(deduped)
    return joined, deduped


def _format_main_subjects(record: Any) -> list[Dict[str, str]]:
    """
    main_subject 배열에서 과목명과 요약을 추출해 LLM 응답에 바로 포함할 수 있는 형태로 가공한다.
    """
    subjects: list[Dict[str, str]] = []
    raw_list = getattr(record, "main_subject", None)
    if not isinstance(raw_list, list):
        return subjects

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        name = (item.get("SBJECT_NM") or item.get("subject_name") or "").strip()
        summary = _strip_html(item.get("SBJECT_SUMRY") or item.get("subject_description") or "").strip()
        if not name and not summary:
            continue
        entry: Dict[str, str] = {}
        if name:
            entry["SBJECT_NM"] = name
        if summary:
            entry["SBJECT_SUMRY"] = summary
        subjects.append(entry)

    return subjects


def _resolve_major_for_career(query: str) -> Any | None:
    """Helper to find the most relevant major record for career info."""
    if not query:
        return None

    # Use _find_majors to get the best match
    matches = _find_majors(query, limit=1)
    return matches[0] if matches else None


@tool
def list_departments(query: str, top_k: int = 10) -> str:
    """
    Pinecone majors vector DB를 기반으로 학과 목록을 조회합니다.
    - query = "전체" → 전체 전공 목록을 반환 (상위 top_k까지만 표시)
    - query = "컴퓨터 / 소프트웨어 / 인공지능" → 해당 키워드와 유사한 전공을 검색
    - query = "컴공" 등 별칭 → major_detail.json에서 추출한 별칭 매핑/벡터 검색으로 정규화
    - 반환 포맷에는 학과명과 개설 대학 예시가 함께 포함됩니다.
    """
    raw_query = (query or "").strip()
    _log_tool_start("list_departments", f"학과 목록 조회 - query='{raw_query or '전체'}', top_k={top_k}")
    print(f"✅ Using list_departments tool with query: '{raw_query}'")

    _ensure_major_records()

    # 전체 목록 요청
    if raw_query == "전체" or not raw_query:
        dept_univ_map: dict[str, list[str]] = {}
        all_names = []
        for record in _get_major_records():
            if not record.major_name:
                continue
            all_names.append(record.major_name)
            pairs = _collect_university_pairs(record)
            if pairs:
                bucket = dept_univ_map.setdefault(record.major_name, [])
                for pair in pairs:
                    if pair not in bucket:
                        bucket.append(pair)
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

    tokens, embed_text = _expand_category_query(raw_query)
    print(f"   ℹ️ Expanded query tokens: {tokens}")
    print(f"   ℹ️ Embedding text: '{embed_text}'")

    matches = _find_majors(raw_query, limit=max(top_k, 10))
    dept_univ_map: dict[str, list[str]] = {}

    for record in matches:
        pairs = _collect_university_pairs(record)
        if pairs:
            bucket = dept_univ_map.setdefault(record.major_name, [])
            for pair in pairs:
                if pair not in bucket:
                    bucket.append(pair)

    department_names = [record.major_name for record in matches if record.major_name]
    if not department_names:
        print("⚠️  WARNING: No majors found for the given query")
        _log_tool_result("list_departments", "검색 결과 없음")
        return "검색 결과가 없습니다. 다른 키워드로 검색해보세요."

    result = department_names[:top_k]
    print(f"✅ Returning {len(result)} majors from major_detail vector DB")
    _log_tool_result("list_departments", f"{len(result)}개 학과 정보 반환")
    return _format_department_output(raw_query, result, dept_univ_map=dept_univ_map)


@tool
def get_major_career_info(major_name: str) -> Dict[str, Any]:
    """
    특정 전공(major)에 대한 세분화된 진출 직업 목록과 진출 분야 설명을 반환합니다.
    추가로 추천 활동, 관련 자격증, 주요 전공 과목 정보도 함께 제공합니다.

    Args:
        major_name: 전공명 또는 별칭 (예: "컴퓨터공학과", "AI융합학부")

    Returns:
        {
            "major": "컴퓨터공학과",
            "jobs": ["3D프린팅전문가", ...],
            "job_summary": "3D프린팅전문가, ...",
            "enter_field": [{"category": "기업 및 산업체", "description": "..."}, ...],
            "career_act": [{"act_name": "건축박람회", "act_description": "..."}, ...],
            "qualifications": "건축기사, ...",
            "qualifications_list": ["건축기사", ...],
            "main_subject": [{"SBJECT_NM": "건축구조시스템", "SBJECT_SUMRY": "..."}, ...],
            "source": "backend/data/major_detail.json"
        }
    """
    query = (major_name or "").strip()
    _log_tool_start("get_major_career_info", f"전공 진로 정보 조회 - major='{query}'")
    print(f"✅ Using get_major_career_info tool for: '{query}'")

    if not query:
        result = {
            "error": "invalid_query",
            "message": "전공명을 입력해 주세요.",
            "suggestion": "예: '컴퓨터공학과', '소프트웨어공학과'"
        }
        _log_tool_result("get_major_career_info", "전공명 누락 - 오류 반환")
        return result

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

    job_text = (getattr(record, "job", "") or "").strip()
    job_list = _extract_job_list(job_text)
    enter_field = _format_enter_field(record)
    career_activities = _format_career_activities(record)
    qualifications_text, qualifications_list = _parse_qualifications(record)
    main_subjects = _format_main_subjects(record)

    response: Dict[str, Any] = {
        "major": record.major_name,
        "jobs": job_list,
        "job_summary": job_text,
        "enter_field": enter_field,
        "source": "backend/data/major_detail.json"
    }

    if career_activities:
        response["career_act"] = career_activities
    if qualifications_text:
        response["qualifications"] = qualifications_text
    if qualifications_list:
        response["qualifications_list"] = qualifications_list
    if main_subjects:
        response["main_subject"] = main_subjects

    if not job_list:
        response["warning"] = "데이터에 등록된 직업 목록이 없습니다."
    else:
        print(f"✅ Retrieved {len(job_list)} jobs for '{record.major_name}'")

    if enter_field:
        print(f"   ℹ️ Enter field categories: {[item.get('category') for item in enter_field]}")

    activity_info = f"활동 {len(career_activities)}건" if career_activities else "활동 정보 없음"
    subject_info = f"주요 과목 {len(main_subjects)}건" if main_subjects else "주요 과목 정보 없음"
    _log_tool_result(
        "get_major_career_info",
        f"{record.major_name} - 직업 {len(job_list)}건, {activity_info}, {subject_info} 반환",
    )
    return response


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
    query = (department_name or "").strip()
    _log_tool_start("get_universities_by_department", f"학과별 대학 조회 - department='{query}'")
    print(f"✅ Using get_universities_by_department tool for: '{query}'")

    if not query:
        result = [{
            "error": "invalid_query",
            "message": "학과명을 입력해 주세요.",
            "suggestion": "예: '컴퓨터공학과', '소프트웨어학부'"
        }]
        _log_tool_result("get_universities_by_department", "학과명 누락 - 오류 반환")
        return result

    _ensure_major_records()

    matches: list[Any] = []
    direct = _lookup_major_by_name(query)
    if direct:
        matches.append(direct)
    else:
        # 정확히 일치하는 학과가 없으면 유사 학과 검색
        matches = _find_majors(query, limit=5)

    aggregated: list[Dict[str, str]] = []
    for record in matches:
        entries = _extract_university_entries(record)
        if entries:
            aggregated.extend(entries)
        if len(aggregated) >= 50:
            break

    if not aggregated:
        print(f"⚠️  WARNING: No universities found offering '{query}' in major_detail.json")
        result = [{
            "error": "no_results",
            "message": f"'{query}' 학과를 개설한 대학 정보를 major_detail 데이터에서 찾을 수 없습니다.",
            "suggestion": "학과명을 정확히 입력하거나 list_departments 툴로 사용 가능한 전공명을 먼저 확인하세요."
        }]
        _log_tool_result("get_universities_by_department", "검색 결과 없음 - 오류 반환")
        return result

    print(f"✅ Found {len(aggregated)} university rows for '{query}'")
    for entry in aggregated[:5]:
        print(
            f"   - {entry.get('university')} / {entry.get('college')} / "
            f"{entry.get('department')}"
        )
    _log_tool_result("get_universities_by_department", f"총 {len(aggregated)}건 대학 정보 반환")
    return aggregated


@tool
def get_search_help() -> str:
    """
    사용자 질문에 대한 정보를 가져올 수 없었을 때 사용하는 툴입니다.
    검색 가능한 방법들(각 툴을 호출할 수 있는 방법들)을 안내합니다.

    ** 언제 사용하나요? **
    1. 다른 툴(list_departments, get_universities_by_department)의 결과가 비어있을 때
    2. 사용자의 질문이 너무 모호하거나 데이터베이스에 없는 정보를 요청할 때
    3. 검색 결과가 없어서 사용자에게 다른 검색 방법을 안내해야 할 때

    Returns:
        검색 가능한 방법들을 설명하는 가이드 메시지
    """
    _log_tool_start("get_search_help", "검색 가이드 안내")
    print("ℹ️  Using get_search_help tool - providing usage guide to user")
    message = _get_tool_usage_guide()
    _log_tool_result("get_search_help", "사용자 가이드 메시지 반환")
    return message
