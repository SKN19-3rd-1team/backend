# backend/graph/nodes.py
"""
LangGraph 그래프를 구성하는 노드 함수들을 정의합니다.

ReAct 패턴: LLM이 자율적으로 tool 호출 여부를 결정 (agent_node, should_continue)
"""

from langchain_core.messages import SystemMessage

from .state import MentorState
from backend.rag.retriever import (
    search_major_docs,
    aggregate_major_scores,
)
from backend.rag.embeddings import get_embeddings

from backend.rag.tools import (
    list_departments,
    get_universities_by_department,
    get_major_career_info,
    get_search_help,
    get_university_admission_info,
)

from backend.config import get_llm

# LLM 인스턴스 생성 (.env에서 설정한 LLM_PROVIDER와 MODEL_NAME 사용)
llm = get_llm()

# doc_type별 기본 가중치
MAJOR_DOC_WEIGHTS = {
    "summary": 0.8,
    "interest": 1.3,
    "property": 0.8,
    "subjects": 1.2,
    "jobs": 1.0,
}


# ==================== ReAct 에이전트용 설정 ====================
# ReAct 패턴: LLM이 필요시 자율적으로 툴을 호출할 수 있도록 설정
tools = [
    list_departments,
    get_universities_by_department,
    get_major_career_info,
    get_search_help,
    get_university_admission_info,
]  # 사용 가능한 툴 목록
llm_with_tools = llm.bind_tools(tools)  # LLM에 툴 사용 권한 부여


def _format_profile_value(value) -> str:
    # 온보딩 답변이 리스트/딕셔너리 등 다양한 형태여서 문자열로 균일하게 변환
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(items)
    if isinstance(value, dict):
        parts = []
        for key, sub_value in value.items():
            sub_text = _format_profile_value(sub_value)
            if sub_text:
                parts.append(f"{key}: {sub_text}")
        return "; ".join(parts)
    return str(value)


def _build_user_profile_text(answers: dict, fallback_question: str | None) -> str:
    # 학생의 선호 정보를 한 덩어리 텍스트로 만들어 임베딩에 활용
    if not answers and not fallback_question:
        return ""

    ordered_keys = [
        # ("preferred_majors", "관심 전공"), # [2025-12-12] 질문 제거됨
        ("subjects", "좋아하는 과목"),
        ("interests", "관심사/취미"),
        ("activities", "교내/대외 활동"),
        ("desired_salary", "희망 연봉"),
        ("career_goal", "진로 목표"),
        ("strengths", "강점"),
        ("career_field", "중요 가치관"),
        ("topics", "관심 주제"),
        ("learning_style", "학습 스타일"),
    ]

    sections: list[str] = []
    used_keys = {key for key, _ in ordered_keys}

    for field, label in ordered_keys:
        value = answers.get(field)
        formatted = _format_profile_value(value)
        if formatted:
            sections.append(f"{label}: {formatted}")

    # Capture any extra onboarding answers that were not explicitly mapped.
    for key, value in answers.items():
        if key in used_keys:
            continue
        formatted = _format_profile_value(value)
        if formatted:
            sections.append(f"{key}: {formatted}")

    if fallback_question and fallback_question.strip():
        sections.append(f"추가 요청: {fallback_question.strip()}")

    return "\n".join(sections).strip()


def _merge_tag_lists(existing: list[str], new_values: list[str]) -> list[str]:
    # 전공 태그는 중복을 허용하지 않으므로 순서를 보존하며 합집합 처리
    merged = list(existing)
    for value in new_values:
        if value not in merged:
            merged.append(value)
    return merged


def _summarize_major_hits(hits, aggregated_scores, limit: int = 10):
    # Pinecone 검색 결과를 전공별로 묶어 상위 doc_type/태그 등을 정리
    per_major: dict[str, dict] = {}

    # [버그 수정] 중복 제거
    # major_id가 다르더라도 major_name이 같으면 중복으로 처리
    seen_names = set()

    for hit in hits:
        if not hit.major_id:
            continue

        # 이름 기반 중복 체크
        if hit.major_name in seen_names:
            # 이미 같은 이름의 전공이 존재하면, 점수가 더 높은 경우 업데이트하거나 스킵
            # 여기서는 간단히 스킵 (hits가 점수순 정렬되어 있다고 가정하면 첫 번째가 베스트)
            # _summarize_major_hits 호출 전에 hits가 정렬되어 있지 않을 수 있으므로
            # 로직을 좀 더 정교하게: 이미 존재하는 entry를 찾아서 병합

            # 기존 entry 찾기 (이름으로)
            existing_id = None
            for mid, data in per_major.items():
                if data["major_name"] == hit.major_name:
                    existing_id = mid
                    break

            if existing_id:
                entry = per_major[existing_id]
            else:
                # Should not happen if seen_names logic is correct
                entry = per_major.setdefault(
                    hit.major_id,
                    {
                        "major_id": hit.major_id,
                        # ...
                    },
                )
        else:
            seen_names.add(hit.major_name)
            entry = per_major.setdefault(
                hit.major_id,
                {
                    "major_id": hit.major_id,
                    "major_name": hit.major_name,
                    "cluster": hit.metadata.get("cluster"),
                    "salary": hit.metadata.get("salary"),
                    "score": aggregated_scores.get(hit.major_id, 0.0),
                    "top_doc_types": {},
                    "sample_docs": [],
                    "relate_subject_tags": [],
                    "job_tags": [],
                    "summary": "",
                },
            )

        # 점수 업데이트 (Merge strategies)
        # 같은 이름의 다른 ID가 들어왔을 때, 점수가 더 높다면 score 업데이트?
        # 여기서는 aggregated_scores가 이미 ID별로 합산되었기 때문에
        # 이름이 같은 ID들의 점수를 합치거나 Max를 취해야 함.
        # 현재는 단순 스킵 방식이나 병합 방식을 써야 함.
        # 여기서는 "병합" 방식으로 구현: 태그 등을 합침.

        entry["top_doc_types"][hit.doc_type] = max(
            entry["top_doc_types"].get(hit.doc_type, 0.0),
            hit.score,
        )

        if len(entry["sample_docs"]) < 3:
            entry["sample_docs"].append(
                {
                    "doc_type": hit.doc_type,
                    "score": hit.score,
                    "text": hit.text,
                }
            )

        # summary doc_type인 경우 summary 필드에 저장
        if hit.doc_type == "summary" and not entry["summary"]:
            entry["summary"] = hit.text

        entry["relate_subject_tags"] = _merge_tag_lists(
            entry["relate_subject_tags"],
            hit.metadata.get("relate_subject_tags", []) or [],
        )
        entry["job_tags"] = _merge_tag_lists(
            entry["job_tags"],
            hit.metadata.get("job_tags", []) or [],
        )

    for entry in per_major.values():
        entry["top_doc_types"] = sorted(
            entry["top_doc_types"].items(),
            key=lambda item: item[1],
            reverse=True,
        )

    ordered = sorted(
        per_major.values(),
        key=lambda item: item["score"],
        reverse=True,
    )
    return ordered[:limit]


def _normalize_majors_with_llm(raw_majors: list[str]) -> list[str]:
    """
    LLM을 사용하여 사용자가 입력한 전공명(줄임말, 오타 등)을 표준 전공명으로 변환합니다.
    예: ["컴공", "화공"] -> ["컴퓨터공학과", "화학공학과"]
    """
    if not raw_majors:
        return []

    # 입력이 너무 많으면 처리 비용이 크므로 제한
    targets = raw_majors[:5]

    prompt = (
        "사용자가 입력한 대학 전공명(줄임말, 오타 포함)을 가장 적절한 '표준 학과명'으로 변환해주세요.\n"
        "학과명을 명확히 판단하기 힘들거나 오타가 있는경우, 사용자에게 자신이 추론한 '표준 학과명'이 맞는지 확인하는 절차를 거쳐주세요.\n"
        "반드시 한국어 학과명만 쉼표(,)로 구분하여 출력하세요. 설명이나 다른 말은 하지 마세요.\n\n"
        f"입력: {', '.join(targets)}\n"
        "출력:"
    )

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # 쉼표로 분리하여 리스트로 변환
        normalized = [item.strip() for item in content.split(",") if item.strip()]
        print(f"🤖 LLM Normalized Majors: {targets} -> {normalized}")
        return normalized
    except Exception as e:
        print(f"⚠️ Failed to normalize majors with LLM: {e}")
        return targets  # 실패 시 원본 반환


def recommend_majors_node(state: MentorState) -> dict:
    """
    온보딩 답변을 사용하여 사용자 프로필 임베딩을 생성하고 전공을 순위별로 추천합니다.
    우선순위: preferred_majors 정확 매칭 > 벡터 유사도 검색
    """
    onboarding_answers = state.get("onboarding_answers") or {}
    profile_text = _build_user_profile_text(onboarding_answers, state.get("question"))

    # 온보딩 답변이 없으면 빈 결과 반환
    if not profile_text:
        return {
            "user_profile_text": "",
            "recommended_majors": [],
            "major_search_hits": [],
            "major_scores": {},
        }

    # 1. 벡터 검색 (Vector Search)
    # 온보딩 텍스트를 단일 임베딩으로 변환하여 Pinecone에서 의미적으로 유사한 전공 문서를 검색합니다.
    embeddings = get_embeddings()
    profile_embedding = embeddings.embed_query(profile_text)

    # Pinecone에서 상위 50개 문서 검색
    hits = search_major_docs(profile_embedding, top_k=50)
    # 검색된 문서들의 점수를 전공별로 합산
    aggregated_scores = aggregate_major_scores(hits, MAJOR_DOC_WEIGHTS)

    recommended = _summarize_major_hits(hits, aggregated_scores)

    serialized_hits = [
        {
            "doc_id": hit.doc_id,
            "major_id": hit.major_id,
            "major_name": hit.major_name,
            "doc_type": hit.doc_type,
            "score": hit.score,
            "metadata": hit.metadata,
        }
        for hit in hits
    ]

    return {
        "user_profile_text": profile_text,
        "user_profile_embedding": profile_embedding,
        "major_search_hits": serialized_hits,
        "major_scores": aggregated_scores,
        "recommended_majors": recommended,
    }


# ==================== ReAct 스타일 에이전트 노드 ====================


def agent_node(state: MentorState) -> dict:
    """
    [ReAct 패턴] LLM이 자율적으로 tool 호출 여부를 결정.
    """
    messages = state.get("messages", [])
    interests = state.get("interests")

    # system_message는 interests 유무와 상관없이 항상 만들어둔다.
    system_message = None
    if not messages or not any(isinstance(m, SystemMessage) for m in messages):
        interests_text = f"{interests}" if interests else "없음"

        # ✅ f-string 내부 JSON 예시는 {{ }} 로 이스케이프!
        system_message = SystemMessage(
            content=f"""
당신은 학생들의 전공 선택을 돕는 '대학 전공 탐색 멘토'입니다. 모든 답변은 한국어로 작성하세요.

[핵심 원칙]
1. **툴 가이드 준수**: 각 툴(get_major_career_info, get_university_admission_info 등)의 설명(docstring)에 명시된 "사용 규칙"과 "제약 사항"을 철저히 따르세요.
2. **환각 방지**: 
   - 일반 전공 정보(취업률, 연봉)를 제공할 때, **절대 특정 대학의 정보인 것처럼** 대학명을 붙여서 설명하지 마세요.
   - 반드시 "OO대학의 구체적 정보는 없지만, 일반적인 OO학과의 정보는..." 형태로 분리하여 답변하세요.
3. **근거 기반 답변**: 반드시 툴 호출 결과를 바탕으로 답변하고, 추측하지 마세요. 데이터가 없으면 없다고 솔직히 말하세요.
4. **출처 명시 필수**: 
   - `get_major_career_info`를 통해 얻은 정보(취업률, 연봉, 진로 등)는 **반드시 '커리어넷' 기반임**을 밝혀야 합니다.
   - **절대** "한양대학교의 취업률은..." 이라고 답변하지 마세요. 대신 "한양대학교의 공식 취업률 자료는 확인되지 않았으나, 커리어넷의 일반적인 컴퓨터공학과 취업률은..." 이라고 답변하세요.
5. **유사 전공 허용**: 툴 검색 결과에 사용자가 query한 학과명과 정확히 일치하지 않는 학과가 있다면, 검색된 학과를 제시하세요.
6. **캠퍼스 구분**: '본교'와 '분교(ERICA, 세종, 글로컬 등)'는 서로 다른 대학으로 취급하여 명확히 구분해서 답변하세요. (예: "한양대학교는 컴퓨터소프트웨어학부, 한양대학교 ERICA는 컴퓨터학부가 개설되어 있습니다.")

[출력 제어]
- **[중요] `get_major_career_info` 호출 시 최적화**: 사용자가 특정 정보(예: 취업률, 진로, 배우는 과목 등)만 물어보는 경우, `specific_field` 파라미터를 사용하여 필요한 정보만 요청하세요. (예: `specific_field='stats'`)
- 사용자가 요청하지 않은 정보는 과도하게 나열하지 말고, 질문에 필요한 핵심 답변만 제공하세요.
- 친절하고 구조화된 설명을 제공하세요.
- **[중요]** 사용자가 처음 인사를 하거나, 무엇을 해야 할지 물어볼 때는 반드시 **"추천 시작"** 기능을 통해 맞춤형 전공 추천을 받을 수 있음을 안내하세요. (예: "저와 함께 나에게 딱 맞는 전공을 찾아볼까요? '추천 시작'이라고 말씀해 주세요!")
- **[예외 처리]** 만약 사용자가 "추천 시작"이라고 말했는데 이 메시지를 받았다면(프론트엔드 트리거 실패), "학과 목록"을 나열하지 말고, **"추천 기능을 시작하려면 '추천 시작'을 정확히 입력해 주세요."** 라고 안내하세요. 절대 `list_departments` 툴을 호출하여 일반 학과 목록을 보여주지 마세요.
                                       
학생 관심사: {interests_text}
"""
        )

    if system_message:
        messages = [system_message] + messages

    response = llm_with_tools.invoke(messages)

    # [MODIFICIATION] Removed internal retry loop to prevent token duplication in stream.
    # The prompt should be sufficient to encourage tool usage.
    # If the LLM responds without tools for greetings, it is acceptable.

    # 4. LLM의 응답(response)을 messages에 추가하여 상태 업데이트
    #    → should_continue가 tool_calls 유무를 확인하여 다음 노드 결정
    return {"messages": [response]}


def should_continue(state: MentorState) -> str:
    """
    [ReAct 패턴 라우팅] tool_calls 있으면 tools 노드로, 없으면 종료.
    """
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if last_message and getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"
