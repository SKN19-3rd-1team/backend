# backend/graph/nodes.py
"""
LangGraph 그래프를 구성하는 노드 함수들을 정의합니다.

ReAct 패턴: LLM이 자율적으로 tool 호출 여부를 결정 (agent_node, should_continue)
"""

import re
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.constants import END

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
)

from backend.config import get_llm

# LLM 인스턴스 생성 (.env에서 설정한 LLM_PROVIDER와 MODEL_NAME 사용)
llm = get_llm()

# doc_type별 기본 가중치 (관심사/과목 비중을 약간 높게 설정)
MAJOR_DOC_WEIGHTS = {
    "summary": 1.0,
    "interest": 1.1,
    "property": 0.9,
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
        ("preferred_majors", "관심 전공"),
        ("subjects", "좋아하는 과목"),
        ("interests", "관심사/취미"),
        ("activities", "교내/대외 활동"),
        ("desired_salary", "희망 연봉"),
        ("career_goal", "진로 목표"),
        ("strengths", "강점"),
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

    for hit in hits:
        if not hit.major_id:
            continue
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
                "summary": "",  # summary 필드 추가
            },
        )

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


def recommend_majors_node(state: MentorState) -> dict:
    """
    Build a user profile embedding from onboarding answers and rank majors.
    """
    onboarding_answers = state.get("onboarding_answers") or {}
    profile_text = _build_user_profile_text(onboarding_answers, state.get("question"))

    if not profile_text:
        return {
            "user_profile_text": "",
            "recommended_majors": [],
            "major_search_hits": [],
            "major_scores": {},
        }

    # 온보딩 텍스트를 단일 임베딩으로 바꿔 Pinecone 검색에 사용
    embeddings = get_embeddings()
    profile_embedding = embeddings.embed_query(profile_text)

    hits = search_major_docs(profile_embedding, top_k=50)
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
    if not messages or not any(isinstance(m, SystemMessage) for m in messages):
        interests_text = f"{interests}" if interests else "없음"

        # ✅ f-string 내부 JSON 예시는 {{ }} 로 이스케이프!
        system_message = SystemMessage(content=f"""
당신은 학생들의 전공 선택을 돕는 '대학 전공 탐색 멘토'입니다. 모든 답변은 한국어로 작성하세요.

[핵심 원칙]
1. Tool(list_departments, get_universities_by_department, get_major_career_info)이 돌려준 학과/대학/직업 이름은 **문자 하나도 바꾸지 말고 그대로 사용**합니다. 새로운 명칭을 추측으로 만들지 마세요.
2. 전공 관련 질문에는 반드시 적절한 툴을 tool_calls로 호출해 근거를 확보한 뒤 답변하세요.
3. 학과 정보를 소개할 때는 가능하면 학과명과 그 학과를 개설한 대학 이름을 함께 보여주세요. list_departments의 "개설 대학 예시"나 get_universities_by_department 결과를 적극 활용합니다.

[진로/직업 정보 제공]
- "졸업 후 진로", "어떤 직업"과 같이 직업/취업을 묻는 질문은 무조건 get_major_career_info를 호출해 major_detail.json의 `job`/`enter_field` 데이터를 가져오세요.
- 반환된 `jobs` 리스트(예: 3D프린팅전문가, 가상현실전문가 등)를 그대로 나열하고, `enter_field` 설명을 바탕으로 분야별 진출처를 요약합니다.
- 데이터 출처가 "커리어 넷"임을 자연스럽게 언급하고, 추측으로 목록을 추가하지 마세요.

[관심사 기반 추천 절차]
1. 학생이 입력한 관심사를 쉼표(,)나 슬래시(/)로 분리해 키워드 목록을 만듭니다.
2. 각 키워드마다 list_departments(query=키워드)를 호출해 관련 학과와 대학 예시를 확보합니다.
3. 학생에게 답변할 때는 관심사별 소제목을 만들고, `**학과명** (개설 대학 예시)` 형식으로 정리하면서 무엇을 배우고 어떤 진로로 이어질 수 있는지 짧게 설명합니다.

[대학 정보 제공]
- 특정 학과가 어느 대학에 있는지 묻는다면 즉시 get_universities_by_department를 호출해 대학/학과 쌍을 그대로 전달하세요.

[응답 방식]
- 항상 툴 결과를 바탕으로 친절하고 구조화된 설명을 제공합니다.
- 이미 받은 툴 결과가 있다면 재사용하고, 정보가 부족하면 같은 툴을 다시 호출해도 됩니다.
- tool_calls 없이 추측하려는 경우, get_search_help()를 호출해 검색 도움말을 제공하세요.

학생 관심사: {interests_text}
""")
                                       
    messages = [system_message] + messages

    response = llm_with_tools.invoke(messages)


    # 3. 검증: 첫 번째 사용자 질문에 대해 툴을 호출하지 않았는지 확인
    # ToolMessage가 없다는 것은 아직 툴 결과를 받지 않았다는 의미
    from langchain_core.messages import ToolMessage
    has_tool_results = any(isinstance(m, ToolMessage) for m in messages)

    # 툴 결과가 없는 상태에서 LLM이 tool_calls 없이 답변하려고 하면 차단
    if not has_tool_results:
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            print("⚠️ WARNING: LLM attempted to answer without using tools. Forcing tool usage.")
            # 강제로 재시도 메시지 추가
            error_message = HumanMessage(content=(
                "❌ 오류: 당신은 툴을 사용하지 않고 답변하려고 했습니다.\n"
                "**반드시 먼저 적절한 툴을 호출해야 합니다.**\n\n"
                "다시 한 번 강조합니다:\n"
                "1. list_departments: 학과 목록 검색\n"
                "2. get_universities_by_department: 특정 학과를 개설한 대학 검색\n"
                "3. get_major_career_info: 전공별 직업/진출 분야 확인\n"
                "4. get_search_help: 검색 도움말\n\n"
                "학생의 원래 질문을 다시 읽고, 적절한 툴을 **지금 즉시** 호출하세요."
            ))
            messages.append(error_message)

            # 재시도
            response = llm_with_tools.invoke(messages)

            # 재시도에도 툴을 사용하지 않으면 get_search_help로 폴백
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                print("⚠️ CRITICAL: LLM still refuses to use tools. Falling back to get_search_help.")
                from langchain_core.messages import AIMessage
                # 강제로 get_search_help 툴 호출 생성
                response = AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "get_search_help",
                        "args": {},
                        "id": "forced_search_help"
                    }]
                )

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
