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
    우선순위: preferred_majors 정확 매칭 > 벡터 유사도 검색
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
    
    # 🎯 preferred_majors 우선 처리
    preferred_majors = onboarding_answers.get("preferred_majors")
    preferred_major_ids = set()
    
    if preferred_majors:
        # preferred_majors를 문자열 또는 리스트로 처리
        if isinstance(preferred_majors, str):
            preferred_list = [m.strip() for m in preferred_majors.split(",") if m.strip()]
        elif isinstance(preferred_majors, list):
            preferred_list = [str(m).strip() for m in preferred_majors if str(m).strip()]
        else:
            preferred_list = []
        
        if preferred_list:
            # tools.py의 검색 함수 사용하여 선호 전공 별도 검색
            from backend.rag.tools import _find_majors, _MAJOR_ID_MAP, _ensure_major_records
            _ensure_major_records()
            
            for preferred in preferred_list:
                print(f"🔍 Searching for preferred major: '{preferred}'")
                
                # 선호 전공 검색 (정확 매칭 + 벡터 검색)
                preferred_matches = _find_majors(preferred, limit=5)
                
                for record in preferred_matches:
                    if not record.major_id:
                        continue
                    
                    preferred_major_ids.add(record.major_id)
                    
                    # 기존 hits에 없으면 추가
                    if record.major_id not in aggregated_scores:
                        # 새로운 전공이므로 기본 점수 1.0 부여
                        aggregated_scores[record.major_id] = 1.0
                        print(f"✅ Added preferred major '{record.major_name}' to results")
                    
                    # 보너스 점수 적용 (5배로 강화)
                    original_score = aggregated_scores[record.major_id]
                    aggregated_scores[record.major_id] = original_score * 5.0
                    print(f"🎯 Boosted '{record.major_name}' score: {original_score:.2f} → {aggregated_scores[record.major_id]:.2f}")
    
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

[🚨 절대 규칙 - 반드시 준수]
1. **툴에서 반환된 학과/대학/직업 이름만 사용**: Tool(list_departments, get_universities_by_department, get_major_career_info)이 돌려준 학과/대학/직업 이름은 **문자 하나도 바꾸지 말고 그대로 사용**합니다.
2. **절대 추측 금지**: 데이터베이스에 없는 학과명, 대학명, 직업명을 절대로 만들어내거나 추측하지 마세요. 
3. **툴 호출 필수**: 전공/학과/대학 관련 질문에는 반드시 적절한 툴을 tool_calls로 호출해 근거를 확보한 뒤 답변하세요.
4. **데이터 출처 명시**: 데이터 출처가 "커리어 넷"임을 자연스럽게 언급하세요.

[응답 방식]
- 항상 툴 결과를 바탕으로 친절하고 구조화된 설명을 제공합니다.
- 이미 받은 툴 결과가 있다면 재사용하고, 정보가 부족하면 같은 툴을 다시 호출해도 됩니다.
- tool_calls 없이 추측하려는 경우, get_search_help()를 호출해 검색 도움말을 제공하세요.

학생 관심사: {interests_text}
""")
                                       
    messages = [system_message] + messages
    
    # 🔍 입력 전처리: 단일 학과명 질문 감지 및 개선
    from backend.graph.helper import is_single_major_query, enhance_single_major_query
    
    # 마지막 사용자 메시지 확인
    last_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg
            break
    
    # 단일 학과명 질문이면 자동으로 명확한 질문으로 변환
    if last_user_msg and is_single_major_query(last_user_msg.content):
        original_query = last_user_msg.content
        enhanced_query = enhance_single_major_query(original_query)
        print(f"🔍 Detected single major query: '{original_query}'")
        print(f"✨ Enhanced to: '{enhanced_query}'")
        
        # 마지막 사용자 메시지를 개선된 버전으로 교체
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage) and messages[i] == last_user_msg:
                messages[i] = HumanMessage(content=enhanced_query)
                break

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
