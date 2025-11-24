# backend/graph/nodes.py
"""
LangGraph 그래프를 구성하는 노드 함수들을 정의합니다.

이 파일에는 두 가지 패턴이 공존합니다:
1. **ReAct 패턴**: LLM이 자율적으로 tool 호출 여부를 결정 (agent_node, should_continue)
2. **Structured 패턴**: 미리 정해진 순서대로 실행되는 파이프라인 (retrieve_node, select_node, answer_node)
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.constants import END
from pydantic import BaseModel, Field

from .state import MentorState
from backend.rag.retriever import retrieve_with_filter
from backend.rag.entity_extractor import extract_filters, build_chroma_filter
from backend.rag.tools import (
    retrieve_courses,
    list_departments,
    recommend_curriculum,
    match_department_name,  # ✅ 다시 포함
)

from backend.config import get_llm

# LLM 인스턴스 생성 (.env에서 설정한 LLM_PROVIDER와 MODEL_NAME 사용)
llm = get_llm()

# ==================== ReAct 에이전트용 설정 ====================
# ReAct 패턴: LLM이 필요시 자율적으로 툴을 호출할 수 있도록 설정
tools = [
    retrieve_courses,
    list_departments,
    recommend_curriculum,
    match_department_name,  # ✅ tools 바인딩에 포함
]
llm_with_tools = llm.bind_tools(tools)  # LLM에 툴 사용 권한 부여


# ==================== Structured 패턴용 Pydantic 모델 ====================
class CourseSelection(BaseModel):
    """
    Structured Output 패턴에서 사용하는 과목 선택 모델.
    LLM이 JSON 형식으로 선택한 과목 ID와 이유를 반환합니다.
    """
    selected_ids: List[str] = Field(
        description="후보 과목 리스트에서 학생에게 추천할 과목의 ID 리스트 (예: ['course_0', 'course_2'])"
    )
    reasoning: str = Field(
        description="해당 과목들을 선택한 간단한 이유"
    )


# ==================== Structured 패턴 노드들 ====================
# 이 노드들은 미리 정해진 순서대로 실행됩니다: retrieve → select → answer
# LLM이 노드 실행 여부를 선택하지 않습니다.

def retrieve_node(state: MentorState) -> dict:
    """
    [Structured 패턴 - 1단계] 벡터 DB에서 관련 과목 후보를 검색합니다.
    """
    question = state["question"]

    extracted_filters = extract_filters(question)
    chroma_filter = build_chroma_filter(extracted_filters) if extracted_filters else None

    search_k = 5
    docs: List[Document] = retrieve_with_filter(
        question=question,
        search_k=search_k,
        metadata_filter=chroma_filter
    )

    filter_applied = chroma_filter is not None
    filter_relaxed = False
    if filter_applied and not docs:
        print(
            "Warning: metadata filter returned no documents "
            f"({chroma_filter}). Falling back to unfiltered retrieval."
        )
        docs = retrieve_with_filter(
            question=question,
            search_k=search_k,
            metadata_filter=None
        )
        filter_relaxed = True

    course_candidates = []
    for idx, doc in enumerate(docs):
        meta = doc.metadata
        course_id = f"course_{idx}"

        candidate = {
            "id": course_id,
            "name": meta.get("name", "[과목명 미정]"),
            "university": meta.get("university", "[대학 정보 없음]"),
            "college": meta.get("college", "[단과대 정보 없음]"),
            "department": meta.get("department", "[학과 정보 없음]"),
            "grade_semester": meta.get("grade_semester", "[학년/학기 정보 없음]"),
            "classification": meta.get("course_classification", "[분류 정보 없음]"),
            "description": doc.page_content or "[설명 없음]"
        }
        course_candidates.append(candidate)

    return {
        "retrieved_docs": docs,
        "course_candidates": course_candidates,
        "metadata_filter_applied": filter_applied,
        "metadata_filter_relaxed": filter_relaxed,
    }


def select_node(state: MentorState) -> dict:
    """
    [Structured 패턴 - 2단계] 후보 과목 중 질문에 적합한 과목 ID만 선택.
    """
    question = state["question"]
    candidates = state.get("course_candidates", [])

    if not candidates:
        return {"selected_course_ids": []}

    candidate_lines = []
    for c in candidates:
        candidate_lines.append(
            f"- ID: {c['id']}\n"
            f"  과목명: {c['name']}\n"
            f"  대학: {c['university']}, 학과: {c['department']}\n"
            f"  학년/학기: {c['grade_semester']}, 분류: {c['classification']}\n"
            f"  설명: {c['description'][:100]}..."
        )
    candidates_text = "\n\n".join(candidate_lines)

    system_prompt = (
        "당신은 대학 전공 탐색 멘토입니다.\n"
        "아래 과목 후보 리스트에서 학생 질문에 가장 적합한 과목들의 ID를 선택하세요.\n\n"
        "**중요:**\n"
        "- 반드시 제공된 과목 리스트의 ID만 사용하세요.\n"
        "- 존재하지 않는 ID를 만들지 마세요.\n"
        "- 보통 2~3개 선택, 필요시 1~5개 가능.\n\n"
        "**응답 형식 (JSON만 출력):**\n"
        '{"selected_ids": ["course_0", "course_1"], "reasoning": "선택 이유"}\n'
        "다른 텍스트 없이 오직 JSON만 출력하세요."
    )

    user_prompt = (
        f"학생 질문: {question}\n\n"
        f"과목 후보 리스트:\n{candidates_text}\n\n"
        "위 후보 중에서 학생에게 추천할 과목의 ID를 JSON 형식으로 선택하세요."
    )

    try:
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        import json
        import re

        response_text = response.content.strip()
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        json_text = json_match.group(1) if json_match else response_text

        selection_data = json.loads(json_text)
        selected_ids = selection_data.get("selected_ids", [])

        valid_ids = {c["id"] for c in candidates}
        filtered_ids = [cid for cid in selected_ids if cid in valid_ids]

        return {"selected_course_ids": filtered_ids}

    except Exception as e:
        print(f"Warning: select_node JSON parsing failed: {e}")
        return {"selected_course_ids": [c["id"] for c in candidates[:3]]}


def answer_node(state: MentorState) -> dict:
    """
    [Structured 패턴 - 3단계] 선택된 과목들만 사용하여 최종 답변 생성.
    """
    question = state["question"]
    selected_ids = state.get("selected_course_ids", [])
    candidates = state.get("course_candidates", [])

    if not selected_ids:
        return {"answer": "죄송합니다. 질문에 맞는 적절한 과목을 찾지 못했습니다. 다른 질문을 해주시겠어요?"}

    id_to_candidate = {c["id"]: c for c in candidates}
    selected_courses = [id_to_candidate[cid] for cid in selected_ids if cid in id_to_candidate]

    if not selected_courses:
        return {"answer": "죄송합니다. 선택된 과목 정보를 찾을 수 없습니다."}

    lines = []
    for i, course in enumerate(selected_courses, start=1):
        lines.append(
            f"[{i}] 과목명: {course['name']}\n"
            f"    대학: {course['university']}, 학과: {course['department']}\n"
            f"    학년/학기: {course['grade_semester']}, 분류: {course['classification']}\n"
            f"    설명: {course['description']}"
        )
    context = "\n\n".join(lines)

    system_prompt = (
        "당신은 대학 전공 탐색 멘토입니다.\n"
        "학생 질문에 대해 아래 '선택된 과목 정보'만 사용해서 답하세요.\n\n"
        "**중요 지침:**\n"
        "1. 제공된 과목 외 새 과목을 만들지 마세요.\n"
        "2. 과목명을 그대로 사용하세요.\n"
        "3. 정보가 없으면 없다고 말하세요.\n"
        "4. 쉬운 말로 설명하고 진로 예시도 들어주세요.\n"
        "5. 같은 과목을 중복 언급하지 마세요."
    )

    user_prompt = (
        f"학생 질문: {question}\n\n"
        f"선택된 과목 정보:\n{context}\n\n"
        "이 과목들에 대해 자세히 설명하고 추천해 주세요."
    )

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return {"answer": response.content}


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
당신은 학생들의 전공 선택을 도와주는 '대학 전공 탐색 멘토'입니다.
반드시 한국어로만 답변하세요.

학생 관심사: {interests_text}

당신이 사용할 수 있는 툴은 다음 네 가지입니다:
1. retrieve_courses — 특정 과목을 검색
2. list_departments — 학과 목록 조회
3. recommend_curriculum — 학기/학년별 커리큘럼 추천
4. match_department_name — 학과명 정규화(컴공·소융·전전 등 → 공식 학과명)

──────────────────────────────────────────
[ TOOL CALL RULE — 가장 중요한 규칙 ]

- Tool이 필요한 질문이면 **반드시 tool_calls로 호출**해야 합니다.
- Tool 호출을 "글로 설명"하거나 JSON을 그냥 출력하면 안 됩니다.
- LangGraph가 실행하려면 **tool_calls 필드가 포함된 응답**이어야 합니다.

절대 금지:
- "retrieve_courses를 호출하겠습니다" 같은 말만 하고 호출 안 하기
- <tool_response> 같은 텍스트를 직접 출력하기
- 툴 없이 추측으로 답하기

──────────────────────────────────────────
[ 학과명 관련 규칙 ]

1. 질문에 학과명이 비표준(예: "컴공", "소융", "전전", "홍대 컴공")으로 들어오면  
   → **match_department_name을 먼저 호출**해서 표준 학과명으로 바꾸세요.
2. 표준화된 학과명/대학명을 확보한 뒤 retrieve_courses 또는 recommend_curriculum을 호출하세요.

──────────────────────────────────────────
[ 반드시 Tool 호출해야 하는 질문 유형 ]

- "~대 ~과 뭐 배워?"
- "~과 커리큘럼 알려줘"
- "~과 전공 내용?"
- "~과 무슨 과목 배우는지 알려줘"
- "이 학과의 2~4학년 커리 알려줘"

→ retrieve_courses 또는 recommend_curriculum 중 하나를 반드시 호출  
→ 추측 금지, 검색 기반으로만 답변

──────────────────────────────────────────
[ Follow-up 질문 처리 ]

학생의 후속 질문(예: "이 과목 몇 학년에 배워?", "이건 필수야?")이 오면:
1. **이전 tool 결과(messages)**를 먼저 확인하고
2. 정보 부족하면 tool을 재호출하세요.
3. tool 없이 추측으로 답하면 안 됩니다.

──────────────────────────────────────────
[ 학과 목록 질문 처리 ]

- "학과 뭐 있어?"
- "공대 학과 종류?"
- "컴퓨터 관련 학과 알려줘"

→ list_departments 반드시 호출

──────────────────────────────────────────
[ 응답 규칙 ]

- Tool이 필요하면 tool_calls 포함
- 필요 없으면 자연어로만 답변
- 항상 한국어로 답변
""")

        messages = [system_message] + messages

    response = llm_with_tools.invoke(messages)
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
