# backend/graph/nodes.py
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.constants import END
from pydantic import BaseModel, Field

from .state import MentorState
from backend.rag.retriever import retrieve_with_filter
from backend.rag.entity_extractor import extract_filters, build_chroma_filter
from backend.rag.tools import retrieve_courses

from backend.config import get_llm

llm = get_llm()

# ReAct 에이전트용 LLM (툴 바인딩)
tools = [retrieve_courses]
llm_with_tools = llm.bind_tools(tools)

# Structured Output을 위한 Pydantic 모델
class CourseSelection(BaseModel):
    """학생 질문에 적합한 과목 ID를 선택하는 구조화된 출력"""
    selected_ids: List[str] = Field(
        description="후보 과목 리스트에서 학생에게 추천할 과목의 ID 리스트 (예: ['course_0', 'course_2'])"
    )
    reasoning: str = Field(
        description="해당 과목들을 선택한 간단한 이유"
    )

def retrieve_node(state: MentorState) -> dict:
    question = state["question"]

    extracted_filters = extract_filters(question)

    if not extracted_filters:
        chroma_filter = None
    else:
        chroma_filter = build_chroma_filter(extracted_filters)



    search_k = 5  # Retrieved candidates per question
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

    # retrieve_node에서 얻은 문서를 그대로 상태에 보관
    course_candidates = []
    for idx, doc in enumerate(docs):
        meta = doc.metadata
        course_id = f"course_{idx}"  # 조회 순번 기반 ID

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
        "retrieved_docs": docs,  # 검색된 LangChain 문서
        "course_candidates": course_candidates,
        "metadata_filter_applied": filter_applied,
        "metadata_filter_relaxed": filter_relaxed,
    }


def answer_node(state: MentorState) -> dict:
    """
    selected_course_ids를 기반으로 선택된 과목들만 사용하여 최종 답변 생성.
    LLM은 이미 선택된 과목 정보만 받으므로, 존재하지 않는 과목을 만들어낼 여지가 없음.
    """
    question = state["question"]
    selected_ids = state.get("selected_course_ids", [])
    candidates = state.get("course_candidates", [])

    if not selected_ids:
        return {"answer": "죄송합니다. 질문에 맞는 적절한 과목을 찾지 못했습니다. 다른 질문을 해주시겠어요?"}

    # selected_ids에 해당하는 과목만 필터링
    id_to_candidate = {c["id"]: c for c in candidates}
    selected_courses = [
        id_to_candidate[course_id]
        for course_id in selected_ids
        if course_id in id_to_candidate
    ]

    if not selected_courses:
        return {"answer": "죄송합니다. 선택된 과목 정보를 찾을 수 없습니다."}

    # 선택된 과목 정보를 텍스트로 포맷
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
        "학생의 질문에 대해 아래에 제공된 과목들을 바탕으로 친절하고 구체적으로 설명해 주세요.\n\n"
        "**중요 지침:**\n"
        "1. 아래 '선택된 과목 정보'에 나오는 과목명과 설명만 사용하세요.\n"
        "   - 제공된 목록에 없는 새로운 과목명을 절대로 만들지 마세요.\n"
        "   - 과목명을 정확히 그대로 사용하세요 (변형 금지).\n"
        "2. 각 과목의 특징, 배우는 내용, 학생에게 도움이 되는 점을 설명하세요.\n"
        "3. 과목 정보에 '[정보 없음]'이나 '[설명 정보가 제공되지 않았습니다]'가 있으면, "
        "   해당 정보가 데이터베이스에 없다고 명확히 전달하세요.\n"
        "4. 너무 어려운 용어는 풀어서 설명하고, 현실적인 진로 예시도 들어주세요.\n"
        "5. 같은 과목을 중복해서 언급하지 마세요."
    )

    user_prompt = (
        f"학생 질문: {question}\n\n"
        "아래는 학생에게 추천된 과목 정보입니다.\n"
        "이 과목들에 대해 자세히 설명하고 추천해 주세요.\n\n"
        f"선택된 과목 정보:\n{context}"
    )

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return {"answer": response.content}


def select_node(state: MentorState) -> dict:
    """
    course_candidates 중에서 학생 질문에 적합한 과목의 ID만 선택.
    JSON 형식으로 course_id 리스트만 반환하도록 프롬프트 강제.
    """
    question = state["question"]
    candidates = state.get("course_candidates", [])

    if not candidates:
        # 후보가 없으면 빈 리스트 반환
        return {"selected_course_ids": []}

    # 후보 과목 정보를 텍스트로 포맷
    candidate_lines = []
    for c in candidates:
        candidate_lines.append(
            f"- ID: {c['id']}\n"
            f"  과목명: {c['name']}\n"
            f"  대학: {c['university']}, 학과: {c['department']}\n"
            f"  학년/학기: {c['grade_semester']}, 분류: {c['classification']}\n"
            f"  설명: {c['description'][:100]}..."  # 설명은 100자까지만
        )
    candidates_text = "\n\n".join(candidate_lines)

    system_prompt = (
        "당신은 대학 전공 탐색 멘토입니다.\n"
        "아래에 제공된 과목 후보 리스트에서 학생의 질문에 가장 적합한 과목들의 ID를 선택하세요.\n\n"
        "**중요:**\n"
        "- 반드시 제공된 과목 리스트의 ID만 사용하세요.\n"
        "- 존재하지 않는 ID를 만들지 마세요.\n"
        "- 보통 2-3개 정도의 과목을 선택하되, 질문에 따라 1개 또는 최대 5개까지 선택 가능합니다.\n"
        "- 같은 과목을 중복해서 선택하지 마세요.\n\n"
        "**응답 형식 (반드시 JSON만 출력):**\n"
        '{"selected_ids": ["course_0", "course_1", ...], "reasoning": "선택 이유"}\n\n'
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

        # JSON 파싱
        import json
        import re

        response_text = response.content.strip()

        # JSON 블록 추출 (```json ... ``` 같은 마크다운 코드블록 처리)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # 마크다운 블록이 없으면 전체를 JSON으로 시도
            json_text = response_text

        selection_data = json.loads(json_text)
        selected_ids = selection_data.get("selected_ids", [])

        # 선택된 ID가 실제 후보 리스트에 있는지 검증
        valid_ids = {c["id"] for c in candidates}
        filtered_ids = [
            course_id for course_id in selected_ids
            if course_id in valid_ids
        ]

        return {"selected_course_ids": filtered_ids}

    except Exception as e:
        # JSON 파싱 실패 시 폴백: 모든 후보 반환
        print(f"Warning: select_node JSON parsing failed: {e}")
        print(f"Response was: {response.content[:200] if 'response' in locals() else 'N/A'}")
        return {"selected_course_ids": [c["id"] for c in candidates[:3]]}


# ==================== ReAct 스타일 에이전트 노드 ====================

def agent_node(state: MentorState) -> dict:
    """
    ReAct 스타일 에이전트 노드.
    LLM이 필요에 따라 retrieve_courses 툴을 호출하여 과목 정보를 가져옵니다.
    """
    messages = state.get("messages", [])

    # 첫 호출인 경우 시스템 프롬프트 추가
    if not messages or not any(isinstance(m, SystemMessage) for m in messages):
        system_message = SystemMessage(content=(
            "당신은 대학 전공 탐색 멘토입니다.\n"
            "학생들의 질문에 답변할 때는 반드시 'retrieve_courses' 툴을 사용하여 "
            "과목 데이터베이스에서 정보를 검색한 후 답변하세요.\n\n"
            "**중요 지침:**\n"
            "1. 학생이 과목 추천을 요청하면, 먼저 retrieve_courses 툴로 관련 과목을 검색하세요.\n"
            "2. 검색된 과목 정보만 사용하여 답변하세요. 절대로 존재하지 않는 과목을 만들어내지 마세요.\n"
            "3. 각 과목의 특징, 내용, 학생에게 도움이 되는 점을 친절하게 설명하세요.\n"
            "4. 너무 어려운 용어는 풀어서 설명하고, 현실적인 진로 예시도 들어주세요."
        ))
        messages = [system_message] + messages

    # LLM 호출
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


def should_continue(state: MentorState) -> str:
    """
    에이전트가 계속 실행할지, 종료할지 결정하는 조건부 엣지.
    """
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    # 마지막 메시지가 툴 호출을 포함하면 tools 노드로
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # 그렇지 않으면 종료
    return "end"
