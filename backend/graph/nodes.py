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
from backend.rag.tools import retrieve_courses, list_departments, recommend_curriculum, get_search_help

from backend.config import get_llm

# LLM 인스턴스 생성 (.env에서 설정한 LLM_PROVIDER와 MODEL_NAME 사용)
llm = get_llm()

# ==================== ReAct 에이전트용 설정 ====================
# ReAct 패턴: LLM이 필요시 자율적으로 툴을 호출할 수 있도록 설정
tools = [retrieve_courses, list_departments, recommend_curriculum, get_search_help]  # 사용 가능한 툴 목록
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

    작동 방식:
    1. 학생의 질문에서 필터 조건 추출 (대학, 학년, 학기 등)
    2. 벡터 DB에서 유사도 기반 검색 수행
    3. 검색된 문서들을 course_candidates 형태로 변환하여 상태에 저장
    """
    question = state["question"]

    # 1. 질문에서 메타데이터 필터 추출 (예: "서울대 1학년" → university, grade)
    extracted_filters = extract_filters(question)

    if not extracted_filters:
        chroma_filter = None
    else:
        chroma_filter = build_chroma_filter(extracted_filters)

    # 2. 벡터 DB에서 유사도 검색 수행
    search_k = 5  # 검색할 과목 후보 개수
    docs: List[Document] = retrieve_with_filter(
        question=question,
        search_k=search_k,
        metadata_filter=chroma_filter
    )

    # 3. 필터가 너무 제한적이어서 결과가 없으면, 필터 없이 재검색
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

    # 4. 검색된 Document를 구조화된 course_candidates로 변환
    course_candidates = []
    for idx, doc in enumerate(docs):
        meta = doc.metadata
        course_id = f"course_{idx}"  # 순번 기반 임시 ID 부여

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

    # 5. 검색 결과를 상태(state)에 반환 → 다음 노드(select_node)로 전달됨
    return {
        "retrieved_docs": docs,  # 원본 LangChain 문서
        "course_candidates": course_candidates,  # 구조화된 과목 후보
        "metadata_filter_applied": filter_applied,  # 필터 적용 여부
        "metadata_filter_relaxed": filter_relaxed,  # 필터 완화 여부
    }


def answer_node(state: MentorState) -> dict:
    """
    [Structured 패턴 - 3단계] 선택된 과목들만 사용하여 최종 답변 생성.

    작동 방식:
    1. select_node에서 선택한 과목 ID들을 가져옴
    2. 해당 과목들의 상세 정보만 LLM에게 제공
    3. LLM이 제공된 과목들만 사용하여 학생에게 답변 생성

    ** 중요: LLM은 이미 선택된 과목 정보만 받으므로, 존재하지 않는 과목을 만들어낼 수 없음 **
    """
    question = state["question"]
    selected_ids = state.get("selected_course_ids", [])  # select_node에서 선택한 ID들
    candidates = state.get("course_candidates", [])  # retrieve_node에서 검색한 전체 후보

    # 1. 선택된 과목이 없으면 에러 메시지 반환
    if not selected_ids:
        return {"answer": "죄송합니다. 질문에 맞는 적절한 과목을 찾지 못했습니다. 다른 질문을 해주시겠어요?"}

    # 2. selected_ids에 해당하는 과목만 필터링
    id_to_candidate = {c["id"]: c for c in candidates}
    selected_courses = [
        id_to_candidate[course_id]
        for course_id in selected_ids
        if course_id in id_to_candidate
    ]

    if not selected_courses:
        return {"answer": "죄송합니다. 선택된 과목 정보를 찾을 수 없습니다."}

    # 3. 선택된 과목 정보를 텍스트로 포맷 (LLM이 읽기 쉽게)
    lines = []
    for i, course in enumerate(selected_courses, start=1):
        lines.append(
            f"[{i}] 과목명: {course['name']}\n"
            f"    대학: {course['university']}, 학과: {course['department']}\n"
            f"    학년/학기: {course['grade_semester']}, 분류: {course['classification']}\n"
            f"    설명: {course['description']}"
        )
    context = "\n\n".join(lines)

    # 4. LLM에게 명확한 지침을 주는 시스템 프롬프트 작성
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

    # 5. 사용자 프롬프트 작성 (질문 + 선택된 과목 정보)
    user_prompt = (
        f"학생 질문: {question}\n\n"
        "아래는 학생에게 추천된 과목 정보입니다.\n"
        "이 과목들에 대해 자세히 설명하고 추천해 주세요.\n\n"
        f"선택된 과목 정보:\n{context}"
    )

    # 6. LLM 호출하여 최종 답변 생성
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # 7. 생성된 답변을 상태에 반환 (그래프 종료)
    return {"answer": response.content}


def select_node(state: MentorState) -> dict:
    """
    [Structured 패턴 - 2단계] 검색된 과목 후보 중에서 질문에 적합한 과목들만 선택.

    작동 방식:
    1. retrieve_node에서 검색한 course_candidates를 받음 (예: 5개)
    2. LLM에게 각 과목의 정보를 제공하고, 질문에 맞는 과목의 ID만 선택하도록 요청
    3. LLM이 JSON 형식으로 선택한 과목 ID 리스트 반환
    4. 선택된 ID들을 검증하여 상태에 저장

    ** 목적: 검색된 5개 중에서 정말 적합한 2-3개만 골라내어 hallucination 방지 **
    """
    question = state["question"]
    candidates = state.get("course_candidates", [])  # retrieve_node에서 검색한 후보들

    # 1. 후보가 없으면 빈 리스트 반환
    if not candidates:
        return {"selected_course_ids": []}

    # 2. 후보 과목 정보를 텍스트로 포맷 (LLM이 읽기 쉽게)
    candidate_lines = []
    for c in candidates:
        candidate_lines.append(
            f"- ID: {c['id']}\n"
            f"  과목명: {c['name']}\n"
            f"  대학: {c['university']}, 학과: {c['department']}\n"
            f"  학년/학기: {c['grade_semester']}, 분류: {c['classification']}\n"
            f"  설명: {c['description'][:100]}..."  # 설명은 100자까지만 (너무 길면 토큰 낭비)
        )
    candidates_text = "\n\n".join(candidate_lines)

    # 3. LLM에게 과목 선택을 위한 시스템 프롬프트 작성
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

    # 4. 사용자 프롬프트 작성 (질문 + 후보 리스트)
    user_prompt = (
        f"학생 질문: {question}\n\n"
        f"과목 후보 리스트:\n{candidates_text}\n\n"
        "위 후보 중에서 학생에게 추천할 과목의 ID를 JSON 형식으로 선택하세요."
    )

    try:
        # 5. LLM 호출하여 과목 선택 수행
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        # 6. JSON 파싱
        import json
        import re

        response_text = response.content.strip()

        # 7. JSON 블록 추출 (```json ... ``` 같은 마크다운 코드블록 처리)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # 마크다운 블록이 없으면 전체를 JSON으로 시도
            json_text = response_text

        selection_data = json.loads(json_text)
        selected_ids = selection_data.get("selected_ids", [])

        # 8. 선택된 ID가 실제 후보 리스트에 있는지 검증 (보안)
        valid_ids = {c["id"] for c in candidates}
        filtered_ids = [
            course_id for course_id in selected_ids
            if course_id in valid_ids
        ]

        # 9. 검증된 ID 리스트를 상태에 반환 → answer_node로 전달됨
        return {"selected_course_ids": filtered_ids}

    except Exception as e:
        # 10. JSON 파싱 실패 시 폴백: 상위 3개 후보를 기본으로 반환
        print(f"Warning: select_node JSON parsing failed: {e}")
        print(f"Response was: {response.content[:200] if 'response' in locals() else 'N/A'}")
        return {"selected_course_ids": [c["id"] for c in candidates[:3]]}


# ==================== ReAct 스타일 에이전트 노드 ====================
# 이 노드들은 LLM이 자율적으로 tool 호출 여부를 결정합니다.
# LLM이 "지금 과목 검색이 필요한가?"를 판단하여 retrieve_courses를 호출합니다.

def agent_node(state: MentorState) -> dict:
    """
    [ReAct 패턴 - 핵심 노드] LLM이 자율적으로 tool 사용 여부를 결정합니다.

    ** ReAct 작동 방식 **
    1. LLM이 학생의 질문을 분석
    2. "과목 정보가 필요하다"고 판단하면 retrieve_courses 툴 호출 결정
    3. LLM이 tool_calls를 response에 포함하여 반환
    4. should_continue가 tool_calls를 감지하면 tools 노드로 라우팅
    5. tools 노드에서 실제 툴 실행 (retrieve_courses 함수 호출)
    6. 툴 결과를 다시 agent_node로 전달
    7. LLM이 툴 결과를 보고 최종 답변 생성

    ** Structured 패턴과의 차이 **
    - Structured: retrieve → select → answer (고정된 순서)
    - ReAct: agent가 필요시에만 tool 호출 (자율적 결정)
    """
    messages = state.get("messages", [])
    interests = state.get("interests")

    # 1. 첫 호출인 경우 시스템 프롬프트 추가 (tool 사용 지침 포함)
    if not messages or not any(isinstance(m, SystemMessage) for m in messages):
        # Interests 정보 포함
        interests_info = ""
        if interests:
            interests_info = f"\n\n**학생의 관심사**: {interests}\n"

        system_message = SystemMessage(content=(
            "당신은 대학 전공 탐색 멘토입니다.\n"
            "학생들의 질문에 **반드시 한국어로** 답변하세요."
            f"{interests_info}\n"
            "**사용 가능한 툴:**\n"
            "1. retrieve_courses [최우선]: 대학/학과/과목에 대한 모든 질문에 사용\n"
            "   - 예시: '홍익대학교 컴퓨터공학', '인공지능 과목 추천해줘', '서울대 전자공학과'\n"
            "   - 대학명이나 학과명이 언급되면 무조건 이 툴을 사용하세요!\n"
            "\n"
            "2. recommend_curriculum: 학기별 커리큘럼을 추천할 때 사용\n"
            "   - 예시: '2학년부터 4학년까지 커리큘럼 추천해줘', '전체 커리큘럼'\n"
            "\n"
            "3. list_departments [제한적]: 학과 목록만 조회할 때 사용 (과목 정보 X)\n"
            "   - 예시: '어떤 학과들이 있어?', '컴퓨터 관련 학과 목록'\n"
            "   - ⚠️ 특정 대학/학과가 언급되면 retrieve_courses를 대신 사용하세요\n"
            "\n"
            "4. get_search_help: 검색 결과가 없거나 정보를 가져올 수 없을 때 사용\n\n"
            "**절대적 규칙 (CRITICAL):**\n"
            "⚠️ 학생의 질문에 답변하기 전에 **반드시** 적절한 툴을 먼저 호출해야 합니다.\n"
            "⚠️ 당신의 사전 지식이나 학습된 정보로 **절대로** 직접 답변하지 마세요.\n"
            "⚠️ 툴 호출 없이 답변을 작성하는 것은 **엄격히 금지**됩니다.\n"
            "⚠️ 모든 답변은 반드시 툴에서 반환된 실제 데이터를 기반으로 작성되어야 합니다.\n\n"
            "**중요 지침 (우선순위 순서):**\n"
            "1. **[최우선]** 학생이 특정 대학명이나 학과명을 언급하면 **무조건 retrieve_courses** 툴을 사용하세요.\n"
            "   - 예시: '홍익대학교 컴퓨터공학', '서울대 전자공학과', '컴퓨터공학 과목', '인공지능 수업'\n"
            "   - 대학명 또는 학과명이 포함된 모든 질문은 retrieve_courses를 사용해야 합니다.\n"
            "\n"
            "2. 학생이 '2학년부터 4학년까지', '전체 커리큘럼', '학기별로' 같은 표현을 사용하면 recommend_curriculum 툴을 사용하세요.\n"
            "\n"
            "3. **[제한적 사용]** 학생이 명확히 '학과 목록', '어떤 학과들이 있어?', '전공 종류' 같이 **목록 조회**를 요청할 때만 list_departments 툴을 사용하세요.\n"
            "   - ⚠️ 주의: 특정 대학/학과가 언급되면 list_departments가 아닌 retrieve_courses를 사용해야 합니다.\n"
            "\n"
            f"4. recommend_curriculum 툴 호출 시, interests 파라미터에 '{interests}'를 반드시 전달하세요.\n"
            "5. **툴 결과가 'error': 'no_results'를 포함하면, 즉시 get_search_help 툴을 호출하여 사용자에게 검색 가능한 방법을 안내하세요.**\n"
            "6. 툴에서 검색된 정보만 사용하여 답변하세요. 절대로 존재하지 않는 과목이나 학과를 만들어내지 마세요.\n"
            "7. 각 과목/학과의 특징, 내용, 학생에게 도움이 되는 점을 친절하게 **한국어로** 설명하세요.\n"
            "8. 너무 어려운 용어는 풀어서 설명하고, 현실적인 진로 예시도 들어주세요.\n"
            "9. **답변은 항상 한국어로 작성하세요. 절대로 다른 언어를 사용하지 마세요.**\n\n"
            "**작동 순서:**\n"
            "1단계: 학생의 질문 분석 (대학명, 학과명, 과목명이 언급되었는지 확인)\n"
            "2단계: 툴 선택 기준\n"
            "   - 대학/학과명 언급됨 → retrieve_courses (최우선)\n"
            "   - 커리큘럼 요청 → recommend_curriculum\n"
            "   - 학과 목록만 요청 → list_departments (제한적)\n"
            "3단계: 선택한 툴 호출 (필수!)\n"
            "4단계: 툴 결과를 바탕으로 답변 작성\n"
            "5단계: 툴 결과가 없으면 get_search_help 호출"
        ))
        messages = [system_message] + messages

    # 2. LLM 호출 (llm_with_tools는 retrieve_courses 툴이 바인딩된 LLM)
    #    LLM이 자율적으로 tool 호출 여부를 결정합니다.
    #    - tool이 필요하면: response에 tool_calls 포함
    #    - tool이 필요없으면: response에 일반 텍스트만 포함
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
                "1. retrieve_courses: 과목 검색\n"
                "2. list_departments: 학과 목록\n"
                "3. recommend_curriculum: 커리큘럼 추천\n\n"
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
    [ReAct 패턴 - 라우팅 함수] 다음 노드를 결정하는 조건부 엣지.

    ** 작동 방식 **
    1. agent_node의 응답(last_message)을 확인
    2. tool_calls가 있으면 → "tools" 반환 → tools 노드로 이동
    3. tool_calls가 없으면 → "end" 반환 → 그래프 종료

    ** 예시 플로우 **
    - 학생: "인공지능 관련 과목 추천해줘"
    - agent_node: tool_calls=[retrieve_courses("인공지능")] → should_continue → "tools"
    - tools 노드: retrieve_courses 실행 → 결과 반환
    - agent_node: 결과 보고 답변 생성 → tool_calls=[] → should_continue → "end"

    이렇게 LLM이 tool 호출 여부를 자율적으로 제어합니다.
    """
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    # 마지막 메시지에서 tool_calls 확인
    # tool_calls가 있으면: LLM이 툴 사용을 원함 → tools 노드로 라우팅
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # tool_calls가 없으면: LLM이 최종 답변 완료 → 그래프 종료
    return "end"
