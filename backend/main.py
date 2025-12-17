# backend/main.py
"""
멘토 시스템의 메인 엔트리포인트.

프론트엔드(Streamlit)에서 이 파일의 run_mentor() 함수를 호출하여
사용자 질문에 대한 답변을 받습니다.
"""

from langchain_core.messages import HumanMessage
from .graph.graph_builder import build_graph

# 그래프 캐싱을 위한 전역 변수
# 그래프 빌드는 비용이 높으므로(컴파일 등), 한 번 빌드한 그래프를 메모리에 상주시켜 재사용합니다.
# 이를 통해 매 요청마다 그래프를 다시 만드는 오버헤드를 줄입니다.
_graph_react = None
_graph_major = None


def get_graph(mode: str = "react"):
    """
    LangGraph 인스턴스를 가져옵니다 (싱글톤 패턴, 캐싱됨).

    그래프 빌드는 비용이 높은 작업(컴파일, 도구 로딩 등)이므로,
    한 번 빌드한 그래프를 전역 변수(`_graph_react`, `_graph_major`)에 저장하여 재사용합니다.

    Args:
        mode (str): 그래프 실행 모드
            - "react": 일반 대화 및 RAG 검색을 수행하는 ReAct 에이전트 (기본값)
            - "major": 온보딩 정보 기반 전공 추천을 수행하는 그래프

    Returns:
        CompiledGraph: 컴파일된 LangGraph 애플리케이션 (invoke 가능)

    Raises:
        ValueError: 지원하지 않는 mode가 입력된 경우
    """
    global _graph_react, _graph_major

    if mode == "react":
        if _graph_react is None:
            _graph_react = build_graph(mode="react")
        return _graph_react
    elif mode == "major":
        if _graph_major is None:
            _graph_major = build_graph(mode="major")
        return _graph_major
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_mentor(
    question: str,
    interests: str | None = None,
    mode: str = "react",
    chat_history: list[dict] | None = None,
) -> str | dict:
    """
    멘토 시스템을 실행하여 학생의 질문에 답변합니다.
    API 서버에서 호출되는 메인 진입점입니다.

    ** 동작 흐름 **
    1. 요청된 `mode`에 맞는 LangGraph 인스턴스를 로드합니다 (캐싱 활용).
    2. `chat_history`가 있다면 LangChain `HumanMessage` 형태로 변환하여 문맥을 구성합니다.
    3. 사용자 `question`을 추가하여 그래프를 실행(`invoke`)합니다.
    4. 최종 상태에서 LLM의 답변(`Last Message`)을 추출하여 반환합니다.

    Args:
        question (str): 학생의 질문 (예: "컴퓨터공학과 전망 어때?")
        interests (str | None): (Legacy) 학생의 관심사/진로 방향 (현재 로직에서는 chat_history로 대체됨)
        mode (str): 실행 모드 ("react" or "major")
        chat_history (list[dict] | None): 이전 대화 기록 ([{"role": "user", "content": "..."}, ...])

    Returns:
        str | dict:
            - 일반적인 경우: LLM이 생성한 최종 답변 문자열
            - `awaiting_user_input` 상태인 경우: 그래프 상태 딕셔너리 (Human-in-the-loop 등)
    """
    # 1. 캐싱된 그래프 인스턴스 가져오기
    graph = get_graph(mode=mode)

    messages = []
    if chat_history:
        for msg in chat_history:
            # LLM이 이전 메시지를 이해하고 맥락을 이어가도록 함
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(HumanMessage(content=msg["content"]))

    # 마지막 질문을 추가
    messages.append(HumanMessage(content=question))

    if mode == "react":
        # ==================== ReAct 모드 ====================
        # messages 기반 상태 초기화
        state = {
            "messages": messages,  # 사용자 메시지로 시작
            "interests": interests,
        }

        # 그래프 실행: agent ⇄ tools 반복하며 답변 생성
        final_state = graph.invoke(state)

        if "awaiting_user_input" in final_state:
            return final_state

        # 마지막 메시지(LLM의 최종 답변)에서 텍스트 추출
        messages = final_state.get("messages", [])
        if messages:
            last_message = messages[-1]
            return last_message.content
        return "답변을 생성할 수 없습니다."


def run_mentor_stream(
    question: str,
    chat_history: list[dict] | None = None,
    mode: str = "react",
    stream_mode: str | list[str] = "updates",
):
    """
    멘토 시스템을 실행하고 결과를 스트리밍합니다 (제너레이터).
    views.py의 stream_chat_responses에서 사용됩니다.

    Args:
        question (str): 사용자 질문
        chat_history (list): 대화 기록
        mode (str): 실행 모드
        stream_mode (str | list[str]): LangGraph 스트리밍 모드

    Yields:
        dict: LangGraph 스트리밍 청크
    """
    graph = get_graph(mode=mode)

    messages = []
    if chat_history:
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(HumanMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    state = {
        "messages": messages,
    }

    # stream_mode="updates"를 사용하여 각 노드의 업데이트 사항을 스트리밍
    return graph.stream(state, stream_mode=stream_mode)


def run_major_recommendation(
    onboarding_answers: dict, question: str | None = None
) -> dict:
    """
    온보딩 단계에서 수집한 정보를 기반으로 Pinecone 전공 추천을 실행합니다.

    'major' 모드의 그래프를 사용하여 사용자 프로필을 분석하고,
    벡터 DB 검색 및 LLM 평가를 거쳐 최적의 전공을 추천합니다.

    Args:
        onboarding_answers (dict): 사용자 입력 딕셔너리
            - subjects (str): 선호 과목
            - interests (str): 관심사/취미
            - career_goal (str): 장래 희망
            - strengths (str): 강점
            - career_field (str, optional): 희망 진출 분야
        question (str | None): 추가 맥락 (선택 사항)

    Returns:
        dict: 추천 결과 데이터
            - user_profile_text (str): LLM이 요약한 사용자 페르소나
            - recommended_majors (list[dict]): 추천 전공 목록 (이름, 점수, 설명 등)
            - major_scores (dict): 주요 전공별 적합도 점수
            - major_search_hits (list): 벡터 검색 원본 결과 (디버깅용)
    """
    graph = get_graph(mode="major")
    state = {
        "onboarding_answers": onboarding_answers,
        "question": question,
    }
    final_state = graph.invoke(state)
    return {
        "user_profile_text": final_state.get("user_profile_text"),
        "recommended_majors": final_state.get("recommended_majors", []),
        "major_scores": final_state.get("major_scores", {}),
        "major_search_hits": final_state.get("major_search_hits", []),
    }
