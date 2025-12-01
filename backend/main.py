# backend/main.py
"""
멘토 시스템의 메인 엔트리포인트.

프론트엔드(Streamlit)에서 이 파일의 run_mentor() 함수를 호출하여
사용자 질문에 대한 답변을 받습니다.
"""

from langchain_core.messages import HumanMessage
from .graph.graph_builder import build_graph

# 그래프 캐싱을 위한 전역 변수
# 그래프 빌드는 비용이 높으므로, 한 번 빌드한 그래프를 재사용합니다.
_graph_react = None
_graph_major = None

def get_graph(mode: str = "react"):
    """
    LangGraph 인스턴스를 가져옵니다 (싱글톤 패턴, 캐싱됨).

    그래프 빌드는 비용이 높은 작업이므로, 한 번 빌드한 그래프를 전역 변수에 저장하여 재사용합니다.

    Args:
        mode: 그래프 실행 모드
            - "react": ReAct 에이전트 방식 (기본값)

    Returns:
        Compiled LangGraph application

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

def run_mentor(question: str, interests: str | None = None, mode: str = "react", chat_history: list[dict] | None = None) -> str | dict:
    """
    멘토 시스템을 실행하여 학생의 질문에 답변합니다.

    ** 사용 예시 **
    ```python
    # ReAct 모드 (기본)
    answer = run_mentor("인공지능 관련 과목 추천해줘")
    ```

    Args:
        question: 학생의 질문 (예: "인공지능 관련 과목 추천해줘")
        interests: 학생의 관심사/진로 방향 (선택, 현재 미사용)
        mode: 실행 모드
            - "react": ReAct 에이전트 방식 (기본값, LLM이 tool 호출 자율 결정)

    Returns:
        LLM이 생성한 최종 답변 문자열
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

        if 'awaiting_user_input' in final_state:
            return final_state

        # 마지막 메시지(LLM의 최종 답변)에서 텍스트 추출
        messages = final_state.get("messages", [])
        if messages:
            last_message = messages[-1]
            return last_message.content
        return "답변을 생성할 수 없습니다."


def run_major_recommendation(onboarding_answers: dict, question: str | None = None) -> dict:
    """
    온보딩 단계에서 수집한 정보를 기반으로 Pinecone 전공 추천을 실행합니다.

    Args:
        onboarding_answers: 선호 과목, 취미, 희망 연봉, 희망 학과 등 사용자 입력
        question: 선택 사항, 추가 맥락으로 사용할 마지막 사용자 발화

    Returns:
        {
            "user_profile_text": "...",
            "recommended_majors": [...],
            "major_scores": {...},
            "major_search_hits": [...],
        }
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
