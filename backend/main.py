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
_graph_structured = None

def get_graph(mode: str = "react"):
    """
    LangGraph 인스턴스를 가져옵니다 (싱글톤 패턴, 캐싱됨).

    그래프 빌드는 비용이 높은 작업이므로, 한 번 빌드한 그래프를 전역 변수에 저장하여 재사용합니다.

    Args:
        mode: 그래프 실행 모드
            - "react": ReAct 에이전트 방식 (기본값)
            - "structured": Structured Output 방식

    Returns:
        Compiled LangGraph application

    Raises:
        ValueError: 지원하지 않는 mode가 입력된 경우
    """
    global _graph_react, _graph_structured

    if mode == "react":
        if _graph_react is None:
            _graph_react = build_graph(mode="react")
        return _graph_react
    elif mode == "structured":
        if _graph_structured is None:
            _graph_structured = build_graph(mode="structured")
        return _graph_structured
    else:
        raise ValueError(f"Unknown mode: {mode}")

def run_mentor(question: str, interests: str | None = None, mode: str = "react", chat_history: list[dict] | None = None) -> str | dict:
    """
    멘토 시스템을 실행하여 학생의 질문에 답변합니다.

    ** 사용 예시 **
    ```python
    # ReAct 모드 (기본)
    answer = run_mentor("인공지능 관련 과목 추천해줘")

    # Structured 모드
    answer = run_mentor("데이터베이스 과목 알려줘", mode="structured")
    ```

    Args:
        question: 학생의 질문 (예: "인공지능 관련 과목 추천해줘")
        interests: 학생의 관심사/진로 방향 (선택, 현재 미사용)
        mode: 실행 모드
            - "react": ReAct 에이전트 방식 (기본값, LLM이 tool 호출 자율 결정)
            - "structured": Structured Output 방식 (고정 파이프라인)

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

    elif mode == "structured":
        # ==================== Structured 모드 ====================
        # question 기반 상태 초기화
        state = {
            "question": question,
            "interests": interests,
            "retrieved_docs": [],  # retrieve_node에서 채워짐
            "answer": None,  # answer_node에서 채워짐
        }

        # 그래프 실행: retrieve → select → answer 순차 실행
        final_state = graph.invoke(state)

        # answer_node가 생성한 최종 답변 반환
        return final_state["answer"]
