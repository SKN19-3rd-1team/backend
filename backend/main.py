# backend/main.py
from langchain_core.messages import HumanMessage
from .graph.graph_builder import build_graph

_graph_react = None
_graph_structured = None

def get_graph(mode: str = "react"):
    """
    그래프 인스턴스를 가져옵니다 (캐싱됨).

    Args:
        mode: "react" 또는 "structured"
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

def run_mentor(question: str, interests: str | None = None, mode: str = "react") -> str:
    """
    멘토 시스템을 실행합니다.

    Args:
        question: 사용자 질문
        interests: 사용자 관심사 (옵션)
        mode: "react" (ReAct 에이전트) 또는 "structured" (Structured Output)

    Returns:
        최종 답변 문자열
    """
    graph = get_graph(mode=mode)

    if mode == "react":
        # ReAct 모드: messages 기반
        state = {
            "messages": [HumanMessage(content=question)],
            "interests": interests,
        }
        final_state = graph.invoke(state)

        # 마지막 메시지에서 답변 추출
        messages = final_state.get("messages", [])
        if messages:
            last_message = messages[-1]
            return last_message.content
        return "답변을 생성할 수 없습니다."

    elif mode == "structured":
        # Structured 모드: 기존 방식
        state = {
            "question": question,
            "interests": interests,
            "retrieved_docs": [],
            "answer": None,
        }
        final_state = graph.invoke(state)
        return final_state["answer"]
