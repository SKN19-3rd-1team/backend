# backend/graph/graph_builder.py
from langgraph.graph import StateGraph
from langgraph.constants import END
from langgraph.prebuilt import ToolNode
from .state import MentorState
from .nodes import (
    retrieve_node, select_node, answer_node,
    agent_node, should_continue, tools
)

def build_graph(mode: str = "react"):
    """
    LangGraph 기반 멘토 시스템 그래프 빌더.

    Args:
        mode: "react" (ReAct 에이전트) 또는 "structured" (Structured Output)

    Returns:
        Compiled graph application
    """
    if mode == "react":
        return build_react_graph()
    elif mode == "structured":
        return build_structured_graph()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'react' or 'structured'.")


def build_react_graph():
    """
    ReAct 스타일 에이전트 그래프 (1번 개선 방안):
    - agent: 질문 이해 → 필요시 retrieve_courses 툴 호출
    - tools: 실제 툴 실행 (ToolNode)
    - agent가 툴 결과를 보고 최종 답변 생성
    """
    graph = StateGraph(MentorState)

    # 노드 추가
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    # 엣지 설정
    graph.set_entry_point("agent")

    # 조건부 엣지: agent → tools or END
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    # tools → agent (툴 실행 후 다시 에이전트로)
    graph.add_edge("tools", "agent")

    app = graph.compile()
    return app


def build_structured_graph():
    """
    Structured Output 기반 RAG 그래프 (4번 개선 방안, 이미 구현됨):
    1. retrieve: 벡터DB에서 과목 후보 검색 및 구조화
    2. select: LLM이 JSON으로 적합한 과목 ID만 선택
    3. answer: 선택된 과목 정보만 사용하여 최종 답변 생성
    """
    graph = StateGraph(MentorState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("select", select_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "select")
    graph.add_edge("select", "answer")
    graph.add_edge("answer", END)

    app = graph.compile()
    return app
