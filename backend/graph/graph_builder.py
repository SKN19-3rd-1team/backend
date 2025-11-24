# backend/graph/graph_builder.py
"""
LangGraph 그래프를 빌드하는 팩토리 함수들을 정의합니다.

이 파일은 두 가지 다른 그래프 구조를 생성합니다:
1. **ReAct 그래프**: LLM이 자율적으로 tool을 호출하는 에이전트 패턴
2. **Structured 그래프**: 고정된 순서로 실행되는 파이프라인 패턴
"""

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
    멘토 시스템 그래프를 빌드합니다.

    Args:
        mode: 그래프 실행 모드
            - "react": ReAct 에이전트 방식 (LLM이 tool 호출 여부 자율 결정)
            - "structured": Structured Output 방식 (고정된 파이프라인)

    Returns:
        Compiled LangGraph application

    Raises:
        ValueError: 지원하지 않는 mode가 입력된 경우
    """
    if mode == "react":
        return build_react_graph()
    elif mode == "structured":
        return build_structured_graph()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'react' or 'structured'.")


def build_react_graph():
    """
    ReAct 스타일 에이전트 그래프를 빌드합니다.

    ** 그래프 구조 **
    ```
    [시작] → agent ⇄ tools → agent → [종료]
                ↓
               END
    ```

    ** 실행 플로우 **
    1. agent_node: LLM이 질문 분석하고 tool 호출 필요 여부 결정
    2. should_continue: tool_calls 확인
       - tool_calls 있음 → tools 노드로
       - tool_calls 없음 → 종료
    3. tools 노드: retrieve_courses 실제 실행
    4. agent_node로 복귀: LLM이 tool 결과 보고 답변 생성

    ** 특징 **
    - LLM이 자율적으로 tool 사용 결정
    - 필요시 여러 번 tool 호출 가능 (반복 루프)
    - Agentic한 동작
    """
    graph = StateGraph(MentorState)

    # 노드 추가
    graph.add_node("agent", agent_node)  # 핵심 에이전트 노드
    # 툴 실행 노드 - LangGraph가 여러 tool call을 병렬 실행하더라도
    # vectorstore.py의 _VECTORSTORE_LOCK이 동시 접근을 방지함
    graph.add_node("tools", ToolNode(tools))

    # 엣지 설정
    graph.set_entry_point("agent")  # 그래프 시작점

    # 조건부 엣지: agent → tools or END
    # should_continue가 tool_calls 확인하여 다음 노드 결정
    graph.add_conditional_edges(
        "agent",
        should_continue,  # 라우팅 함수
        {
            "tools": "tools",  # tool_calls 있으면 tools 노드로
            "end": END        # tool_calls 없으면 종료
        }
    )

    # tools → agent (툴 실행 후 다시 에이전트로 복귀)
    # 이 엣지 덕분에 agent ⇄ tools 반복 가능
    graph.add_edge("tools", "agent")

    # 그래프 컴파일 (실행 가능한 앱으로 변환)
    app = graph.compile()
    return app


def build_structured_graph():
    """
    Structured Output 기반 RAG 그래프를 빌드합니다.

    ** 그래프 구조 **
    ```
    [시작] → retrieve → select → answer → [종료]
    ```

    ** 실행 플로우 **
    1. retrieve_node: 벡터 DB에서 과목 후보 검색 (5개)
    2. select_node: LLM이 JSON으로 적합한 과목 ID만 선택 (2-3개)
    3. answer_node: 선택된 과목만 사용하여 최종 답변 생성

    ** 특징 **
    - 고정된 순서로 무조건 실행 (retrieve → select → answer)
    - LLM이 노드 실행 여부를 결정하지 않음 (Agentic하지 않음)
    - 파이프라인 방식
    - Hallucination 방지에 유리 (선택된 과목만 제공)

    ** ReAct와의 차이 **
    - ReAct: LLM이 tool 호출 여부 자율 결정
    - Structured: 무조건 retrieve → select → answer 순서대로 실행
    """
    graph = StateGraph(MentorState)

    # 노드 추가 (순차 실행)
    graph.add_node("retrieve", retrieve_node)  # 1단계: 검색
    graph.add_node("select", select_node)      # 2단계: 선택
    graph.add_node("answer", answer_node)      # 3단계: 답변 생성

    # 엣지 설정 (고정된 순서)
    graph.set_entry_point("retrieve")      # 시작점
    graph.add_edge("retrieve", "select")   # retrieve → select
    graph.add_edge("select", "answer")     # select → answer
    graph.add_edge("answer", END)          # answer → 종료

    # 그래프 컴파일
    app = graph.compile()
    return app
