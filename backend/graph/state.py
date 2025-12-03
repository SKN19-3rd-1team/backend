# backend/graph/state.py
"""
LangGraph의 상태(State) 모델을 정의합니다.

State는 그래프의 노드들 사이에서 데이터를 전달하는 공유 저장소입니다.
각 노드는 state를 읽고, 업데이트된 값을 반환하여 다음 노드로 전달합니다.
"""

from typing import List, TypedDict, Optional, Dict, Any, Annotated
from typing_extensions import NotRequired
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MentorState(TypedDict):
    """
    대학 전공 탐색 멘토 시스템의 상태 모델.

    두 가지 패턴을 모두 지원하기 위해 필드가 나뉘어 있습니다:
    - ReAct 패턴: messages 필드 사용
    """

    # ==================== ReAct 패턴용 필드 ====================
    # add_messages: 메시지를 리스트에 추가하는 reducer 함수
    # agent_node와 tools 노드 간에 메시지를 주고받을 때 사용
    messages: Annotated[List[BaseMessage], add_messages]

    question: NotRequired[Optional[str]]  # 학생의 질문 (retrieve_node에서 사용)
    interests: Optional[str]  # 학생의 관심사/진로 방향 (현재 미사용, 향후 확장 가능)

    retrieved_docs: NotRequired[List[Document]]  # retrieve_node에서 검색한 원본 Document 리스트
    course_candidates: NotRequired[List[Dict[str, Any]]]  # retrieve_node에서 생성한 구조화된 과목 후보
    selected_course_ids: NotRequired[List[str]]  # select_node에서 선택한 과목 ID 리스트
    answer: NotRequired[Optional[str]]  # answer_node에서 생성한 최종 답변

    # 메타데이터 필터 적용 정보 (디버깅/로깅용)
    metadata_filter_applied: NotRequired[bool]  # 필터가 적용되었는지 여부
    metadata_filter_relaxed: NotRequired[bool]  # 필터가 완화되었는지 여부 (결과 없을 때)

    # Major recommendation pipeline
    onboarding_answers: NotRequired[Dict[str, Any]]  # 온보딩 단계에서 수집된 학생 선호
    user_profile_text: NotRequired[Optional[str]]  # 온보딩 정보를 요약한 텍스트
    user_profile_embedding: NotRequired[Optional[List[float]]]  # 임베딩된 학생 프로필
    major_search_hits: NotRequired[List[Dict[str, Any]]]  # Pinecone 검색 결과 요약
    major_scores: NotRequired[Dict[str, float]]  # 전공별 점수 집계
    recommended_majors: NotRequired[List[Dict[str, Any]]]  # 최종 추천 전공 리스트
