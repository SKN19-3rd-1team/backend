# backend/graph/state.py
from typing import List, TypedDict, Optional, Dict, Any, Annotated
from typing_extensions import NotRequired
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MentorState(TypedDict):
    # ReAct 에이전트용 메시지 필드
    messages: Annotated[List[BaseMessage], add_messages]

    # 기존 필드들 (하위 호환성 유지)
    question: NotRequired[Optional[str]]                    # 사용자 질의
    interests: Optional[str]         # 사용자가 말한 관심사 / 진로 방향
    retrieved_docs: NotRequired[List[Document]]   # RAG 결과
    course_candidates: NotRequired[List[Dict[str, Any]]]  # 구조화된 과목 후보 리스트 (id, name, metadata 등)
    selected_course_ids: NotRequired[List[str]]  # LLM이 선택한 과목 ID 리스트
    answer: NotRequired[Optional[str]]            # 최종 답변
    metadata_filter_applied: NotRequired[bool]
    metadata_filter_relaxed: NotRequired[bool]
