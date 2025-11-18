# backend/graph/state.py
from typing import List, TypedDict, Optional
from langchain_core.documents import Document

class MentorState(TypedDict):
    question: str                    # 사용자 질의
    interests: Optional[str]         # 사용자가 말한 관심사 / 진로 방향
    retrieved_docs: List[Document]   # RAG 결과
    answer: Optional[str]            # 최종 답변
