# backend/graph/nodes.py
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI  # 또는 sLLM
from langgraph.prebuilt import ToolNode
from langgraph.constants import END
from .state import MentorState
from backend.rag.retriever import get_retriever

llm = ChatOpenAI(model="gpt-4.1-mini")  # 예시

def retrieve_node(state: MentorState) -> MentorState:
    retriever = get_retriever()
    docs: List[Document] = retriever.invoke(state["question"])
    state["retrieved_docs"] = docs
    return state

def answer_node(state: MentorState) -> MentorState:
    docs = state.get("retrieved_docs", [])
    context = "\n\n".join(
        [f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)]
    )

    system_prompt = (
        "당신은 대학 전공 탐색 멘토입니다. "
        "학생의 관심사와 배경을 고려해 과목들을 설명하고 추천해 주세요. "
        "너무 어려운 용어는 풀어서 설명하고, 현실적인 진로 예시도 들어주세요."
    )

    user_prompt = (
        f"학생 질문: {state['question']}\n\n"
        f"참고할 과목 정보:\n{context}"
    )

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    state["answer"] = response.content
    return state
