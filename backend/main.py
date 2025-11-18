# backend/main.py
from .graph.graph_builder import build_graph

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

def run_mentor(question: str, interests: str | None = None) -> str:
    graph = get_graph()
    state = {
        "question": question,
        "interests": interests,
        "retrieved_docs": [],
        "answer": None,
    }
    final_state = graph.invoke(state)
    return final_state["answer"]
