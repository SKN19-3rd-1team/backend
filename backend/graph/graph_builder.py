# backend/graph/graph_builder.py
from langgraph.graph import StateGraph
from .state import MentorState
from .nodes import retrieve_node, answer_node

def build_graph():
    graph = StateGraph(MentorState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "__end__")

    app = graph.compile()
    return app
