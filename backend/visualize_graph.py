import sys
import os

# Add the parent directory to sys.path to allow importing backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.graph.graph_builder import build_graph

def visualize_graph(mode: str):
    print(f"--- Visualizing '{mode}' Graph ---")
    try:
        app = build_graph(mode)
        # Generate Mermaid diagram
        mermaid_png = app.get_graph().draw_mermaid_png()
        mermaid_text = app.get_graph().draw_mermaid()
        
        print(mermaid_text)
        print("\n" + "="*50 + "\n")
        
    except Exception as e:
        print(f"Error visualizing {mode} graph: {e}")

if __name__ == "__main__":
    visualize_graph("react")
    visualize_graph("major")
