"""
LangGraph 아키텍처를 시각화하는 스크립트

사용법:
    python visualize_graph.py --mode react       # ReAct 그래프 시각화
    python visualize_graph.py --mode structured  # Structured 그래프 시각화
    python visualize_graph.py --mode both        # 둘 다 시각화 (기본값)

출력:
    - react_graph.png: ReAct 패턴 그래프 이미지
    - structured_graph.png: Structured 패턴 그래프 이미지
"""

import argparse
from backend.graph.graph_builder import build_graph


def visualize_graph(mode: str, output_file: str):
    """
    LangGraph를 PNG 이미지로 시각화합니다.

    Args:
        mode: "react" 또는 "structured"
        output_file: 출력할 PNG 파일 경로
    """
    print(f"Building {mode} graph...")
    graph = build_graph(mode=mode)

    try:
        # LangGraph의 get_graph() 메서드를 사용하여 Mermaid 형식으로 출력
        # 또는 draw_mermaid_png()를 사용하여 PNG로 저장
        png_data = graph.get_graph().draw_mermaid_png()

        with open(output_file, "wb") as f:
            f.write(png_data)

        print(f"[SUCCESS] Graph visualization saved to: {output_file}")

    except Exception as e:
        print(f"[ERROR] Failed to generate PNG: {e}")
        print(f"[INFO] Trying ASCII representation instead...\n")

        # PNG 생성 실패 시 ASCII로 출력
        try:
            print(graph.get_graph().draw_ascii())
        except Exception as ascii_err:
            print(f"[ERROR] ASCII visualization also failed: {ascii_err}")
            print(f"[INFO] Printing Mermaid code instead:\n")
            print(graph.get_graph().draw_mermaid())


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LangGraph architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["react", "structured", "both"],
        default="both",
        help="Which graph to visualize (default: both)"
    )

    args = parser.parse_args()

    if args.mode == "both":
        visualize_graph("react", "react_graph.png")
        print()
        visualize_graph("structured", "structured_graph.png")
    else:
        output_file = f"{args.mode}_graph.png"
        visualize_graph(args.mode, output_file)


if __name__ == "__main__":
    main()
