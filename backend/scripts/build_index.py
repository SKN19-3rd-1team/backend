from pathlib import Path
from backend.rag.loader import load_courses
from backend.rag.vectorstore import build_vectorstore

if __name__ == "__main__":
    json_path = Path(__file__).resolve().parents[2] / "backend" / "data" / "raw" / "konkuk_all_건국대.json"
    docs = load_courses(json_path)
    build_vectorstore(docs)
    print("✅ Vector store built.")
