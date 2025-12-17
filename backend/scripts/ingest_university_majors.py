import sys
import os
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from backend.rag.loader import load_major_detail, build_university_major_docs
from backend.rag.vectorstore import index_university_majors


def main():
    print("🚀 Starting University-Major Ingestion (Full)...")

    # 1. Load Data
    print("📥 Loading major details...")
    records = load_major_detail()
    print(f"✅ Loaded {len(records)} major records.")

    # 2. Build Documents
    print("🔨 Building university-major documents...")
    all_univ_docs = []
    for record in records:
        univ_docs = build_university_major_docs(record)
        all_univ_docs.extend(univ_docs)

    print(f"✅ Generated {len(all_univ_docs)} university-major documents.")

    # 3. Indexing
    if all_univ_docs:
        print(f"📤 Indexing to Pinecone (Namespace: university_majors)...")
        try:
            count = index_university_majors(all_univ_docs)
            print(f"✨ Successfully indexed {count} documents.")
        except Exception as e:
            print(f"❌ Indexing Failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("⚠️ No documents to index.")


if __name__ == "__main__":
    main()
