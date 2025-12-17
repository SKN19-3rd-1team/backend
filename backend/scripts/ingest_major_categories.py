import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from backend.db.connection import get_db
from backend.db.models import Major
from backend.rag.vectorstore import get_major_category_vectorstore
from backend.rag.embeddings import get_embeddings


def ingest_major_categories():
    print("🚀 Starting Major Category Ingestion...")

    # 1. DB에서 모든 표준 학과명(major_name) 가져오기
    db = next(get_db())
    try:
        # DISTINCT major_name 조회
        majors = db.query(Major.major_name).distinct().all()
        major_names = [m[0] for m in majors if m[0]]
        print(f"✅ Found {len(major_names)} unique major categories in DB.")

        if not major_names:
            print("⚠️ No majors found. Exiting.")
            return

        # 2. VectorStore 준비
        vectorstore = get_major_category_vectorstore()

        import hashlib

        # 3. 데이터 준비 (Text itself is the major name)
        texts = major_names
        metadatas = [
            {"major_name": name, "doc_type": "category"} for name in major_names
        ]
        # Pinecone IDs must be ASCII (safe). Use MD5 hash of the name.
        ids = [hashlib.md5(name.encode("utf-8")).hexdigest() for name in major_names]

        # 4. 업로드 (배치 처리 권장하지만 300개라 한 번에 가능)
        print(
            "Wait... Embedding and Uploading to Pinecone (namespace='major_categories')..."
        )
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        print(f"🎉 Successfully indexed {len(texts)} major categories.")

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    ingest_major_categories()
