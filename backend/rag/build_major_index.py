"""
Pinecone 전공 인덱스를 처음부터 다시 구성할 때 사용하는 유틸리티 스크립트.

로컬 JSON 데이터를 MajorDoc으로 변환한 뒤 Pinecone 네임스페이스를 비우고
모든 문서를 재업서트하는 일련의 과정을 자동화합니다.
"""

from __future__ import annotations

from backend.rag.loader import load_major_detail, build_all_major_docs
from backend.rag.vectorstore import (
    clear_major_index,
    index_major_docs,
    get_major_vectorstore,
)


def rebuild_major_index() -> None:
    # 로컬 JSON → MajorDoc 생성 → Pinecone 업서트 순서로 인덱스를 재구축
    records = load_major_detail()
    docs = build_all_major_docs(records)
    print(f"Loaded {len(records)} majors and prepared {len(docs)} documents.")

    # 인덱스가 존재하지 않는 환경에서도 안전하게 초기화되도록 벡터스토어를 먼저 준비
    get_major_vectorstore()
    clear_major_index()
    print("Cleared existing Pinecone index namespace.")

    indexed = index_major_docs(docs)
    print(f"Indexed {indexed} documents into Pinecone.")


if __name__ == "__main__":
    rebuild_major_index()
