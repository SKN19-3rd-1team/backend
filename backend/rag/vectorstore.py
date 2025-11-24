"""
벡터 데이터베이스 (Vector Store) 관리 모듈

Chroma DB를 사용하여 과목 정보를 벡터로 저장하고 로드하는 기능을 제공합니다.

** Chroma DB란? **
- 오픈소스 벡터 데이터베이스
- 텍스트를 벡터(임베딩)로 변환하여 저장
- 유사도 기반 검색 (Similarity Search) 지원
- SQLite 기반으로 로컬 파일 시스템에 저장

** 주요 기능 **
1. build_vectorstore(): JSON 파일에서 과목 데이터를 읽어 벡터 DB 생성
2. load_vectorstore(): 저장된 벡터 DB를 디스크에서 로드
"""
# backend/rag/vectorstore.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import threading

from langchain_core.documents import Document
from langchain_chroma import Chroma

from backend.config import get_settings, resolve_path, expand_paths
from .embeddings import get_embeddings

# 벡터 스토어 싱글톤 캐시
# 여러 툴이 동시에 호출될 때 Chroma 인스턴스를 중복 생성하지 않도록 전역 변수에 캐싱
# ChromaDB 1.3.4의 재초기화 버그를 방지하기 위해 한 번만 로드하고 재사용
_VECTORSTORE_CACHE = None
_VECTORSTORE_LOCK = threading.Lock()


def _resolve_persist_dir(persist_directory: Path | str | None) -> Path:
    """
    Vector DB 저장 디렉토리 경로 해석 및 생성

    .env 파일의 VECTORSTORE_DIR 설정을 사용하거나, 직접 경로를 지정할 수 있습니다.
    디렉토리가 존재하지 않으면 자동으로 생성합니다.

    Args:
        persist_directory: Vector DB 저장 경로 (None이면 .env의 VECTORSTORE_DIR 사용)

    Returns:
        Path: 절대 경로로 변환된 Vector DB 디렉토리
    """
    settings = get_settings()

    # persist_directory가 None이면 설정에서 가져옴
    directory_str = (
        settings.vectorstore_dir
        if persist_directory is None
        else str(persist_directory)
    )

    # 상대 경로를 절대 경로로 변환
    directory = resolve_path(directory_str)

    # 디렉토리가 없으면 생성 (부모 디렉토리도 함께 생성)
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def build_vectorstore(
    docs: Iterable[Document],
    persist_directory: Path | str | None = None,
):
    """
    과목 Document 리스트로부터 Chroma Vector DB 생성

    ** 중요: 이 함수는 Vector DB를 처음 생성하거나 재생성할 때만 사용합니다 **
    - 실행 시간이 오래 걸림 (문서 수백 개 이상일 경우 수 분 소요)
    - 모든 과목을 임베딩으로 변환하고 Chroma DB에 저장
    - 디스크에 영구 저장되므로 재실행 불필요

    ** 실행 방법 **
    ```bash
    # 터미널에서 직접 실행
    python -m backend.rag.vectorstore
    ```

    Args:
        docs: LangChain Document 리스트 (loader.py에서 생성됨)
        persist_directory: Vector DB 저장 경로 (None이면 .env의 VECTORSTORE_DIR 사용)

    Returns:
        Chroma: 생성된 Vector Store 인스턴스
    """
    # 저장 디렉토리 경로 해석
    persist_directory = _resolve_persist_dir(persist_directory)

    # 임베딩 모델 로드 (OpenAI 또는 HuggingFace)
    embeddings = get_embeddings()

    # Chroma DB 생성
    # - documents의 page_content를 임베딩으로 변환
    # - 임베딩과 metadata를 함께 Chroma DB에 저장
    # - persist_directory에 영구 저장 (SQLite 파일 생성)
    vs = Chroma.from_documents(
        documents=list(docs),          # Document 리스트를 리스트로 변환 (iterator 지원 안 함)
        embedding=embeddings,           # 임베딩 모델
        persist_directory=str(persist_directory),  # 저장 경로
    )
    return vs


def load_vectorstore(persist_directory: Path | str | None = None):
    """
    디스크에 저장된 Chroma Vector DB 로드 (싱글톤 패턴)

    build_vectorstore()로 생성한 Vector DB를 메모리에 로드합니다.
    실제 검색 시에는 이 함수를 사용합니다 (빠름).

    한 번 로드된 Vector Store는 전역 캐시에 저장되어 재사용됩니다.
    이는 ChromaDB 1.3.4의 재초기화 버그를 방지하고 성능을 향상시킵니다.

    ** 주의사항 **
    - build_vectorstore()와 같은 임베딩 모델을 사용해야 함
    - 임베딩 모델이 다르면 벡터 차원 불일치로 오류 발생
    - .env의 EMBEDDING_PROVIDER와 EMBEDDING_MODEL_NAME 확인 필수

    Args:
        persist_directory: Vector DB 저장 경로 (None이면 .env의 VECTORSTORE_DIR 사용)

    Returns:
        Chroma: 로드된 Vector Store 인스턴스 (검색 가능 상태)
    """
    global _VECTORSTORE_CACHE

    # 이미 로드된 인스턴스가 있으면 재사용 (싱글톤 패턴)
    # 여러 툴이 호출되어도 Vector Store는 한 번만 로딩됨
    # Lock을 사용하여 동시 접근 시 Chroma 인스턴스 생성을 직렬화
    with _VECTORSTORE_LOCK:
        if _VECTORSTORE_CACHE is not None:
            return _VECTORSTORE_CACHE

        # 저장 디렉토리 경로 해석
        persist_directory = _resolve_persist_dir(persist_directory)

        # 임베딩 모델 로드 (Vector DB 생성 시 사용한 것과 동일해야 함)
        embeddings = get_embeddings()

        # 디스크에서 Chroma DB 로드
        # - persist_directory의 SQLite 파일과 벡터 데이터 읽기
        # - embedding_function으로 새 쿼리를 임베딩하여 검색
        _VECTORSTORE_CACHE = Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )

        return _VECTORSTORE_CACHE


if __name__ == "__main__":
    """
    Vector DB 빌드 스크립트

    이 파일을 직접 실행하면 JSON 파일에서 과목 데이터를 읽어 Vector DB를 생성합니다.

    ** 실행 방법 **
    ```bash
    python -m backend.rag.vectorstore
    ```

    ** 동작 과정 **
    1. .env에서 RAW_JSON 경로 패턴 읽기 (예: backend/data/*.json)
    2. 패턴에 매칭되는 모든 JSON 파일 로드
    3. 각 파일의 과목 정보를 Document로 변환
    4. 모든 Document를 Chroma DB에 저장 (VECTORSTORE_DIR)
    """
    from backend.rag.loader import load_courses

    # 설정 로드
    settings = get_settings()

    # RAW_JSON 패턴에 매칭되는 모든 파일 찾기
    # 예: "backend/data/*.json" → [backend/data/file1.json, backend/data/file2.json, ...]
    json_files = expand_paths(settings.raw_json)

    # 모든 JSON 파일에서 과목 데이터 로드
    docs: list[Document] = []
    for json_path in json_files:
        print(f"Loading courses from {json_path}...")
        docs.extend(load_courses(json_path))

    # Vector DB 저장 경로
    target_dir = _resolve_persist_dir(None)

    # 빌드 시작
    print(
        f"\nBuilding vector store with {len(docs)} documents "
        f"at '{target_dir}'..."
    )
    print("This may take a few minutes depending on the number of documents and embedding model speed...\n")

    build_vectorstore(docs, persist_directory=target_dir)

    print("\nVector store build complete!")
    print(f"Saved to: {target_dir}")
