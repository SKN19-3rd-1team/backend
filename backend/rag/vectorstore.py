# backend/rag/vectorstore.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from backend.config import get_settings, resolve_path, expand_paths
from .embeddings import get_embeddings


def _resolve_persist_dir(persist_directory: Path | str | None) -> Path:
    settings = get_settings()
    directory_str = (
        settings.vectorstore_dir
        if persist_directory is None
        else str(persist_directory)
    )
    directory = resolve_path(directory_str)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_vectorstore(
    docs: Iterable[Document],
    persist_directory: Path | str | None = None,
):
    """Create a Chroma vector store from the provided documents."""
    persist_directory = _resolve_persist_dir(persist_directory)

    embeddings = get_embeddings()
    vs = Chroma.from_documents(
        documents=list(docs),
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )
    vs.persist()
    return vs


def load_vectorstore(persist_directory: Path | str | None = None):
    """Load a Chroma vector store from disk."""
    persist_directory = _resolve_persist_dir(persist_directory)

    embeddings = get_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


if __name__ == "__main__":
    from backend.rag.loader import load_courses

    settings = get_settings()
    json_files = expand_paths(settings.raw_json)

    docs: list[Document] = []
    for json_path in json_files:
        docs.extend(load_courses(json_path))

    target_dir = _resolve_persist_dir(None)
    print(
        f"Building vector store with {len(docs)} documents "
        f"at '{target_dir}'..."
    )
    build_vectorstore(docs, persist_directory=target_dir)
    print("Vector store build complete.")
