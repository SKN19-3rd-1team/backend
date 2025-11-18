# backend/rag/vectorstore.py
from pathlib import Path
from langchain_community.vectorstores import Chroma
from .embeddings import get_embeddings

DEFAULT_DB_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_db"

def build_vectorstore(docs, persist_directory: Path | None = None):
    if persist_directory is None:
        persist_directory = DEFAULT_DB_DIR

    embeddings = get_embeddings()
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )
    vs.persist()
    return vs

def load_vectorstore(persist_directory: Path | None = None):
    if persist_directory is None:
        persist_directory = DEFAULT_DB_DIR

    embeddings = get_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
