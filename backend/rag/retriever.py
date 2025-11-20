# backend/rag/retriever.py
from typing import Dict, Optional, List
from langchain_core.documents import Document
from .vectorstore import load_vectorstore


def get_retriever(search_k: int = 5, metadata_filter: Optional[Dict] = None):
    """
    Get a retriever with optional metadata filtering.

    Args:
        search_k: Number of documents to retrieve
        metadata_filter: Chroma-compatible metadata filter dictionary

    Returns:
        Configured retriever
    """
    vs = load_vectorstore()

    search_kwargs = {"k": search_k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    return vs.as_retriever(search_kwargs=search_kwargs)


def retrieve_with_filter(
    question: str,
    search_k: int = 5,
    metadata_filter: Optional[Dict] = None
) -> List[Document]:
    """
    Retrieve documents with optional metadata filtering.

    Note: Uses similarity_search directly instead of as_retriever to ensure
    filtering is applied before semantic search (not after).

    Args:
        question: Query string
        search_k: Number of documents to retrieve
        metadata_filter: Chroma-compatible metadata filter

    Returns:
        List of retrieved documents
    """
    vs = load_vectorstore()

    # Use similarity_search directly to apply filter BEFORE retrieval
    # as_retriever applies filter AFTER retrieval which can result in 0 results
    if metadata_filter:
        return vs.similarity_search(
            query=question,
            k=search_k,
            filter=metadata_filter
        )
    else:
        return vs.similarity_search(
            query=question,
            k=search_k
        )
