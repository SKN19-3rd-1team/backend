# backend/rag/retriever.py
from typing import Dict, Optional, List
from langchain_core.documents import Document
from .vectorstore import load_vectorstore
from .entity_extractor import normalize_department_name


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


def _relax_filter(metadata_filter: Optional[Dict], relax_field: str) -> Optional[Dict]:
    """
    Remove a specific field from the metadata filter.

    Args:
        metadata_filter: Original filter
        relax_field: Field to remove (e.g., "department", "college")

    Returns:
        Relaxed filter without the specified field, or None if no conditions remain
    """
    if not metadata_filter:
        return None

    # Handle single condition
    if relax_field in metadata_filter:
        return None

    # Handle $and conditions
    if "$and" in metadata_filter:
        remaining_conditions = [
            cond for cond in metadata_filter["$and"]
            if relax_field not in cond
        ]

        if len(remaining_conditions) == 0:
            return None
        elif len(remaining_conditions) == 1:
            return remaining_conditions[0]
        else:
            return {"$and": remaining_conditions}

    return metadata_filter


def _build_fuzzy_department_filter(
    base_filter: Optional[Dict],
    department_base: str
) -> Optional[Dict]:
    """
    Build a fuzzy filter that matches department names with or without suffixes.

    Matches: "컴퓨터공학", "컴퓨터공학부", "컴퓨터공학과"

    Args:
        base_filter: Base filter without department field
        department_base: Normalized department name (e.g., "컴퓨터공학")

    Returns:
        Filter with $in operator for fuzzy department matching
    """
    # Generate possible department name variations
    dept_variations = [
        department_base,           # 컴퓨터공학
        department_base + "부",    # 컴퓨터공학부
        department_base + "과"     # 컴퓨터공학과
    ]

    dept_filter = {"department": {"$in": dept_variations}}

    if base_filter is None:
        return dept_filter

    # Combine with base filter
    if "$and" in base_filter:
        return {"$and": base_filter["$and"] + [dept_filter]}
    else:
        return {"$and": [base_filter, dept_filter]}


def retrieve_with_filter(
    question: str,
    search_k: int = 5,
    metadata_filter: Optional[Dict] = None,
    warn_on_fallback: bool = False
) -> List[Document]:
    """
    Retrieve documents with optional metadata filtering and automatic fallback.

    Fallback strategy (executed in order until results are found):
    1. Try exact filter match
    2. Try fuzzy department matching (컴퓨터공학 matches 컴퓨터공학부, 컴퓨터공학과)
    3. Remove department filter and retry
    4. Remove college filter and retry
    5. Remove all filters (pure semantic search)

    Args:
        question: Query string
        search_k: Number of documents to retrieve
        metadata_filter: Chroma-compatible metadata filter
        warn_on_fallback: If True, print warnings when fallback occurs

    Returns:
        List of retrieved documents
    """
    vs = load_vectorstore()

    # No filter: just do semantic search
    if not metadata_filter:
        return vs.similarity_search(query=question, k=search_k)

    # Step 1: Try exact filter match
    try:
        results = vs.similarity_search(
            query=question,
            k=search_k,
            filter=metadata_filter
        )
        if results:
            print(f"[Retriever] Found {len(results)} results with exact filter")
            return results
    except Exception as e:
        print(f"[Retriever] Exact filter failed: {e}")

    # Step 2: Try fuzzy department matching
    # Extract department from filter and try variations (부, 과)
    department_value = None
    if "department" in metadata_filter:
        department_value = metadata_filter["department"].get("$eq")
    elif "$and" in metadata_filter:
        for cond in metadata_filter["$and"]:
            if "department" in cond:
                department_value = cond["department"].get("$eq")
                break

    if department_value:
        dept_base = normalize_department_name(department_value)
        base_filter = _relax_filter(metadata_filter, "department")
        fuzzy_filter = _build_fuzzy_department_filter(base_filter, dept_base)

        try:
            results = vs.similarity_search(
                query=question,
                k=search_k,
                filter=fuzzy_filter
            )
            if results:
                if warn_on_fallback:
                    print(f"⚠️  [Fallback] Exact filter failed, using fuzzy department matching")
                print(f"[Retriever] Found {len(results)} results with fuzzy department matching")
                return results
        except Exception as e:
            print(f"[Retriever] Fuzzy department matching failed: {e}")

    # Step 3: Remove department filter
    relaxed_filter = _relax_filter(metadata_filter, "department")
    if relaxed_filter:
        try:
            results = vs.similarity_search(
                query=question,
                k=search_k,
                filter=relaxed_filter
            )
            if results:
                if warn_on_fallback:
                    print(f"⚠️  [Fallback] Department filter removed - searching without department constraint")
                print(f"[Retriever] Found {len(results)} results without department filter")
                return results
        except Exception as e:
            print(f"[Retriever] Relaxed filter (no department) failed: {e}")

    # Step 4: Remove college filter
    relaxed_filter2 = _relax_filter(relaxed_filter, "college")
    if relaxed_filter2:
        try:
            results = vs.similarity_search(
                query=question,
                k=search_k,
                filter=relaxed_filter2
            )
            if results:
                if warn_on_fallback:
                    print(f"⚠️  [Fallback] College filter also removed - searching with minimal constraints")
                print(f"[Retriever] Found {len(results)} results without college filter")
                return results
        except Exception as e:
            print(f"[Retriever] Relaxed filter (no college) failed: {e}")

    # Step 5: Final fallback - pure semantic search
    if warn_on_fallback:
        print(f"🚨 [CRITICAL FALLBACK] All filters failed! Using pure semantic search.")
        print(f"   This may return courses from different universities/departments!")
    print("[Retriever] Falling back to pure semantic search (no filters)")
    return vs.similarity_search(query=question, k=search_k)
