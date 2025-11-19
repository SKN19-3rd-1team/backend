from backend.config import get_settings, resolve_path, expand_paths
from backend.rag.loader import load_courses
from backend.rag.vectorstore import build_vectorstore


def main():
    settings = get_settings()
    json_paths = expand_paths(settings.raw_json)

    docs = []
    for json_path in json_paths:
        docs.extend(load_courses(json_path))

    persist_dir = resolve_path(settings.vectorstore_dir)
    build_vectorstore(docs, persist_directory=persist_dir)
    print(
        f"Vector store built with {len(docs)} documents at '{persist_dir}'."
    )


if __name__ == "__main__":
    main()
