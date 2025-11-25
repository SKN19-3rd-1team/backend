# 코드 모듈 & 주요 함수 개요

테스트 스크립트(`test_*.py`)와 시각화 스크립트(`visualize_*.py`)를 제외한 코드 파일에서,
어떤 함수를 제공하고 어떻게 호출하는지 한눈에 볼 수 있도록 정리했습니다.

## backend

- `backend/main.py`
  - `get_graph(mode="react" | "structured")`: ReAct/Structured LangGraph 그래프 인스턴스를 빌드·캐시합니다.
  - `run_mentor(question, interests=None, mode="react", chat_history=None)`: 멘토 봇 메인 엔트리 포인트로,
    Python REPL에서 `from backend.main import run_mentor` 후 호출하거나 Streamlit 앱이 내부에서 사용합니다.

- `backend/config.py`
  - `get_settings()`: `.env`를 읽어 `Settings` 데이터클래스를 생성합니다.
  - `get_llm()`: `LLM_PROVIDER`/`MODEL_NAME`에 맞는 LangChain `ChatModel`을 생성합니다.
  - `resolve_path(path_str)`: 프로젝트 루트 기준 상대/절대 경로를 `Path`로 변환합니다.
  - `expand_paths(pattern)`: 글로브 패턴으로 매칭되는 파일 경로 리스트를 반환합니다.

- `backend/graph/graph_builder.py`
  - `build_graph(mode)`: 주어진 모드에 따라 ReAct/Structured 그래프를 빌드합니다.
  - `build_react_graph()`: ReAct 스타일 에이전트 그래프를 구성합니다.
  - `build_structured_graph()`: 단계형 RAG(검색→선택→답변) 그래프를 구성합니다.

- `backend/graph/nodes.py`
  - `extract_departments_from_tool_results(messages)`: `list_departments` Tool 결과에서 학과명 집합을 추출합니다.
  - `validate_and_fix_department_names(response, valid_departments)`: (완화 모드) 학과명을 툴 결과에 맞게 보정합니다.
  - `strict_validate_and_fix_department_names(response, valid_departments)`: (엄격 모드) 학과명이 정확히 일치하지 않으면
    가능한 경우 교정하고, 그렇지 않으면 학과명 마크업에서 제외합니다.
  - `retrieve_node(state)`: Structured 모드 1단계 – 벡터스토어에서 과목 후보를 검색합니다.
  - `select_node(state)`: Structured 모드 2단계 – LLM이 과목 `id` 목록을 JSON으로 선택합니다.
  - `answer_node(state)`: Structured 모드 3단계 – 선택된 과목들만 사용해 자연어 답변을 생성합니다.
  - `agent_node(state)`: ReAct 모드 에이전트 – LLM이 Tool을 호출하고, 최종 답변을 생성합니다.
  - `should_continue(state)`: 마지막 LLM 메시지에 `tool_calls`가 있는지에 따라 다음 노드를 `tools`/`end`로 분기합니다.

- `backend/rag/loader.py`
  - `load_courses(json_path)`: 원본 강의 JSON 파일을 LangChain `Document` 리스트로 변환합니다.

- `backend/rag/embeddings.py`
  - `get_embeddings()`: `.env` 설정에 맞는 임베딩 모델을 초기화하고 캐시합니다.

- `backend/rag/vectorstore.py`
  - `_resolve_persist_dir(persist_directory)`: Chroma 벡터스토어가 저장될 디렉터리를 계산합니다.
  - `build_vectorstore(raw_json_pattern=None, persist_directory=None)`: 원본 JSON에서 코스 임베딩을 생성하고
    Chroma 벡터스토어를 빌드합니다. 터미널에서는 `python -m backend.rag.vectorstore`로 실행합니다.
  - `load_vectorstore(persist_directory=None)`: 이미 빌드된 Chroma 벡터스토어를 메모리로 로드합니다.

- `backend/rag/retriever.py`
  - `get_retriever(search_k=5, metadata_filter=None)`: LangChain `VectorStoreRetriever`를 생성합니다.
  - `_build_fuzzy_department_filter(...)`: 학과명 유사도 기반 필터를 구성하는 내부 유틸입니다.
  - `_relax_filter(metadata_filter, relax_field)`: 필터가 너무 엄격할 때 특정 필드를 완화하는 내부 유틸입니다.
  - `retrieve_with_filter(question, search_k, metadata_filter)`: 질문과 메타데이터 필터를 기준으로 코스를 검색합니다.

- `backend/rag/entity_extractor.py`
  - `_load_university_mapping()`, `_load_department_mapping()`: 대학/학과 매핑 JSON을 로드하는 내부 함수입니다.
  - `normalize_university_name(university_query)`: 다양한 표현의 대학명을 표준 대학명으로 정규화합니다.
  - `get_all_department_variants(department_query)`: 한 학과에 대한 여러 약어/변형 표현 후보를 생성합니다.
  - `normalize_department_name(department_query)`: 학과명을 표준 학과명으로 정규화하려고 시도합니다.
  - `extract_filters(question)`: 질문에서 대학/단과대/학과/학년/학기 정보를 추출해 필터 딕셔너리를 만듭니다.
  - `build_chroma_filter(filters)`: 추출된 필터를 Chroma 메타데이터 필터 형식으로 변환합니다.

- `backend/rag/tools.py` (LangChain Tools)
  - `_get_tool_usage_guide()`: 검색 도구 사용법을 한국어 가이드 문자열로 반환합니다.
  - `_load_department_embeddings()`: 벡터스토어에서 학과명 목록과 임베딩 행렬을 로드·캐시합니다.
  - `_expand_category_query(query)`: 카테고리/키워드 기반 학과 쿼리를 토큰 목록과 임베딩용 텍스트로 확장합니다.
  - `retrieve_courses(...)`: 과목 검색 Tool – 질문/필터를 받아 관련 과목 리스트를 반환합니다.
  - `get_course_detail(course_id, courses_context)`: 검색된 과목 리스트에서 특정 ID의 상세 정보를 반환합니다.
  - `list_departments(query, top_k=10)`: 문자열/임베딩 하이브리드 검색으로 관련 학과명을 찾고, 포맷된 목록을 반환합니다.
  - `get_universities_by_department(department_name)`: 주어진 학과를 개설한 대학/단과대 정보를 조회합니다.
  - `recommend_curriculum(university, department, ...)`: 관심사·학년 범위 등을 기반으로 학기별 과목 커리큘럼을 추천합니다.
  - `match_department_name(department_query)`: 줄임말/오타가 섞인 학과명을 표준 학과명(+대학명)으로 정규화합니다.
  - `get_search_help()`: 어떤 Tool을 어떻게 조합해서 써야 할지에 대한 도움말을 반환합니다.

- `backend/scripts/build_index.py`
  - `main()`: CLI에서 호출되는 인덱스 빌드 엔트리로, 내부에서 `build_vectorstore`를 사용합니다.

## frontend

- `frontend/app.py`
  - `format_interests_from_selection()`: 사용자가 체크박스로 선택한 관심사를 LLM 입력용 문자열로 변환합니다.
  - `format_interests_for_llm()`: 자유 입력/선택 정보를 합쳐 LLM에 넘길 관심사 텍스트를 만듭니다.
  - `is_curriculum_query(text)`: 질문이 커리큘럼 추천인지 여부를 간단히 판별합니다.
  - `render_format_options_inline(original_question)`: Streamlit 상단에 질문 포맷 옵션 버튼을 렌더링합니다.
  - `handle_button_click(selection)`: 포맷 옵션 버튼 클릭 시 세션 상태를 갱신합니다.
  - 전체 모듈은 `streamlit run frontend/app.py`로 실행되며, 위 함수들을 조합해 챗봇 UI를 구성합니다.

