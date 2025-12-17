# 리팩토링 로그 (Refactoring Log)

## 2025-12-15 리팩토링

### 1. 데이터베이스 정리 (Database Cleanup)
- **삭제**: `unigo/db.sqlite3`
  - 사유: 프로젝트 데이터베이스가 MySQL로 마이그레이션됨 (`MYSQL_HOST` 환경변수 사용).

### 2. 스크립트 재구성 (Script Reorganization)
- **이동**: 테스트 및 디버그용 스크립트를 `backend/scripts/`에서 `backend/scripts/test/`로 이동했습니다.
  - `manual_feedback_test.py`
  - `test_major_similarity.py`
  - `test_search.py`
  - `verify_refactor.py`
  - `verify_tool_output.py`
  - `debug_loader.py`
- **유지**: 데이터 수집(Ingestion) 관련 스크립트(`ingest_major_categories.py`, `ingest_university_majors.py`)는 `backend/scripts/`에 그대로 유지했습니다.

### 3. 코드 품질 개선 (Code Quality Improvements)
- **문서화 주석(Docstrings)**: 핵심 모듈에 상세한 한글 Docstring을 추가했습니다.
  - `backend/main.py`: `get_graph` (싱글톤/캐싱), `run_mentor`, `run_major_recommendation`
  - `backend/graph/graph_builder.py`: `build_major_graph` (실행 흐름 설명)
  - `unigo_app/views.py`: `chat_api`, `auth_*`, `onboarding_api` (인자/반환값 설명)
- **코드 정리**: 사용하지 않는 import (`os`, `json`) 및 변수(`target_user`, `email` 등 views 내 미사용 변수)를 제거했습니다.
- **Import 경로 수정**: 이동된 테스트 스크립트들이 프로젝트 루트를 올바르게 찾을 수 있도록 `sys.path` 설정을 수정했습니다.

### 4. 문서화 (Documentation)
- **업데이트**: `README.md`
  - Mermaid 다이어그램을 포함한 "LangGraph Workflow" 섹션을 추가하여 에이전트 구조와 데이터 흐름을 시각화했습니다.
