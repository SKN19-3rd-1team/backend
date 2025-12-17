# Backend Refactoring: Remove JSON Dependencies [2025-12-12]

## 개요
기존 백엔드 코드(`backend/rag`)에서 `backend/data` 폴더 내의 JSON 파일들(`major_detail.json`, `major_categories.json`, `university_data_cleaned.json`)을 직접 참조하던 로직을 제거하고, MySQL 데이터베이스를 유일한 데이터 소스로 사용하도록 리팩토링했습니다.

## 변경 사항

### 1. `backend/rag/university_lookup.py`
- **변경 전**: `university_data_cleaned.json` 파일을 로드하여 메모리에 캐싱 후 대학 정보를 조회.
- **변경 후**: `University` 테이블을 직접 쿼리하여 정보 조회.
    - `lookup_university_url`: 정확한 매칭, [본교] 접미사 매칭, 부분 일치 검색 순으로 DB 조회.
    - `search_universities`: `LIKE` 검색을 통해 매칭되는 모든 대학 반환.

### 2. `backend/rag/tools.py`
- **변경 전**: `major_categories.json` 파일을 로드하여 전공 카테고리 정보 사용.
- **변경 후**: `MajorCategory` 테이블에서 `major_names` (JSON 문자열)을 파싱하여 카테고리 정보 로드.
- `MAJOR_CATEGORIES_FILE` 상수 제거.

### 3. `backend/rag/loader.py`
- **변경 전**: DB 연결 실패 시 또는 설정에 따라 `major_detail.json` 파일을 로드하는 Fallback 로직 존재.
- **변경 후**: JSON Fallback 로직 완전 제거. 오직 DB(`Major` 테이블)에서만 데이터를 로드함.

### 4. `backend/config.py`
- 불필요해진 JSON 파일 경로 설정(`major_detail_path`, `raw_json`) 제거.

## 검증
- `backend/scripts/verify_refactor.py` 스크립트를 통해 다음 항목 검증 완료:
    - 전공 카테고리 로드 (DB)
    - 전공 상세 정보 로드 (DB)
    - 대학 URL 조회 (DB)
    - 대학 검색 (DB)

## 주의사항
- **DB존재 필수**: 이제 애플리케이션 실행을 위해 MySQL 데이터베이스에 데이터가 반드시 시딩되어 있어야 합니다. (`backend/data`의 JSON 파일이 없어도 동작하지만, DB가 비어있으면 동작하지 않음)
