# 데이터베이스 마이그레이션 로그

## 2025-12-10: 전공 데이터 적재 스크립트 생성 (`backend/db/seed_majors.py`)

### 개요
`backend/data/major_detail.json` 파일의 원본 데이터를 MySQL 데이터베이스의 `majors` 테이블로 적재하기 위한 파이썬 스크립트를 작성했습니다.

### 주요 기능 및 전처리 로직
1.  **중첩 구조 해제**: JSON의 `dataSearch.content` 내부 데이터를 추출하여 평탄화했습니다.
2.  **ID 생성**: `major_name`을 기반으로 `uuid5(NAMESPACE_DNS)`를 사용하여 결정론적(Deterministic) 고유 ID를 생성했습니다. (재실행 시 중복 방지)
3.  **데이터 타입 변환**:
    - `salary`: 문자열 -> 실수(Float) 변환
    - 리스트/딕셔너리 필드: `json.dumps`를 통해 JSON 문자열로 직렬화하여 적재
4.  **통계 데이터 추출 (Chart Data)**:
    - 원본의 `chartData` 필드에서 **취업률(`employment_rate`)**과 **입학률(`acceptance_rate`, 경쟁률 역산)**을 추출하여 별도 컬럼에 저장했습니다.

### 실행 방법
프로젝트 루트 디렉토리에서 아래 명령어를 실행합니다.
```bash
python -m backend.db.seed_majors
```

### 참고 사항
- 스크립트는 **Upsert** 방식으로 동작합니다. 이미 존재하는 `major_id`가 있으면 정보를 업데이트하고, 없으면 새로 생성합니다.
- `qualifications` 필드는 원본 데이터(문자열)를 그대로 유지했습니다.

## 2025-12-10: 전공 카테고리 및 대학 정보 적재 (`backend/db/seed_categories.py`, `backend/db/seed_universities.py`)

### 개요
`backend/data/major_categories.json` (전공 카테고리) 및 `backend/data/university_data_cleaned.json` (대학 정보) 파일을 MySQL 데이터베이스로 적재하기 위한 스크립트를 추가했습니다.

### 주요 기능
1.  **전공 카테고리 적재 (`seed_categories.py`)**:
    - `MajorCategory` 테이블에 적재
    - 카테고리별 전공 목록을 JSON으로 직렬화하여 `major_names` 컬럼(LONGTEXT)에 저장
2.  **대학 정보 적재 (`seed_universities.py`)**:
    - `University` 테이블에 적재
    - 대학 코드 및 URL 정보 저장

## 2025-12-10: 통합 적재 스크립트 (`backend/db/seed_all.py`)

### 개요
모든 데이터(전공, 카테고리, 대학)를 한번에 적재할 수 있는 마스터 스크립트입니다.

### 실행 방법
```bash
python -m backend.db.seed_all
```

### 기존 스크립트와의 관계
- 기존 `scripts/migrate_to_mysql.py`는 `backend/db/seed_majors.py` 등으로 기능이 분화 및 개선되어 **삭제 권장(Deprecated)** 상태입니다.
