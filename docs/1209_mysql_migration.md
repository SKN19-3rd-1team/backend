# MySQL 데이터베이스 마이그레이션 가이드

이 문서는 Unigo 데이터베이스를 로컬 JSON/SQLite에서 통합 MySQL 데이터베이스로 마이그레이션하는 과정과 변경 사항을 설명합니다.

## 🔄 개요

이 마이그레이션은 RAG 시스템과 Django 웹 애플리케이션의 데이터 저장소를 단일 MySQL 데이터베이스로 통합합니다.

-   **변경 전**:
    -   Django: SQLite (`db.sqlite3`)
    -   RAG 시스템: `backend/data/major_detail.json`
-   **변경 후**:
    -   Django: MySQL (`majors` 테이블, Django 인증 테이블)
    -   RAG 시스템: MySQL (`majors` 테이블)

## 🛠 설정

### 환경 변수 (`.env`)

MySQL 연결을 위해 새로운 환경 변수가 필요합니다:

```env
# Database Configuration
MYSQL_HOST=your_host
MYSQL_PORT=3306
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_DB=your_database_name
```

### 모델 (Models)

`backend/db/models.py` 파일에 `Major` 스키마가 정의되어 있습니다.
**중요**: MySQL의 기본 `TEXT` 타입 제한(64KB)을 초과하는 대용량 데이터를 수용하기 위해, 큰 텍스트 필드(기존 JSON)는 `LONGTEXT`로 저장됩니다.

## 🚀 마이그레이션 절차

마이그레이션은 `scripts/migrate_to_mysql.py` 스크립트로 수행됩니다.

### 1. 스키마 생성
스크립트는 깨끗한 스키마 상태를 보장하기 위해 테이블을 자동으로 삭제하고 재생성합니다.

```python
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
```

### 2. 데이터 변환
-   **JSON 직렬화**: 복잡한 필드(리스트, 딕셔너리)는 삽입 전 JSON 문자열로 직렬화됩니다.
-   **통계 처리**: 소스 데이터에서 문자열화된 JSON 리스트로 저장되어 있던 `employment_rate`, `acceptance_rate` 같은 필드는 `extract_stat_value` 헬퍼 함수를 사용하여 단순 `Float` 값으로 추출됩니다.

### 3. 실행
프로젝트 루트에서 마이그레이션 스크립트를 실행합니다:

```bash
python scripts/migrate_to_mysql.py
```

## ✅ 검증

검증 스크립트를 사용하여 마이그레이션 상태를 확인할 수 있습니다:

```bash
python scripts/verify_migration.py
```

이 스크립트는 다음을 확인합니다:
-   총 레코드 수 (304개여야 함).
-   샘플 데이터 무결성.

## 🕸 Django 연동

Django 애플리케이션은 `django.db.backends.mysql`을 통해 동일한 MySQL 데이터베이스를 사용하도록 설정되어 있습니다.

Django 테이블(세션, 인증 등)을 초기화하려면 다음 명령어를 실행하세요:

```bash
python unigo/manage.py migrate
```
