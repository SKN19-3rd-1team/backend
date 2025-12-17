
# seed_majors.py 데이터 미반영 문제 분석 및 해결 가이드

## 1. 개요
사용자가 `backend/db/seed_majors.py`를 실행하여 ChartData에서 취업률(`employment_rate`)과 입학률(`acceptance_rate`)을 추출하는 로직을 추가했으나, DB에 반영되지 않는 현상이 발생함.

## 2. seed_majors.py 동작 방식
- **Upsert (Insert or Update)**: `seed_majors.py`는 `major_id`를 기준으로 데이터가 이미 존재하는지 확인합니다.
  - **존재할 경우**: 기존 레코드의 필드를 새로운 값으로 **Update(덮어쓰기)** 합니다.
  - **존재하지 않을 경우**: 새로운 레코드를 **Insert** 합니다.
- 따라서, DB가 이미 생성되어 있더라도 스크립트를 재실행하면 변경된 로직(필드 추출 등)이 반영되어야 정상입니다.

## 3. 데이터가 반영되지 않는 원인 분석
논리적으로 코드는 정상 동작해야 하지만, 반영이 안 된다면 다음 두 가지 원인이 유력합니다.

### A. 데이터베이스 스키마 불일치 (가장 유력)
- Python 코드(`models.py`)에는 `employment_rate`, `acceptance_rate` 컬럼을 정의했더라도, **실제 운영 중인 데이터베이스(MySQL) 테이블에 해당 컬럼이 생성되지 않았을 가능성**이 높습니다.
- 일반적으로 컬럼이 없으면 스크립트 실행 시 `OperationalError: (1054, "Unknown column ...")` 에러가 발생합니다.
- 만약 에러를 못 보셨거나 무시했다면, 이것이 원인입니다.

### B. 데이터 추출 실패
- `major_detail.json`의 구조가 예상과 달라 `extract_chart_stats` 함수가 데이터를 `None`으로 반환하는 경우입니다.
- 하지만 디버깅 결과(샘플 데이터 확인), 정상적으로 데이터를 추출하고 있음이 확인되었습니다.

## 4. 해결 방법

### 1단계: DB 컬럼 확인 및 추가
DB 클라이언트를 통해 `majors` 테이블에 `employment_rate`, `acceptance_rate` 컬럼이 있는지 확인하세요. 없다면 다음 SQL을 실행하여 추가해야 합니다.

```sql
ALTER TABLE majors ADD COLUMN employment_rate FLOAT COMMENT '취업률';
ALTER TABLE majors ADD COLUMN acceptance_rate FLOAT COMMENT '입학률';
```

### 2단계: 시드 스크립트 재실행
컬럼이 준비되었다면 스크립트를 다시 실행하여 데이터를 업데이트합니다.

```bash
python backend/db/seed_majors.py
```

### 3단계: 확인
정상적으로 업데이트되었는지 확인합니다.
