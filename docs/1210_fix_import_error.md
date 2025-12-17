# 전공 추천 로직의 ImportError 및 TypeError 수정

## 작업 내용
### 1. ImportError 해결
- `backend/graph/nodes.py`에서 발생하던 `ImportError: cannot import name '_ensure_major_records'` 오류를 수정했습니다.
- **원인**: `_ensure_major_records` 함수가 `backend/rag/tools.py`에 존재하지 않는데 호출하고 있었습니다.
- **해결**: 해당 import 및 호출 코드를 제거했습니다.

### 2. TypeError 해결
- `backend/rag/tools.py`에서 DB 데이터를 `MajorRecord` 객체로 변환할 때 발생하던 `TypeError: MajorRecord.__init__() missing 1 required positional argument: 'cluster'` 오류를 수정했습니다.
- **원인**: `MajorRecord` 데이터클래스 정의에는 `cluster` 필드가 필수 인자(`cluster: Optional[str]`, 3번째 위치)로 되어 있습니다. 그러나 `tools.py`의 `_convert_db_model_to_record` 함수에서 객체 생성 시 이 인자를 누락했습니다.
- **해결**: `MajorRecord` 생성자에 `cluster=None` 인자를 추가했습니다.

## 추가 검토 사항
- `MajorRecord`가 사용되는 다른 파일(`backend/rag/loader.py`)을 검토했습니다.
- MySQL 데이터 로드 로직과 JSON 파일 로드 로직 모두 `cluster=None`이 이미 정상적으로 포함되어 있어 추가 수정이 필요하지 않음을 확인했습니다.

## 결과
- 온보딩 API (`/api/onboarding`) 호출 시 발생하던 500 에러(ImportError, TypeError)가 모두 해결되어 정상적인 전공 추천 기능이 동작합니다.
