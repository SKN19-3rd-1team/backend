# 데이터베이스 스키마 및 데이터 호환성 검토 보고서

## 1. 개요
본 문서는 `backend/data/major_detail.json`의 원본 데이터를 `backend/db/models.py`에 정의된 `Major` 테이블에 적재하기 위한 적합성을 검토하고, 필요한 전처리 과정을 제안합니다.

## 2. 구조적 불일치 (Structural Mismatches)

### 2.1 중첩 구조 (Nested Structure)
- **현상**: 원본 JSON은 `[ { "dataSearch": { "content": [ {실제데이터} ] } }, ... ]` 형태의 중첩된 구조를 가지고 있습니다.
- **DB 요구사항**: `Major` 모델은 평탄화된(Flat) 필드를 요구합니다.
- **해결 방안**: 전처리 단계에서 껍데기를 벗겨내고 `content` 내부의 객체를 추출해야 합니다.

### 2.2 식별자 부재 (Missing ID)
- **현상**: `Major` 모델은 고유 식별자 `major_id`를 요구하지만, 원본 데이터에는 명시적인 ID 필드가 없습니다.
- **해결 방안**: `major` 이름(예: "고고학과")을 기반으로 UUID를 생성하거나, 해시값을 생성하여 `major_id`로 사용해야 합니다.

## 3. 필드별 매핑 및 타입 검토

| DB 컬럼명 | 원본 JSON 키 | 타입 매칭 | 조치 사항 / 전처리 로직 |
|---|---|---|---|
| `major_name` | `major` | 일치 | 그대로 사용 |
| `salary` | `salary` | **불일치** (Float vs String) | 문자열 "186.2" → 실수 `186.2` 변환 필요 |
| `employment` | `employment` | 주의 (HTML 포함) | "40 % 이상"과 같은 HTML 태그 제거 권장 (또는 그대로 저장) |
| `department_aliases` | `department` | **형식 차이** | 원본은 콤마 구분 문자열("A, B"). 리스트로 변환 후 `json.dumps()`하여 저장 권장 |
| `summary` | `summary` | 일치 | 그대로 사용 |
| `relate_subject` | `relate_subject` | 구조적 데이터 | List[Dict] 형태이므로 `json.dumps()`로 직렬화하여 저장 |
| `career_act` | `career_act` | 구조적 데이터 | List[Dict] → `json.dumps()` |
| `job` | `job` | 일치 | 콤마 구분 문자열. 그대로 저장하거나 리스트 변환 고려 |
| `qualifications` | `qualifications` | 일치 | 콤마 구분 문자열. 그대로 저장 |
| `interest` | `interest` | 일치 | 그대로 사용 |
| `property` | `property` | 일치 | 그대로 사용 |
| `enter_field` | `enter_field` | 구조적 데이터 | List[Dict] → `json.dumps()` |
| `main_subject` | `main_subject` | 구조적 데이터 | List[Dict] → `json.dumps()` |
| `university` | `university` | 구조적 데이터 | List[Dict] → `json.dumps()` **(데이터 큼 주의)** |
| `chart_data` | `chartData` | 키 이름 불일치 | `chartData` → `chart_data` 매핑 및 `json.dumps()` |
| `cluster` | **없음** | **누락** | 원본에 없음. 비워두거나 별도 분류 로직 필요 |

## 4. 모델링 적합성 평가

### 4.1 JSON 컬럼 사용 (`LONGTEXT` vs `JSON`)
- 현재 MySQL 방언의 `JSON` 타입 대신 `LONGTEXT`를 사용하여 호환성을 확보하려는 접근은 안전하고 좋습니다 (특히 MariaDB/MySQL 버전 파편화 대비).
- 다만, `university` 필드의 경우 데이터 양이 상당히 많을 수 있어(개설 대학이 많은 학과), 이 필드를 자주 조회한다면 별도 테이블(`MajorUniversity`)로 정규화(Normalization)하는 것이 원칙적으로는 좋습니다.
- **결론**: 현재 RAG 시스템의 특성상(한 번에 문맥을 가져와야 함) **비정규화된 JSON 저장 방식이 조회 성능과 구현 편의성 면에서 더 적합**할 수 있습니다. 그대로 유지를 추천합니다.

## 5. 제안: 전처리 및 적재 파이프라인 (Preprocessing Logic)

데이터 적재 스크립트 작성 시 아래 로직을 포함해야 합니다.

```python
import json
import uuid
import re

def preprocess_major_data(raw_item):
    # 1. 중첩 구조 해제
    if "dataSearch" not in raw_item or "content" not in raw_item["dataSearch"]:
        return None
    
    content_list = raw_item["dataSearch"]["content"]
    if not content_list:
        return None
        
    data = content_list[0] # 첫 번째 요소 추출
    
    # 2. 데이터 변환
    
    # 2.1 ID 생성
    major_name = data.get("major", "").strip()
    if not major_name:
        return None
    # 고정된 ID 생성을 위해 namespace DNS 사용 (재실행 시에도 동일 ID 보장)
    major_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, major_name))
    
    # 2.2 수치 변환
    try:
        salary = float(data.get("salary", "0"))
    except ValueError:
        salary = 0.0
        
    # 2.3 차트 데이터 내 추가 통계 추출 (옵션)
    # chartData가 있다면 취업률 등을 추출하여 별도 컬럼에 저장 가능
    
    # 2.4 JSON 직렬화 (DB에 LONGTEXT로 저장하기 위함)
    def to_json(obj):
        return json.dumps(obj, ensure_ascii=False)
        
    # 2.5 학과 별칭 리스트 변환
    dept_str = data.get("department", "")
    dept_list = [d.strip() for d in dept_str.split(",") if d.strip()]
    
    processed = {
        "major_id": major_id,
        "major_name": major_name,
        "salary": salary,
        "employment": data.get("employment"),
        "department_aliases": to_json(dept_list), # 리스트로 변환 후 저장 추천
        "summary": data.get("summary"),
        "relate_subject": to_json(data.get("relate_subject", [])),
        "career_act": to_json(data.get("career_act", [])),
        "job": data.get("job"), # 그대로 저장
        "qualifications": data.get("qualifications"),
        "interest": data.get("interest"),
        "property": data.get("property"),
        "enter_field": to_json(data.get("enter_field", [])),
        "main_subject": to_json(data.get("main_subject", [])),
        "university": to_json(data.get("university", [])),
        "chart_data": to_json(data.get("chartData", [])),
        "raw_data": to_json(raw_item) # 원본 백업
    }
    
    return processed
```

### 6. 종합 의견
- **스키마 적합성**: 현재 `models.py`는 원본 데이터의 정보를 모두 담기에 충분하며 적절합니다. `LONGTEXT`를 활용한 반정규화 방식은 RAG 워크로드에 효율적입니다.
- **데이터 품질**: 원본 데이터는 풍부하지만(`chartData` 등), DB에 넣기 위해서는 타입을 맞추고 불필요한 HTML 태그를 정리하는 등의 **전처리(Preprocessing)가 필수적**입니다.
- **Cluster 필드**: `cluster` 컬럼은 원본에 없으므로, 추후 데이터를 보강하거나 비워둬야 합니다.
