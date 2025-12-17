# RAG 시스템 최적화 및 메타데이터 강화 보고서

## 개요
RAG 시스템 평가(`docs/rag_evaluation.md`)에서 식별된 주요 성능 병목 현상과 기능 공백을 해결하기 위해 시스템 전반의 최적화를 수행했습니다.

## 주요 수정 사항

### 1. 메모리 최적화 (Memory Optimization)
- **변경 전**: `backend/rag/tools.py`가 실행 시 `major_detail.json` (약 600개 이상의 레코드, 10MB+) 전체를 메모리에 로드하여 캐싱(`_MAJOR_RECORDS_CACHE`)함. 서버 메모리 사용량이 높고 스케일링에 불리함.
- **변경 후**:
    - 인메모리 캐시(`_MAJOR_RECORDS_CACHE`) 및 관련 로직(`_ensure_major_records`)을 **완전히 제거**했습니다.
    - `tools.py`의 모든 조회 로직(`_lookup_major_by_name`, `_filter_majors_by_token`, `list_departments`)을 **SQLAlchemy를 사용한 실시간 DB 쿼리**로 전환했습니다.
    - `list_departments` 툴 호출 시 전체 데이터를 로드하지 않고 필요한 만큼만 DB에서 조회하도록 변경했습니다.

### 2. 메타데이터 강화 (Metadata Enhancement)
- **변경 전**: Pinecone 벡터 스토어에 `salary` 정보만 있고, 중요 지표인 **취업률(employment_rate)**과 **입학 경쟁률(acceptance_rate)**이 누락되어 있어 필터링이 불가능했습니다.
- **변경 후**:
    - `MajorDoc` 데이터클래스에 위 두 필드를 추가했습니다.
    - `vectorstore.py`의 인덱싱 로직을 업데이트하여 Pinecone 메타데이터에 해당 수치를 포함시켰습니다.
    - 이를 통해 향후 취업률 70% 이상 전공 검색 등의 고급 필터링이 가능해졌습니다.

### 3. 검색 리콜 개선 (Search Recall Improvement)
- **변경 전**: `retriever.py`의 검색 후보 수(`top_k`)가 50개로 설정되어 있어, 관련성이 높지만 벡터 유사도가 약간 낮은 전공들이 최종 집계 단계 전에 탈락하는 문제가 있었습니다.
- **변경 후**: `top_k` 기본값을 **150**으로 상향 조정하여 더 넓은 범위의 후보군을 확보하고, 4단계 하이브리드 검색의 정확도를 높였습니다.

## 검증 결과
- **인덱스 재구축**: `backend.rag.build_major_index` 스크립트를 통해 메타데이터가 포함된 새로운 인덱스를 성공적으로 생성했습니다.
- **기능 테스트**: DB 기반의 검색 로직이 정확히 동작함을 확인했습니다(정확 일치, 유사어 검색, 벡터 검색 연동).
