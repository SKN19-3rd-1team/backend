# 수정 로그 (Fixed Log)

## 2025-12-02 11:24

### 1. `backend/rag/tools.py` 리팩토링
- **상수 분리**: 검색 제한, 파일 경로 등 매직 넘버를 상수로 추출하여 관리 용이성 증대
- **타입 힌트 개선**: 모든 함수에 명확한 타입 힌트(`Optional`, `List`, `Dict` 등) 추가
- **한글 주석 추가**: 함수별 상세한 한글 독스트링 및 인라인 주석 추가로 가독성 향상
- **코드 구조 개선**: 긴 함수(`_find_majors` 등)를 작은 단위로 분리하고 에러 처리 로직 표준화

### 2. 데이터 파싱 로직 개선 (`backend/rag/loader.py`)
- **새로운 필드 추가**: `MajorRecord` 클래스에 `gender`(성비), `satisfaction`(만족도), `employment_rate`(취업률), `acceptance_rate`(합격률) 필드 추가
- **합격률 계산 로직 구현**: `chartData` 내의 지원자/입학자 수 데이터를 기반으로 합격률(%)을 자동 계산하는 `_calculate_acceptance_rate` 함수 구현
- **데이터 추출 개선**: `major_detail.json`의 `chartData` 리스트 구조에 맞춰 성비, 만족도, 취업률 데이터를 올바르게 추출하도록 파싱 로직 수정

### 3. 툴 기능 확장 (`backend/rag/tools.py`)
- **통계 정보 제공**: `get_major_career_info` 툴이 전공의 성비, 취업률, 만족도, 합격률 정보를 함께 반환하도록 수정
- **로그 개선**: 툴 실행 시 해당 통계 정보의 유무를 로그에 출력하도록 개선

### 4. 프론트엔드 UX 개선 (`frontend/app.py`)
- **스트리밍 응답 구현**: `st.write_stream`과 제너레이터를 사용하여 챗봇 응답이 타자를 치듯 실시간으로 출력되는 효과 적용 (Simulated Streaming)

## 2025-12-02 12:21

### 1. 연봉 정보 제공 로직 개선 (`backend/rag/tools.py`)
- **연봉 환산 기능 추가**: `get_major_career_info` 툴에서 월평균 임금(`salary`) 데이터에 12를 곱하여 연봉(`annual_salary`)을 자동 계산하여 반환하도록 수정

### 2. 시스템 프롬프트 고도화 (`backend/graph/nodes.py`)
- **답변 포맷 표준화**: 연봉 관련 질문 시 "**{학과명}**을(를) 졸업시 평균 연봉은 **{annual_salary}**만원 정도입니다!"라는 고정된 형식으로 답변하도록 지시
- **출처 명시**: 직업/진로 정보 제공 시 데이터 출처가 "커리어 넷"임을 자연스럽게 언급하도록 가이드라인 추가

## 2025-12-02 14:17

### 1. 데이터 경량화 및 전처리 (`backend/data/major_detail_v2.json`)
- **불필요 필드 제거**: 데이터 용량 최적화를 위해 `GenCD`, `SchClass`, `lstMiddleAptd`, `lstHighAptd`, `lstVals` 필드 삭제
- **차트 데이터 정리**: `chartData` 내의 `after_graduation` 필드 삭제로 데이터 구조 단순화

### 2. 데이터 동기화 (`backend/data/major_categories.json`)
- **카테고리 업데이트**: `major_detail_v2.json`의 최신 전공 및 학과 정보를 기반으로 `major_categories.json`을 자동 갱신하여 데이터 일관성 확보

## 2025-12-02 15:23

### 1. LLM 환각(Hallucination) 방지 강화 (`backend/graph/nodes.py`)
- **절대 규칙 섹션 신설**: 🚨 시각적 강조와 함께 "절대 추측 금지" 규칙을 최상단에 명시하여 LLM이 데이터베이스에 없는 학과명을 만들어내지 못하도록 강력히 제한
- **유사 전공 추천 가이드라인 추가**: "~와 비슷한", "~와 유사한" 질문 시 반드시 `list_departments` 툴을 호출하도록 명시적 지시 추가
- **검색 키워드 추출 방법 제공**: 예시를 통해 LLM이 핵심 키워드를 추출하여 검색하도록 유도 (예: "국어교육과" → "교육", "국어", "언어")
- **존재하지 않는 학과 언급 금지**: "영어치료학과", "아동언어치료학과" 등 실제로 존재하지 않는 학과를 구체적 예시로 제시하며 절대 언급 금지 강조
- **정직한 답변 유도**: 적합한 전공을 찾지 못한 경우 솔직하게 "데이터베이스에서 관련 전공을 찾지 못했습니다"라고 답변하도록 지시

## 2025-12-02 16:32 ~ 17:01

### 1. 온보딩 전공 추천 알고리즘 개선 (`backend/graph/nodes.py`)
- **선호 전공 우선 처리 강화**: `recommend_majors_node` 함수에 `preferred_majors` 별도 검색 로직 추가
- **보너스 점수 시스템**: 사용자가 명시한 `preferred_majors`를 `_find_majors`로 별도 검색하여 발견 보장
- **강제 결과 포함**: 벡터 검색 top 50에 없어도 선호 전공을 결과에 강제 추가 (기본 점수 1.0)
- **5배 보너스 적용**: 선호 전공에 5배 보너스 점수를 적용하여 최상위 배치 보장
- **디버깅 로그 추가**: `🔍 Searching for...`, `✅ Added...`, `🎯 Boosted...` 메시지로 추적 가능

### 2. 단일 학과명 질문 처리 개선
**파일 생성: `backend/graph/helper.py`**
- **입력 감지 함수**: `is_single_major_query()` - 단일 학과명 질문인지 자동 감지 ("고분자공학과", "컴퓨터공학" 등)
- **질문 개선 함수**: `enhance_single_major_query()` - 명확한 질문으로 자동 변환
  - 예: "고분자공학과" → "고분자공학과에 대해 자세히 알려주세요. 어떤 학과이고, 어디 대학에 있으며..."

**`backend/graph/nodes.py` - `agent_node` 함수 수정**
- **입력 전처리 로직 추가**: 사용자 질문 전송 전 자동 감지 및 변환
- **디버깅 로그**: `🔍 Detected single major query`, `✨ Enhanced to` 메시지로 변환 과정 추적

**시스템 프롬프트 강화**
- **새 섹션 추가**: `[학과명 단일 입력 처리 - 매우 중요!]`
- **명시적 지시사항**: 학과명만 입력 시 "찾을 수 없다" 절대 금지, 반드시 `get_major_career_info` + `get_universities_by_department` 툴 호출

### 3. LLM 신뢰성 개선 (`backend/config.py`)
- **Temperature 조정**: OpenAI ChatModel의 temperature를 0.7 → 0.1로 낮춤
- **효과**: 더 deterministic한 응답 생성 및 툴 호출 신뢰성 향상
- **적용 범위**: 공식 OpenAI API 및 커스텀 base_url 모두 적용

### 4. 전체 개선 효과
- **선호 전공 추천 정확도**: 사용자가 "고분자공학과" 선호 시 반드시 TOP 5 내 포함
- **단일 질문 처리**: "고분자공학과" 입력 시 자동으로 상세 정보 제공 (이전에는 "찾을 수 없다" 오류)
- **일관성 향상**: Temperature 0.1로 매 실행마다 일관된 툴 호출 보장

## 2025-12-02 17:41

### 1. 툴 정의 및 시스템 프롬프트 최적화
- **툴 정의 고도화 (`backend/rag/tools.py`)**:
  - `list_departments`, `get_major_career_info`, `get_universities_by_department`, `get_search_help` 툴의 docstring을 `backend/tools_quote_example.py` 스타일로 전면 개편
  - **사용 시나리오 추가**: 각 툴마다 "이 툴을 호출해야 하는 상황 (LLM용 가이드)" 섹션을 추가하여 LLM의 판단력 향상
  - **파라미터 설명 추가**: 입력값에 대한 상세 가이드를 포함하여 정확한 인자 전달 유도

- **시스템 프롬프트 경량화 (`backend/graph/nodes.py`)**:
  - **중복 제거**: 툴 정의로 이관된 사용 지침(단일 학과명 처리, 유사 전공 추천 등)을 시스템 프롬프트에서 제거
  - **핵심 집중**: 페르소나와 절대 규칙(추측 금지, 툴 결과 사용 등)만 남겨 컨텍스트 효율성 증대
  - **토큰 절약**: 툴 정의 자체에서 사용법을 학습하게 하여 더 적은 토큰으로 정확한 툴 호출 기대
