# langchain_practice

## 실행 순서

프로젝트를 처음부터 끝까지 실행하기 위해 필요한 명령어들을 순서대로 정리했습니다.

1. **환경 변수 파일 생성**

   ```bash
   cp .env.example .env          # Windows PowerShell: copy .env.example .env
   ```

   - 발급받은 `OPENAI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`, `LANGCHAIN_API_KEY`를 `.env`에 채웁니다.
   - `RAW_JSON`, `VECTORSTORE_DIR`, `LLM_PROVIDER`, `MODEL_NAME`, `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL_NAME` 등을 환경에 맞게 조정하세요.

2. **가상환경 및 의존성 설치 (최초 1회)**

   ```bash
   conda activate langchain_env
   pip install -r requirements.txt
   ```

3. **벡터스토어 구축**

   ```bash
   python -m backend.rag.build_major_index
   ```

   - `.env`의 `RAW_JSON`(glob 가능)에서 코스를 읽어 LangChain `Document`로 변환하고, `VECTORSTORE_DIR` 경로에 Vector DB를 생성/갱신합니다.
   - 원본 데이터나 임베딩 모델 설정을 바꿨다면 이 명령을 다시 실행하세요.

4. **Streamlit 챗봇 실행**

   ```bash
   streamlit run frontend/app.py
   ```

   - 터미널에 현재 사용 중인 LLM/임베딩 모델명이 출력되고, 브라우저에서 상담 챗봇 UI가 열립니다.
   - 질문 입력 시 LangGraph 기반 RAG 파이프라인이 동작하여 답변을 생성합니다.

## 프로젝트 구조

```
major-mentor-bot/
├─ backend/
│  ├─ data/
│  │  ├─ raw/
│  │  │  └─ *.json                        # 원본 JSON
│  │  └─ processed/
│  │     └─ courses.parquet               # 전처리/캐시(optional)
│  ├─ graph/
│  │  ├─ __init__.py
│  │  ├─ nodes.py                         # LangGraph 노드 정의
│  │  ├─ state.py                         # State 모델 정의
│  │  └─ graph_builder.py                 # Graph 생성 함수
│  ├─ rag/
│  │  ├─ loader.py                        # JSON → Document 로딩
│  │  ├─ embeddings.py                    # 임베딩 설정
│  │  ├─ vectorstore.py                   # 벡터DB 초기화/저장/로딩
│  │  └─ retriever.py                     # Retriever 래퍼
│  ├─ api/
│  │  └─ __init__.py
│  ├─ config.py                           # 경로, 모델 이름, 설정값
│  ├─ main.py                             # LangGraph 엔트리포인트
│  └─ requirements.txt
│
├─ frontend/
│  ├─ app.py                              # Streamlit 메인 앱
│  └─ requirements.txt
│
├─ .env.example                           # API 키/환경변수 예시
├─ README.md
└─ pyproject.toml or environment.yml (선택)
```

## 프로젝트 작동 방식

이 프로젝트는 **두 가지 주요 워크플로우**를 지원합니다:

### 1. ReAct 패턴 (대화형 멘토링)

**LLM이 자율적으로 tool 호출 여부를 결정하는 에이전트 방식**

```
[사용자 질문] → agent_node → should_continue
                    ↑              ↓
                    └── tools ←────┘
                         ↓
                      [답변]
```

**작동 순서:**

1. 사용자가 질문 입력 (예: "인공지능 관련 과목 추천해줘")
2. `agent_node`: LLM이 질문을 분석하고 정보가 필요한지 판단
3. LLM이 적절한 tool (`list_departments`, `get_universities_by_department` 등) 호출 결정
4. `should_continue`: tool_calls 감지 → tools 노드로 라우팅
5. `tools` 노드: 선택된 툴 실행 (학과 검색, 대학 찾기 등)
6. `agent_node`로 복귀: LLM이 툴 결과를 바탕으로 답변 생성
7. `should_continue`: tool_calls 없음 → 종료

**핵심 파일:**

- `backend/rag/tools.py`: 학과 검색, 대학 조회, 진로 정보 조회 등의 툴 정의
- `backend/graph/nodes.py`: `agent_node`, `should_continue`
- `backend/graph/graph_builder.py`: `build_react_graph()`

### 2. Major Recommendation 패턴 (온보딩 추천)

**사용자의 온보딩 답변을 분석하여 전공을 추천하는 파이프라인**

```
[온보딩 답변] → recommend_majors_node → [추천 결과]
```

**작동 순서:**

1. 사용자의 온보딩 답변(관심사, 과목, 희망 연봉 등) 수집
2. `recommend_majors_node`: 답변을 텍스트 프로필로 변환 및 임베딩
3. Vector Store(Pinecone)에서 관련 전공 검색
4. 가중치를 적용하여 전공 점수 계산 및 상위 전공 추천

**핵심 파일:**

- `backend/graph/nodes.py`: `recommend_majors_node`
- `backend/graph/graph_builder.py`: `build_major_graph()`

### 실행 방법

**대화형 멘토링 (ReAct):**

```python
from backend.main import run_mentor

answer = run_mentor("컴퓨터공학과가 있는 대학 알려줘")
```

**전공 추천 (Onboarding):**

```python
from backend.main import run_major_recommendation

results = run_major_recommendation({
    "interests": "코딩, 로봇",
    "subjects": "수학, 과학"
})
```

## 구성 설명

- **backend**
  - LangGraph + LangChain RAG 파이프라인 전체를 담당합니다.
  - `config.py`는 모든 경로 및 모델 설정을 중앙에서 관리하며, `.env` 기반으로 LLM/임베딩을 선택합니다.
  - `rag/loader.py`는 JSON 데이터를 LangChain `Document`로 변환하고, `rag/vectorstore.py`는 Vector 벡터스토어를 생성·로드합니다.
  - `rag/tools.py`는 `@tool` 데코레이터로 LLM이 호출할 수 있는 tool을 정의합니다.
  - `graph/` 폴더에는 RAG 파이프라인을 LangGraph로 정의한 노드, 상태, 그래프 빌더가 들어 있습니다.

- **frontend**
  - `frontend/app.py`는 Streamlit UI를 제공하며 `backend.main.run_mentor`를 직접 호출해 답변을 보여줍니다.

- **데이터 소스**
  - API에서 받아온 과목 정보를 JSON으로 저장 후 `RAW_JSON` 경로에 둡니다.

### 전공 카테고리 데이터 관리

이 프로젝트는 `backend/rag/tools.py`에서 사용하는 전공 카테고리 정보(`MAIN_CATEGORIES`)를 `backend/data/major_detail.json`에서 동적으로 추출하여 사용합니다.

1. **카테고리 추출 스크립트**: `backend/rag/extract_categories.py`
   - `major_detail.json`을 분석하여 전공명과 관련 학과들을 매핑합니다.
   - 실행 결과는 `backend/data/major_categories.json`에 저장됩니다.

2. **데이터 갱신 방법**:
   - `major_detail.json` 데이터가 변경되면 아래 명령어로 카테고리 정보를 갱신해야 합니다.

   ```bash
   python backend/rag/extract_categories.py
   ```

3. **동적 로딩**:
   - `backend/rag/tools.py`는 실행 시 `major_categories.json`을 로드하여 최신 카테고리 정보를 반영합니다.

## 데이터 아키텍처 및 메타데이터 구조

이 프로젝트는 **2단계 데이터 저장 전략**을 사용하여 효율적인 검색과 상세 정보 조회를 동시에 지원합니다.

### 메타데이터 구조

전공 정보는 `MajorDoc` 클래스를 통해 Pinecone 벡터 데이터베이스에 저장되며, 다음과 같은 메타데이터를 포함합니다:

```python
@dataclass
class MajorDoc:
    doc_id: str                    # 문서 고유 ID (예: "컴퓨터공학:summary")
    major_id: str                  # 전공 고유 ID
    major_name: str                # 전공명
    doc_type: str                  # 문서 타입 (summary, interest, property, subjects, jobs)
    text: str                      # 임베딩할 텍스트 내용
    cluster: Optional[str]         # 전공 클러스터 분류
    salary: Optional[float]        # 평균 급여 정보
    relate_subject_tags: list[str] # 관련 과목 태그 리스트
    job_tags: list[str]            # 진출 직업 태그 리스트
    raw_subjects: Optional[str]    # 원본 과목 정보
    raw_jobs: Optional[str]        # 원본 직업 정보
```

### 문서 타입 (doc_type) 분류

각 전공은 여러 개의 문서로 분해되어 저장됩니다:

- **`summary`**: 전공 요약 설명
- **`interest`**: 흥미/적성 정보 + 추천 활동
- **`property`**: 전공 특성
- **`subjects`**: 관련 과목 안내
- **`jobs`**: 진출 직업 및 분야

### 데이터 흐름 구조

```
major_detail.json (원본 데이터)
        ↓
load_major_detail() [backend/rag/loader.py]
        ↓
MajorRecord 객체 생성 (raw 필드에 전체 데이터 보관)
        ↓
┌───────────────────┬────────────────────┐
│                   │                    │
Pinecone 업로드      메모리 캐시           직접 조회
(메타데이터만)       (_MAJOR_RECORDS_CACHE)  (tools.py)
```

### 2단계 데이터 저장 전략

#### 1단계: Pinecone 벡터 데이터베이스 (검색용)

**저장 데이터:**
- 메타데이터: `doc_id`, `major_id`, `major_name`, `doc_type`, `cluster`, `salary`, `relate_subject_tags`, `job_tags`
- 임베딩 벡터: `text` 필드를 임베딩한 벡터

**용도:**
- 의미 기반 유사도 검색
- 빠른 전공 필터링 및 매칭

**예시:**
```python
# "AI 관련 전공 찾기"와 같은 의미 기반 검색
results = vectorstore.similarity_search("인공지능 관련 전공")
```

#### 2단계: 메모리 캐시 (상세 정보용)

**저장 데이터:**
- 전체 `MajorRecord` 객체 (모든 원본 데이터 포함)
- `raw` 필드에 `major_detail.json`의 원본 JSON 데이터 전체 보관

**용도:**
- 상세 정보 조회 (대학 목록, 자격증, 주요 과목 등)
- Pinecone에 저장되지 않은 추가 정보 제공

**캐싱 메커니즘:**
```python
# backend/rag/tools.py
_MAJOR_RECORDS_CACHE = None  # 전체 MajorRecord 리스트
_MAJOR_ID_MAP = {}           # major_id로 빠른 조회
_MAJOR_NAME_MAP = {}         # 전공명으로 빠른 조회
_MAJOR_ALIAS_MAP = {}        # 별칭으로 빠른 조회

def _ensure_major_records():
    """첫 호출 시 major_detail.json을 로드하여 메모리에 캐싱"""
    global _MAJOR_RECORDS_CACHE
    if _MAJOR_RECORDS_CACHE is not None:
        return
    
    records = load_major_detail()  # 전체 원본 데이터 로드
    _MAJOR_RECORDS_CACHE = records
    # 인덱스 생성...
```

### 실제 사용 흐름

```python
# 1단계: Pinecone으로 관련 전공 검색 (벡터 유사도)
results = vectorstore.similarity_search("AI 관련 전공")

# 2단계: 검색된 전공의 major_id로 상세 정보 조회
record = _MAJOR_ID_MAP[major_id]

# 3단계: 원본 데이터의 모든 필드 활용
university_list = record.university      # Pinecone에 없는 정보
chart_data = record.chart_data          # Pinecone에 없는 정보
employment_rate = record.employment     # Pinecone에 없는 정보
raw_json = record.raw                   # 전체 원본 JSON
```

### 메타데이터 활용 예시

#### 필터링 및 정렬
```python
# 급여 정보를 기반으로 필터링
high_salary_majors = [doc for doc in docs if doc.salary and doc.salary > 4000]

# 클러스터별 그룹화
engineering_majors = [doc for doc in docs if doc.cluster == "공학계열"]
```

#### 태그 기반 매칭
```python
# 사용자가 선택한 과목과 매칭
user_subjects = ["수학", "물리"]
matched = [doc for doc in docs 
           if any(subj in doc.relate_subject_tags for subj in user_subjects)]
```

### 왜 이렇게 설계했을까?

| 구분 | Pinecone (벡터 DB) | 메모리 캐시 (MajorRecord) |
|------|-------------------|------------------------|
| **용도** | 의미 기반 검색 | 상세 정보 조회 |
| **저장 데이터** | 메타데이터 + 임베딩 | **전체 원본 데이터** |
| **장점** | 빠른 유사도 검색 | 모든 필드 접근 가능 |
| **예시** | "AI 관련 전공 찾기" | "컴퓨터공학과의 자격증 목록" |
| **크기** | 가볍고 효율적 | 전체 데이터 보관 |

### 관련 파일

- **`backend/rag/loader.py`**: `MajorRecord`, `MajorDoc` 클래스 정의 및 데이터 로딩
- **`backend/rag/tools.py`**: 메모리 캐시 관리 및 상세 정보 조회 툴
- **`backend/rag/vectorstore.py`**: Pinecone 벡터 데이터베이스 관리
- **`backend/data/major_detail.json`**: 원본 전공 데이터

## LLM/Embedding 모델 변경 방법

이 프로젝트는 **LLM 모델**과 **Embedding 모델**을 각각 독립적으로 설정할 수 있습니다.

### 중요: 모델 변경 시 주의사항

⚠️ **Embedding 모델을 변경하면 반드시 벡터스토어를 재생성해야 합니다!**

```bash
# Embedding 모델 변경 후 필수 실행
python -m backend.rag.vectorstore
```

벡터스토어는 기존 embedding 모델로 생성된 벡터를 저장하고 있으므로, 다른 embedding 모델로 변경하면 차원(dimension)이나 벡터 공간이 달라져 검색이 제대로 작동하지 않습니다.

### 1. OpenAI 모델 사용

#### .env 설정

```bash
# API Key 설정
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx

# LLM 설정
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o                    # 또는 gpt-4o-mini, gpt-4-turbo 등

# Embedding 설정
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL_NAME=text-embedding-3-large  # 또는 text-embedding-3-small
```

#### 사용 가능한 모델

- **LLM**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Embedding**: `text-embedding-3-large`, `text-embedding-3-small`, `text-embedding-ada-002`

#### 필요한 패키지 (이미 설치됨)

```bash
pip install langchain-openai openai
```

### 2. HuggingFace 모델 사용 (기본값)

#### .env 설정

```bash
# API Token 설정 (선택, 없어도 public 모델 사용 가능)
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxx

# LLM 설정
LLM_PROVIDER=huggingface
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct   # HuggingFace 모델 repo ID

# Embedding 설정
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=upskyy/bge-m3-korean   # 한국어 임베딩 모델
```

#### 추천 모델 조합

**⚠️ 중요: 반드시 `-Instruct`, `-Chat`, `-it` 등 instruction-tuned 모델을 사용하세요!**

이 프로젝트는 tool binding을 사용하므로 chat/conversational API를 지원하는 모델이 필요합니다.

**한국어 특화 (권장):**

```bash
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct           # ✅ 한국어 성능 우수, Chat 지원, Inference API 활성화
EMBEDDING_MODEL_NAME=upskyy/bge-m3-korean     # 한국어 임베딩

# 또는 (더 경량)
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct           # ✅ 빠른 응답, Inference API 활성화
```

**영어/다국어:**

```bash
MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct   # ✅ Chat 지원
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

**경량 모델 (빠른 응답):**

```bash
MODEL_NAME=google/gemma-2-2b-it                # ✅ Chat 지원 (it = instruction-tuned)
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**❌ 사용 불가능한 모델 예시:**

```bash
# 다음 모델들은 Chat API를 지원하지 않아 에러 발생
MODEL_NAME=upstage/SOLAR-10.7B-Instruct-v1.0  # ❌ text-generation only
MODEL_NAME=meta-llama/Llama-2-7b-hf           # ❌ base model
MODEL_NAME=EleutherAI/gpt-j-6B                # ❌ base model
```

#### 필요한 패키지 (이미 설치됨)

```bash
pip install langchain-huggingface huggingface_hub sentence-transformers
```

### 3. vLLM 서버 사용 (Inference API 미배포 모델 지원)

**Inference API가 배포되지 않은 모델**(예: `skt/A.X-4.0-Light`, `upstage/SOLAR-10.7B-Instruct-v1.0`)을 사용하려면 vLLM 서버로 로컬에서 모델을 직접 실행할 수 있습니다.

#### 사전 준비

1. vLLM 설치:

```bash
pip install vllm
```

2. vLLM 서버 시작 (별도 터미널):

```bash
# Linux/macOS
vllm serve skt/A.X-4.0-Light \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --api-key token-abc123

# Windows
python -m vllm.entrypoints.openai.api_server ^
    --model skt/A.X-4.0-Light ^
    --host 0.0.0.0 ^
    --port 8000 ^
    --dtype auto ^
    --api-key token-abc123
```

#### .env 설정

vLLM은 OpenAI 호환 API를 제공하므로 `openai` provider를 사용합니다:

```bash
# LLM 설정
LLM_PROVIDER=openai
MODEL_NAME=skt/A.X-4.0-Light              # vLLM에서 실행 중인 모델명
OPENAI_API_KEY=token-abc123               # vLLM 서버 시작 시 설정한 키
OPENAI_API_BASE=http://localhost:8000/v1  # vLLM 서버 주소

# Embedding 설정 (vLLM은 embedding 미지원, HuggingFace 사용)
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=upskyy/bge-m3-korean
```

#### 장점

- ✅ Inference API 미배포 모델 사용 가능 (`skt/A.X-4.0-Light` 등)
- ✅ Chat, Tool calling 완벽 지원
- ✅ GPU 사용으로 빠른 추론
- ✅ API 비용 없음 (로컬 실행)

#### 단점

- ❌ GPU 메모리 필요 (A.X-4.0-Light: 약 8-10GB VRAM)
- ❌ 별도 서버 프로세스 실행 필요

#### 필요한 패키지

```bash
pip install vllm  # vLLM 서버용
# langchain-openai는 이미 설치됨
```

### 4. Ollama 로컬 모델 사용

Ollama를 사용하면 로컬에서 모델을 실행할 수 있어 API 비용이 들지 않습니다.

#### 사전 준비

1. [Ollama 설치](https://ollama.ai/download)
2. 모델 다운로드:

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:7b-instruct
```

#### .env 설정

```bash
# LLM 설정
LLM_PROVIDER=ollama
MODEL_NAME=qwen2.5:7b-instruct        # Ollama 모델명

# Embedding 설정 (Ollama는 embedding 미지원, HuggingFace 사용)
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=upskyy/bge-m3-korean
```

#### 사용 가능한 모델

- `llama3.2:1b`, `llama3.2:3b` (경량)
- `qwen2.5:7b-instruct` (한국어 우수)
- `gemma2:2b`, `gemma2:9b`

#### 필요한 패키지 (이미 설치됨)

```bash
pip install langchain-community
```

### 4. 모델 변경 체크리스트

#### LLM 모델만 변경하는 경우

- [x] `.env`에서 `LLM_PROVIDER` 설정
- [x] `.env`에서 `MODEL_NAME` 설정
- [x] 필요시 API 키 설정
- [x] 애플리케이션 재시작: `streamlit run frontend/app.py`

#### Embedding 모델을 변경하는 경우

- [x] `.env`에서 `EMBEDDING_PROVIDER` 설정
- [x] `.env`에서 `EMBEDDING_MODEL_NAME` 설정
- [x] 필요시 API 키 설정
- [x] **⚠️ 벡터스토어 재생성 (필수!):** `python -m backend.rag.vectorstore`
- [x] 애플리케이션 재시작: `streamlit run frontend/app.py`

### 5. 일반적인 에러 및 해결 방법

#### 에러: "Unsupported LLM_PROVIDER"

```
ValueError: Unsupported LLM_PROVIDER: gpt-4. Use one of ['openai', 'ollama', 'huggingface'].
```

**원인:** `LLM_PROVIDER`에 모델명을 입력함
**해결:** `LLM_PROVIDER`는 `openai`, `ollama`, `huggingface` 중 하나만 가능

```bash
# 잘못된 예
LLM_PROVIDER=gpt-4

# 올바른 예
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o
```

#### 에러: "No API key found"

```
openai.AuthenticationError: No API key found for OpenAI
```

**해결:** `.env` 파일에 API 키 추가

```bash
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
```

#### 에러: "Model not found" (HuggingFace)

```
HfHubHTTPError: 404 Client Error: Repository Not Found
```

**원인:** 모델명이 잘못되었거나 private 모델
**해결:**

1. [HuggingFace Hub](https://huggingface.co/models)에서 정확한 모델명 확인
2. Private 모델이면 `HUGGINGFACEHUB_API_TOKEN` 설정

#### 에러: "Connection refused" (Ollama)

```
ConnectionError: [Errno 111] Connection refused
```

**원인:** Ollama 서버가 실행 중이지 않음
**해결:**

```bash
# Ollama 서버 시작
ollama serve
```

## 참고/아이디어 메모

- 챗봇이 제공할 수 있는 기능
  - 커리큘럼 비교, 특정 과목 설명, 학기별 개설 여부 안내.
  - 능력/학년별 추천 과목, 특정 학교/학과에서 경험할 학업 로드맵 요약.
  - 불필요한 질문 필터링, 사용 가이드(프롬프팅 팁) 제공.

- RAG 품질 점검
  - 다양한 학교·학과 조합에 대한 검색 결과 확인.
  - 과목 설명, 비교 분석, 특색 있는 강의 정보 등을 충분히 포함하는지 검증.
