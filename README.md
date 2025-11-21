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
   conda create -n {가상환경명} python=3.11
   pip install -r requirements.txt
   ```

3. **벡터스토어 구축**

   ```bash
   python -m backend.rag.vectorstore
   ```

   - `.env`의 `RAW_JSON`(glob 가능)에서 코스를 읽어 LangChain `Document`로 변환하고, `VECTORSTORE_DIR` 경로에 Chroma DB를 생성/갱신합니다.
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
│  │  │  └─ *.json         # 원본 JSON
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

이 프로젝트는 **두 가지 다른 RAG 패턴**을 지원합니다:

### 1. ReAct 패턴 (기본값, Agentic)

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
2. `agent_node`: LLM이 질문을 분석하고 "과목 정보가 필요하다"고 판단
3. LLM이 `retrieve_courses` tool 호출 결정 (tool_calls 포함하여 응답)
4. `should_continue`: tool_calls 감지 → tools 노드로 라우팅
5. `tools` 노드: `retrieve_courses` 함수 실행 → 벡터 DB에서 과목 검색
6. `agent_node`로 복귀: LLM이 검색 결과 보고 최종 답변 생성
7. `should_continue`: tool_calls 없음 → 종료

**핵심 파일:**
- `backend/rag/tools.py`: `@tool` 데코레이터로 정의된 LangChain tool
- `backend/graph/nodes.py`: `agent_node`, `should_continue`
- `backend/graph/graph_builder.py`: `build_react_graph()`

**장점:**
- LLM이 필요시에만 tool 호출 (효율적)
- 여러 번 tool 호출 가능 (복잡한 질문 처리)
- 진정한 Agentic 동작

### 2. Structured 패턴 (고정 파이프라인)

**미리 정해진 순서대로 실행되는 파이프라인 방식**

```
[사용자 질문] → retrieve_node → select_node → answer_node → [답변]
```

**작동 순서:**
1. `retrieve_node`: 벡터 DB에서 관련 과목 5개 검색
2. `select_node`: LLM이 JSON 형식으로 적합한 과목 2-3개 선택
3. `answer_node`: 선택된 과목만 사용하여 최종 답변 생성

**핵심 파일:**
- `backend/graph/nodes.py`: `retrieve_node`, `select_node`, `answer_node`
- `backend/graph/graph_builder.py`: `build_structured_graph()`

**장점:**
- Hallucination 방지 (선택된 과목만 LLM에게 제공)
- 명확한 실행 순서 (디버깅 용이)
- 예측 가능한 동작

### 패턴 비교표

| | **ReAct** | **Structured** |
|---|---|---|
| **Agentic** | ✅ 예 (LLM이 tool 호출 결정) | ❌ 아니오 (고정 순서) |
| **실행 방식** | agent ⇄ tools (반복 가능) | retrieve → select → answer |
| **tool 호출** | LLM이 자율 결정 | 무조건 실행 |
| **유연성** | 높음 (복잡한 질문 처리) | 낮음 (단순 파이프라인) |
| **Hallucination** | 가능성 있음 | 낮음 (선택된 과목만 제공) |
| **현재 사용** | ✅ 기본값 | 옵션 |

### 모드 변경 방법

`backend/main.py`의 `run_mentor()` 함수에서 `mode` 파라미터로 변경:

```python
# ReAct 모드 (기본)
answer = run_mentor("인공지능 과목 추천해줘")

# Structured 모드
answer = run_mentor("인공지능 과목 추천해줘", mode="structured")
```

또는 `frontend/app.py`에서 직접 변경:

```python
response = run_mentor(
    question=prompt,
    interests=st.session_state.interests or None,
    mode="structured"  # ← 여기를 변경
)
```

## 구성 설명

- **backend**
  - LangGraph + LangChain RAG 파이프라인 전체를 담당합니다.
  - `config.py`는 모든 경로 및 모델 설정을 중앙에서 관리하며, `.env` 기반으로 LLM/임베딩을 선택합니다.
  - `rag/loader.py`는 JSON 데이터를 LangChain `Document`로 변환하고, `rag/vectorstore.py`는 Chroma 벡터스토어를 생성·로드합니다.
  - `rag/tools.py`는 `@tool` 데코레이터로 LLM이 호출할 수 있는 tool을 정의합니다.
  - `graph/` 폴더에는 RAG 파이프라인을 LangGraph로 정의한 노드, 상태, 그래프 빌더가 들어 있습니다.

- **frontend**
  - `frontend/app.py`는 Streamlit UI를 제공하며 `backend.main.run_mentor`를 직접 호출해 답변을 보여줍니다.

- **데이터 소스**
  - <https://www.career.go.kr/cnet/front/openapi/openApiMajorCenter.do>
  - API에서 받아온 과목 정보를 JSON으로 저장 후 `RAW_JSON` 경로에 둡니다.

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

#### LLM 모델만 변경하는 경우:
- [x] `.env`에서 `LLM_PROVIDER` 설정
- [x] `.env`에서 `MODEL_NAME` 설정
- [x] 필요시 API 키 설정
- [x] 애플리케이션 재시작: `streamlit run frontend/app.py`

#### Embedding 모델을 변경하는 경우:
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
