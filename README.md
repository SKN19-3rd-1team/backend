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
   python -m venv .venv
   .\.venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
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

## 참고/아이디어 메모

- 챗봇이 제공할 수 있는 기능
  - 커리큘럼 비교, 특정 과목 설명, 학기별 개설 여부 안내.
  - 능력/학년별 추천 과목, 특정 학교/학과에서 경험할 학업 로드맵 요약.
  - 불필요한 질문 필터링, 사용 가이드(프롬프팅 팁) 제공.

- RAG 품질 점검
  - 다양한 학교·학과 조합에 대한 검색 결과 확인.
  - 과목 설명, 비교 분석, 특색 있는 강의 정보 등을 충분히 포함하는지 검증.
