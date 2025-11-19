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

## 구성 설명

- **backend**
  - LangGraph + LangChain RAG 파이프라인 전체를 담당합니다.
  - `config.py`는 모든 경로 및 모델 설정을 중앙에서 관리하며, `.env` 기반으로 LLM/임베딩을 선택합니다.
  - `rag/loader.py`는 JSON 데이터를 LangChain `Document`로 변환하고, `rag/vectorstore.py`는 Chroma 벡터스토어를 생성·로드합니다.
  - `graph/` 폴더에는 RAG 파이프라인을 LangGraph로 정의한 노드, 상태, 그래프 빌더가 들어 있습니다.

- **frontend**
  - `frontend/app.py`는 Streamlit UI를 제공하며 `backend.main.run_mentor`를 직접 호출해 답변을 보여줍니다.

- **데이터 소스**
  - https://www.career.go.kr/cnet/front/openapi/openApiMajorCenter.do
  - API에서 받아온 과목 정보를 JSON으로 저장 후 `RAW_JSON` 경로에 둡니다.

## 참고/아이디어 메모

- 챗봇이 제공할 수 있는 기능
  - 커리큘럼 비교, 특정 과목 설명, 학기별 개설 여부 안내.
  - 능력/학년별 추천 과목, 특정 학교/학과에서 경험할 학업 로드맵 요약.
  - 불필요한 질문 필터링, 사용 가이드(프롬프팅 팁) 제공.

- RAG 품질 점검
  - 다양한 학교·학과 조합에 대한 검색 결과 확인.
  - 과목 설명, 비교 분석, 특색 있는 강의 정보 등을 충분히 포함하는지 검증.
