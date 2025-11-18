# langchain_practice
```
major-mentor-bot/
├─ backend/
│  ├─ data/
│  │  ├─ raw/
│  │  │  └─ *.json         # 원본 JSON
│  │  └─ processed/
│  │     └─ courses.parquet               # 전처리/챙킹 후 저장(optional)
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
│  │  ├─ __init__.py
│  ├─ config.py                           # 경로, 모델 이름, 설정값
│  ├─ main.py                             # LangGraph 엔트리포인트 (CLI/백엔드)
│  └─ requirements.txt
│
├─ frontend/
│  ├─ app.py                              # Streamlit 메인 앱
│  └─ requirements.txt
│
├─ .env.example                           # API 키, 경로 등
├─ README.md
└─ pyproject.toml or environment.yml (선택)
```

- backend:

    - LangGraph + LangChain RAG 파이프라인

    - JSON 로딩, 벡터스토어 구축, 그래프 실행

- frontend:

    - Streamlit로 유저 질의/응답 UI

    - 백엔드를 Python import로 직접 호출하거나, FastAPI endpoint를 REST로 호출하는 구조 중 택1



- 학생 정보 → 서버로
- https://www.career.go.kr/cnet/front/openapi/openApiMajorCenter.do
    - API로 필요한 정보 로드
    - 모델 파인튜닝

챗봇
- 모델 차원에서 제공할 수 있는 답변
    - 서비스 사용 방법 (프롬프팅)
    - 쓸데없는 질문 -> 답변 x
    - 최대한 갖고있는 정보 기반으로 (프롬프팅)
    
- RAG 시스템을 통해 제공할 수 있는 답변
    - 컴퓨터공학과가 있는 대학이 뭐가 있어? -> "대학->공과대학->컴퓨터공학" -> 오버헤드가 크기 때문에 고민
    - ~~ 과목에 대한 설명 해줄래?
    - description + 학년/학기 기반으로 특정 대학, 특정 과에 입학 했을 때 어떠한 경험을 하게 될지 요약.
    - 여러 대학 커리큘럼 비교 (A 학교는 2학년데 알고리즘 배우는데 B 학교는 3학년에 배운다!)
    - 특정 학기에만 개강하는 과목들을 알려준다(1학기에 못들으면 1년 기다려야된다!)
    - **비슷한 레벨(수능 성적)의 비슷한 학과**의 커리큘럼 비교 -> **특정 과목**, **특색있는 강의**에 대한 정보 제공

[troubleshooting]
- 커리큘럼 추천 -> 학교, 학과 고려 x (2학년에 물리학과 수업듣고 3학년데 전기전자 수업듣고 ...)