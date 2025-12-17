# SK네트웍스 Family AI 캠프 19기 4차 프로젝트 - Refactored Version

## 1. 팀 소개

### 팀명

**Unigo (유니고)** - University Go, 당신을 위한 대학 입시 가이드

### 멤버

<div align="center">
  <table>
  <tr>
    <td align="center">      
      강지완
      <br/>
      <a href="https://github.com/Maroco0109">
        <img src="https://img.shields.io/badge/GitHub-Maroco0109-181717?style=flat&logo=github&logoColor=white">
      </a>
    </td> 
    <td align="center"> 
      김진
      <br/>
      <a href="https://github.com/KIMjjjjjjjj">
        <img src="https://img.shields.io/badge/GitHub-KIMjjjjjjjj-181717?style=flat&logo=github&logoColor=white">
      </a>
    </td>
    <td align="center"> 
      마한성
      <br/>
      <a href="https://github.com/gitsgetit">
        <img src="https://img.shields.io/badge/GitHub-gitsgetit-181717?style=flat&logo=github&logoColor=white">
      </a>
    </td> 
    <td align="center"> 
      오하원
      <br/>
      <a href="https://github.com/Hawon-Oh">
        <img src="https://img.shields.io/badge/GitHub-Hawon--Oh-181717?style=flat&logo=github&logoColor=white">
      </a>
    </td> 
  </tr>
</table>
</div>

---

## 2. 프로젝트 개요 (Refactored)

**Unigo CLI (AI 기반 대학 전공 추천 및 입시 상담 봇)**

이 프로젝트는 기존의 Django 기반 웹 애플리케이션을 **경량화된 Backend-Only CLI (Command Line Interface)** 형태로 리팩토링한 버전입니다. 핵심적인 AI 로직(LangGraph Agent, RAG)과 데이터 처리(SQLAlchemy, Pinecone) 기능은 그대로 유지하면서, 복잡한 웹 프론트엔드 및 컨테이너 의존성을 제거하여 핵심 로직 검증 및 학습에 초점을 맞췄습니다.

### 핵심 목표

1.  **Core Logic 집중**: 복잡한 웹 프레임워크 걷어내고 LangGraph Agent와 RAG 시스템의 순수 로직 구현체 보존.
2.  **가벼운 실행 환경**: Docker나 복잡한 설정 없이 로컬 Python 환경에서 즉시 실행 가능한 구조.
3.  **데이터 무결성 유지**: MySQL(정형)과 Pinecone(비정형)의 하이브리드 데이터 구조 유지.

---

## 3. 기술 스택

| 분류          | 기술                 | 비고                                 |
| ------------- | -------------------- | ------------------------------------ |
| **Language**  | Python 3.11+         | 메인 언어                            |
| **Interface** | CLI (Terminal)       | `cli.py` 엔트리포인트                |
| **Data**      | MySQL                | 관계형 데이터베이스 (전공/대학 정보) |
| **AI / RAG**  | LangChain, LangGraph | AI 에이전트 및 워크플로우 관리       |
| **LLM**       | OpenAI GPT-4o-mini   | 추론 및 자연어 생성                  |
| **Vector DB** | Pinecone             | 고성능 벡터 검색 (Serverless)        |

---

## 4. 프로젝트 구조

불필요한 웹 관련 디렉토리(`unigo/`, `nginx/`, `static/`)가 제거되고 구조가 단순화되었습니다.

```
Unigo/
├── backend/                             # AI 로직 핵심 패키지
│   ├── data/                            # 초기 데이터 (Seeding)
│   ├── db/                              # DB 관리 (SQLAlchemy)
│   ├── graph/                           # LangGraph 워크플로우 (Agent, Nodes)
│   ├── rag/                             # RAG 시스템 (Tools, Retriever)
│   ├── config.py                        # 설정 관리
│   └── main.py                          # AI 로직 엔트리포인트 (run_mentor)
│
├── docs/                                # 문서 자료
├── cli.py                               # CLI 실행 파일 (Entry Point)
├── .env                                 # 환경 변수 설정
└── requirements.txt                     # 의존성 패키지 목록
```

---

## 5. 시스템 아키텍처

웹 서버 계층이 제거되고, 사용자가 터미널을 통해 직접 `cli.py`와 상호작용하는 구조입니다.

```mermaid
flowchart TD
    User([User (Terminal)]) -->|Input Text| CLI[cli.py]
    CLI -->|Call| Main[backend.main]

    subgraph Backend ["Backend Logic"]
        Main -->|Invoke| Graph[LangGraph Agent]

        Graph -->|Reasoning| LLM[OpenAI GPT-4o]
        Graph -->|Tool Call| Tools[RAG Tools]

        Tools -->|Query| MySQL[(MySQL DB)]
        Tools -->|Search| Pinecone[(Pinecone Vector DB)]
    end

    Graph -->|Stream Response| CLI
    CLI -->|Print| User
```

---

## 6. 주요 기능

### 1) 💬 대화형 AI 멘토 (React Agent)

CLI 상에서 실시간으로 스트리밍되는 답변을 받을 수 있습니다.

- "인공지능 배우려면 무슨 과 가야 해?" (학과 추천)
- "서울대학교 입시 정보 알려줘." (입시 정보 검색)

### 2) 🎓 전공 추천 모드

CLI에서 `major` 명령어를 입력하여 전공 추천 전용 모드를 실행할 수 있습니다.

- 사용자의 관심사/흥미를 입력받아 Pinecone 벡터 검색을 수행하고 적합한 전공을 추천합니다.

---

## 7. 설치 및 실행 (Setup & Usage)

### 1. 환경 설정 (.env)

프로젝트 루트에 `.env` 파일을 생성하고 다음 정보를 입력합니다.

```ini
# OpenAI
OPENAI_API_KEY=sk-...

# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=majors-index

# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DB=unigo_db
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. CLI 실행

```bash
python cli.py
```

- **일반 대화**: 프롬프트에 질문을 입력하여 대화합니다.
- **전공 추천**: `major` 입력 시 추천 모드로 진입합니다.
- **종료**: `exit` 또는 `quit` 입력.

---

## 8. 한 줄 회고 (Original)

> _이전 웹 프로젝트 버전을 진행하며 남긴 회고입니다._

- **강지완**: "단순 크롤링의 한계를 넘어 API와 정제된 데이터를 통합하는 Pipeline을 구축했습니다. 배포 경험을 통해 인프라 이해도를 높였습니다."
- **김진**: "사용자의 의도를 파악하는 툴 호출 구조와 요약 기능을 통해 사용자 경험 중심의 개발을 훈련했습니다."
- **마한성**: "데이터 정형화의 중요성과 전체 아키텍처를 아우르는 풀스택 개발 경험을 얻었습니다."
- **오하원**: "여러 번의 피벗과 기술 스택 변경에도 팀원들과 협업하며 유연하게 대처하는 경험을 쌓았습니다."
