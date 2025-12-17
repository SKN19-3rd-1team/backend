# Unigo 챗봇 실행 가이드

**최종 업데이트**: 2025-12-10

이 문서는 Unigo 대학 전공 추천 챗봇을 로컬 환경에서 실행하는 방법을 안내합니다.

## 📋 목차

1. [사전 요구사항](#사전-요구사항)
2. [설치 및 설정](#설치-및-설정)
3. [Django 서버 실행](#django-서버-실행)
4. [사용 방법](#사용-방법)
5. [문제 해결](#문제-해결)

## 🔧 사전 요구사항

### 필수 소프트웨어

- **Python 3.10 이상**
- **pip** (Python 패키지 관리자)
- **Git** (저장소 클론용)

### 필수 API 키

- **OpenAI API Key**: GPT 모델 및 임베딩 사용

## 📦 설치 및 설정

### 1. 저장소 클론

```bash
git clone <repository-url>
cd unigo
```

### 2. Python 가상환경 생성 (권장)

```bash
# Windows
conda create -n unigo python=3.11
conda activate unigo
```

### 3. 의존성 설치

```bash
# Backend 의존성 설치
pip install -r requirements.txt

# Django 설치
pip install django
```

**주요 패키지**:

- `langchain`: LLM 오케스트레이션
- `langgraph`: 상태 기반 에이전트
- `openai`: OpenAI API 클라이언트
- `pinecone-client`: 벡터 DB 클라이언트
- `langchain-pinecone`: LangChain용 Pinecone 통합
- `python-dotenv`: 환경 변수 관리
- `django`: 웹 프레임워크

### 4. 환경 변수 설정

#### 4.1 .env 파일 생성

프로젝트 루트에 `.env.example` 파일을 `.env`로 복사하고 실제 값을 입력하세요:

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux/Mac
cp .env.example .env
```

#### 4.2 필수 설정 입력

`.env` 파일을 열어 다음 항목을 설정하세요:

```env
# ============================================
# Project Configuration
# ============================================
# 프로젝트 루트 경로 (backend 모듈 import를 위해 필요)
# ⚠️ 반드시 본인의 실제 경로로 변경하세요!
PROJECT_ROOT=C:\Users\user\github\frontend  # Windows
# PROJECT_ROOT=/home/user/github/frontend  # Linux/Mac

# ============================================
# API Keys
# ============================================
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Langchain & LangSmith
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key_here
```

**중요**:

- `PROJECT_ROOT`는 **반드시 본인의 실제 프로젝트 경로**로 변경하세요
- Windows에서는 백슬래시(`\`)를 사용합니다
- `.env` 파일은 `.gitignore`에 포함되어 Git에 커밋되지 않습니다

#### 4.3 python-dotenv 설치

`.env` 파일을 자동으로 로드하기 위해 `python-dotenv`를 설치하세요:

```bash
pip install python-dotenv
```

**참고**: `requirements.txt`에 이미 포함되어 있으므로, `pip install -r requirements.txt`를 실행했다면 별도 설치가 필요 없습니다.

### 5. 데이터베이스 설정 (MySQL)

프로젝트 실행을 위해 MySQL 데이터베이스 설정과 초기 데이터 적재가 필요합니다.

```bash
# 1. Django 마이그레이션 파일 생성 (데이터베이스 변경 사항 반영)
cd unigo
python manage.py makemigrations

# 2. Django 마이그레이션 적용 (테이블 생성)
python manage.py migrate

# 3. 초기 데이터 적재 (전공, 카테고리, 대학 정보 통합 적재)
# 프로젝트 루트에서 실행 -> DB 테이블에 데이터 삽입
cd ..
python -m backend.db.seed_all

### 4.2. 벡터 DB(Pinecone) 적재
검색 품질을 높이기 위해 전공 상세 정보와 **학과 대분류(표준 학과명)**을 벡터화하여 Pinecone에 저장합니다.

```bash
# 1. 전공 상세 정보 (Major Detail) 임베딩 -> 'majors' namespace
python backend/scripts/ingest_university_majors.py

# 2. [New] 학과 대분류 (Major Categories) 임베딩 -> 'major_categories' namespace
# (학과 검색 시 '컴퓨터' -> '소프트웨어' 등 의미 기반 자동 확장을 위해 필수)
python backend/scripts/ingest_major_categories.py
```

와 `PINECONE_API_KEY`가 `.env`에 올바르게 설정되어 있어야 합니다.

```bash
# RAG용 Pinecone 인덱스 구축 (자동 생성 및 데이터 업로드)
# DB에 적재된 전공 데이터를 기반으로 임베딩을 생성합니다.
python -m backend.rag.build_major_index
```

정상적으로 완료되면 "✅ Indexing complete!" 메시지가 표시됩니다.

## 🚀 Django 서버 실행

### 기본 실행

```bash
cd unigo
python manage.py runserver
```

서버가 시작되면 다음과 같은 메시지가 표시됩니다:

```
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

### 다른 포트로 실행

```bash
python manage.py runserver 8001
```

### 외부 접속 허용

```bash
python manage.py runserver 0.0.0.0:8000
```

**주의**: 개발 서버는 프로덕션 환경에서 사용하지 마세요.

## 💬 사용 방법

### 1. 웹 브라우저 접속

서버 실행 후 브라우저에서 다음 URL로 접속:

```
http://127.0.0.1:8000/chat/
```

### 2. 온보딩 질문 답변

처음 접속하여 "추천 시작"을 입력하면 7가지 온보딩 질문이 순차적으로 표시됩니다:

1. **선호 교과목 (Subjects)**: 가장 자신 있거나 흥미로운 고교 과목
   - 예: "수학과 물리를 잘하고 과학 실험을 좋아합니다."

2. **관심사 및 활동 (Interests)**: 학교 밖 취미나 동아리 활동
   - 예: "코딩 동아리, 역사 소설 읽기, 유튜브 영상 편집"

3. **관심 활동 유형 (Activity Type)**: 향후 하고 싶은 일의 스타일
   - 예: "남을 돕는 일, 데이터를 분석해서 문제를 해결하는 일"

4. **선호 환경 및 성향 (Environment)**: 보람을 느끼는 상황
   - 예: "혼자 조용히 깊게 생각할 때, 팀원들과 함께 목표를 달성했을 때"

5. **중요하게 생각하는 가치 (Values)**: 직업 선택 시 우선순위
   - 예: "안정적인 삶, 높은 연봉, 새로운 도전, 사회적 기여"

6. **평소 관심 주제 (Topics)**: 유튜브/뉴스 등에서 즐겨보는 주제
   - 예: "우주 다큐멘터리, 최신 IT 기기 리뷰, 심리 테스트"

7. **학습 스타일 (Learning Style)**: 이론 vs 실습 선호도
   - 예: "원리는 책으로 배우는 게 좋아요, 직접 해봐야 직성이 풀려요"

### 3. 전공 추천 확인

온보딩 완료 후 AI가 분석한 추천 전공 TOP 5가 표시됩니다:

- 채팅 창에 요약 표시
- 우측 패널에 상세 정보 표시

### 4. 추가 질문

온보딩 이후에는 자유롭게 질문할 수 있습니다:

**전공 정보 질문 예시**:

- "컴퓨터공학과에 대해 알려줘"
- "기계공학과 졸업 후 연봉은 얼마야?"
- "심리학과에서는 주로 뭘 배워?"

**대학 정보 질문 예시**:

- "컴퓨터공학과가 있는 대학 어디야?"
- "서울에 있는 간호학과 알려줘"

**입시 정보 질문 예시**:

- "서울대학교 컴퓨터공학과 정시컷 알려줘"
- "연세대학교 수시컷이 궁금해"

### 5. 채팅 관리

- **새 채팅 시작**: 사이드바의 "새 채팅" 버튼을 클릭하면 대화 기록이 초기화되고 새로운 온보딩/대화를 시작할 수 있습니다.
- **로그아웃**: 사이드바 하단의 "로그아웃" 버튼을 클릭하면 세션이 종료됩니다.

## 🔍 주요 기능

### 차등 점수 시스템

사용자가 명시한 희망 전공에 대해 정확도에 따라 차등 점수를 부여합니다:

| 티어 | 점수 | 설명 |
|------|------|------|
| Tier 1 | 20.0 | 정확히 일치 |
| Tier 2 | 15.0 | 접두어 일치 |
| Tier 3 | 10.0 | 포함 |
| Tier 4 | 5.0 | 벡터 유사도 |

### Markdown 링크 지원

챗봇이 제공하는 입시 정보 링크는 클릭 가능합니다:

- `[한양대학교 입시정보](URL)` → 클릭 가능한 링크로 표시

### 세션 유지

- 대화 기록은 브라우저 세션에 저장됩니다
- 페이지를 새로고침해도 대화가 유지됩니다
- 온보딩 상태도 세션에 저장됩니다

## 🐛 문제 해결

### 1. ModuleNotFoundError: No module named 'backend'

**원인**: Python이 backend 모듈을 찾지 못함

**해결 방법**:

```bash
# Windows
set PYTHONPATH=%PYTHONPATH%;C:\path\to\frontend

# Linux/Mac
export PYTHONPATH=$PYTHONPATH:/path/to/frontend
```

또는 Django 프로젝트 내에서 이미 처리되어 있으므로, `unigo/` 디렉토리에서 실행하세요.

### 2. OpenAI API Error

**원인**: API 키가 설정되지 않았거나 잘못됨

**해결 방법**:

1. `.env` 파일에 올바른 `OPENAI_API_KEY` 설정
2. API 키에 충분한 크레딧이 있는지 확인
3. 서버 재시작

### 3. Port already in use

**원인**: 8000 포트가 이미 사용 중

**해결 방법**:

```bash
# 다른 포트 사용
python manage.py runserver 8001

# 또는 기존 프로세스 종료 (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### 4. 마이그레이션 경고

```
You have 18 unapplied migration(s)...
```

**해결 방법**:

```bash
cd unigo
python manage.py migrate
```

### 5. Pinecone 연결 오류

**원인**: Pinecone API 키 또는 인덱스 설정 문제

**해결 방법**:

1. `.env`에 `PINECONE_API_KEY` 설정 확인
2. 인덱스 이름 확인
3. 로컬 벡터 DB 사용 (Pinecone 없이도 작동 가능)

### 6. 채팅 응답이 느린 경우

**원인**: OpenAI API 응답 지연 또는 벡터 검색 시간

**해결 방법**:

- 정상적인 현상입니다 (보통 3-10초)
- 더 빠른 모델 사용 (`gpt-3.5-turbo`)
- 네트워크 연결 확인

### 7. ImportError: name 'run_major_recommendation' is not defined

**원인**: Python이 `backend` 모듈을 찾지 못해 import 실패

**증상**:

```
Error in onboarding_api: name 'run_major_recommendation' is not defined
Internal Server Error: /api/onboarding
```

**해결 방법**:

**방법 1 - .env 파일에서 PROJECT_ROOT 설정 (가장 권장)**:

1. `.env` 파일을 열어 `PROJECT_ROOT` 설정:

```env
# Windows
PROJECT_ROOT=C:\Users\user\github\frontend

# Linux/Mac
PROJECT_ROOT=/home/user/github/frontend
```

2. `python-dotenv` 설치 확인:

```bash
pip install python-dotenv
```

3. 서버 실행:

```bash
cd unigo
python manage.py runserver
```

서버 시작 시 다음 메시지가 표시되면 성공:

```
✅ Loaded environment variables from: C:\Users\user\github\frontend\.env
✅ Added to PYTHONPATH: C:\Users\user\github\frontend
```

**방법 2 - PYTHONPATH 환경 변수 임시 설정**:

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "$env:PYTHONPATH;C:\Users\user\github\frontend"
python manage.py runserver

# Windows (CMD)
set PYTHONPATH=%PYTHONPATH%;C:\Users\user\github\frontend
python manage.py runserver

# Linux/Mac
export PYTHONPATH=$PYTHONPATH:/path/to/frontend
python manage.py runserver
```

**방법 3 - 환경 변수 영구 설정 (Windows)**:

1. "시스템 환경 변수 편집" 검색
2. "환경 변수" 클릭
3. 사용자 변수에서 "새로 만들기"
4. 변수 이름: `PYTHONPATH`
5. 변수 값: `C:\Users\user\github\frontend`
6. 터미널 재시작 후 서버 실행

## 📊 로그 확인

### Django 서버 로그

서버 실행 중 터미널에 다음과 같은 로그가 표시됩니다:

```
[Majors] ✅ Pinecone search returned 50 hits
🤖 LLM Normalized Majors: ['컴퓨터공학과'] -> ['컴퓨터공학과']
🔍 Searching for preferred major: '컴퓨터공학과'
🎯 Set '컴퓨터공학과' score to 20.00 [Tier 1 (Exact Match)]
```

### DB 쿼리 로그 (New)

`unigo/settings.py` 설정에 따라 실행되는 SQL 쿼리가 로그에 기록됩니다.

```bash
# 로그 파일 확인
tail -f unigo/logs/unigo.log
```

로그 예시:

```
(0.001) SELECT ... FROM ...
```

### 브라우저 콘솔

개발자 도구(F12)의 콘솔에서 프론트엔드 로그 확인 가능

## 🔄 서버 재시작

코드 변경 후 Django 개발 서버는 자동으로 재시작됩니다.

**수동 재시작이 필요한 경우**:

- `.env` 파일 변경
- 새로운 Python 패키지 설치
- Django 설정 파일 변경

```bash
# Ctrl+C로 서버 중지 후
python manage.py runserver
```

## ✅ 변경 사항 검증 (Test)

주요 로직 변경(예: 검색 로직 수정) 후에는 제공된 테스트 스크립트를 사용하여 기능을 검증할 수 있습니다.

```bash
# LLM 툴 및 검색 로직 검증
python test_llm.py
```

## 📝 추가 정보

### API 엔드포인트 테스트

Postman이나 curl로 API를 직접 테스트할 수 있습니다:

```bash
# 채팅 API
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "컴퓨터공학과 알려줘", "history": []}'

# 온보딩 API
curl -X POST http://127.0.0.1:8000/api/onboarding \
  -H "Content-Type: application/json" \
  -d '{"answers": {"subjects": "수학", "interests": "코딩", "desired_salary": "5000만원", "preferred_majors": "컴퓨터공학과"}}'
```

### 개발 모드 vs 프로덕션

현재 설정은 **개발 모드**입니다. 프로덕션 배포 시:

- `DEBUG = False` 설정
- `ALLOWED_HOSTS` 설정
- 정적 파일 수집 (`collectstatic`)
- WSGI 서버 사용 (Gunicorn, uWSGI 등)

---

**도움이 필요하신가요?**

- [개발 계획](plans.md) 참고
- [수정 로그](fixed_log.md) 참고
- [README](../README.md) 참고
