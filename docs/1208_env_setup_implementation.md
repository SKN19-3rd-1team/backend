# .env 기반 PYTHONPATH 설정 구현 완료

**작업 일시**: 2025-12-08

## 📋 작업 요약

`.env` 파일을 활용하여 `PYTHONPATH`를 자동으로 설정하는 시스템을 구현했습니다. 이제 사용자는 환경 변수를 수동으로 설정할 필요 없이 `.env` 파일만 수정하면 됩니다.

## ✅ 구현 내용

### 1. `.env` 파일 업데이트
- **`PROJECT_ROOT`** 환경 변수 추가
- 섹션별로 구분하여 가독성 향상
- 각 설정에 대한 명확한 주석 추가

**위치**: `c:\Users\minek\github\frontend\.env`

```env
# ============================================
# Project Configuration
# ============================================
# 프로젝트 루트 경로 (backend 모듈 import를 위해 필요)
PROJECT_ROOT=C:\Users\minek\github\frontend
```

### 2. `.env.example` 파일 생성
- 새로운 환경에서 프로젝트를 설정할 때 사용할 템플릿 파일
- 실제 API 키 대신 플레이스홀더 사용
- 모든 필수 및 선택 환경 변수 포함

**위치**: `c:\Users\minek\github\frontend\.env.example`

### 3. `manage.py` 자동 로딩 구현
Django 서버 시작 시 자동으로 다음 작업 수행:

1. `.env` 파일에서 환경 변수 로드
2. `PROJECT_ROOT`를 `sys.path`에 자동 추가
3. 성공/실패 메시지 출력

**위치**: `c:\Users\minek\github\frontend\unigo\manage.py`

**출력 예시**:
```
✅ Loaded environment variables from: C:\Users\user\github\frontend\.env
✅ Added to PYTHONPATH: C:\Users\user\github\frontend
```

### 4. `views.py` 에러 메시지 개선
Import 실패 시 `.env` 파일 설정 방법을 우선적으로 안내

**위치**: `c:\Users\minek\github\frontend\unigo\unigo_app\views.py`

### 5. `.gitignore` 업데이트
- `.env` 파일을 다시 gitignore에 추가 (민감한 정보 보호)
- `.env.example`은 추적되도록 설정

**위치**: `c:\Users\minek\github\frontend\.gitignore`

### 6. 문서 업데이트

#### `guide.md`
- 환경 변수 설정 섹션을 `.env` 기반으로 전면 개편
- ImportError 해결 방법을 `.env` 우선으로 재구성
- 단계별 설정 가이드 추가

**위치**: `c:\Users\minek\github\frontend\docs\guide.md`

#### `README.md`
- 빠른 시작 섹션에 `.env.example` 사용법 추가
- 문제 해결 섹션에 `.env` 기반 해결 방법 추가

**위치**: `c:\Users\minek\github\frontend\README.md`

## 🚀 사용 방법

### 새로운 환경에서 프로젝트 설정

1. **저장소 클론**:
```bash
git clone <repository-url>
cd frontend
```

2. **의존성 설치**:
```bash
pip install -r requirements.txt
```

3. **`.env` 파일 생성**:
```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux/Mac
cp .env.example .env
```

4. **`.env` 파일 수정**:
```env
# 본인의 실제 경로로 변경!
PROJECT_ROOT=C:\Users\user\github\frontend  # Windows
# PROJECT_ROOT=/home/user/github/frontend  # Linux/Mac

# API 키 입력
OPENAI_API_KEY=your_actual_api_key_here
```

5. **서버 실행**:
```bash
cd unigo
python manage.py runserver
```

### 성공 확인

서버 시작 시 다음 메시지가 표시되면 성공:
```
✅ Loaded environment variables from: C:\Users\user\github\frontend\.env
✅ Added to PYTHONPATH: C:\Users\user\github\frontend
Starting development server at http://127.0.0.1:8000/
```

## 🔧 기술적 세부사항

### 자동 PYTHONPATH 추가 로직

`manage.py`에서 다음 순서로 처리:

1. **`.env` 파일 탐색**: `unigo/manage.py`에서 상위 디렉토리의 `.env` 파일 찾기
2. **환경 변수 로드**: `python-dotenv`로 `.env` 파일 로드
3. **PROJECT_ROOT 확인**: 
   - `.env`에 `PROJECT_ROOT`가 설정되어 있으면 해당 경로 사용
   - 없으면 자동 감지된 프로젝트 루트 사용
4. **sys.path 추가**: 중복 확인 후 `sys.path`에 추가

### Fallback 메커니즘

- `python-dotenv`가 설치되지 않은 경우: 자동 감지된 경로 사용
- `.env` 파일이 없는 경우: 자동 감지된 경로 사용
- `PROJECT_ROOT`가 설정되지 않은 경우: 자동 감지된 경로 사용

## 📊 장점

### 이전 방식 (수동 PYTHONPATH 설정)
```bash
# 매번 서버 실행 전에 실행해야 함
set PYTHONPATH=%PYTHONPATH%;C:\Users\user\github\frontend
python manage.py runserver
```

### 새로운 방식 (.env 기반)
```bash
# .env 파일에 한 번만 설정
# PROJECT_ROOT=C:\Users\user\github\frontend

# 이후 간단하게 실행
python manage.py runserver
```

### 주요 이점

1. ✅ **한 번만 설정**: `.env` 파일에 한 번만 설정하면 영구적으로 적용
2. ✅ **환경별 관리**: 개발/테스트/프로덕션 환경별로 다른 `.env` 파일 사용 가능
3. ✅ **팀 협업 용이**: `.env.example`을 통해 필요한 설정 공유
4. ✅ **보안 강화**: `.env`는 gitignore되어 민감한 정보 보호
5. ✅ **자동 감지**: `PROJECT_ROOT`가 없어도 자동으로 경로 감지

## 🐛 문제 해결

### ImportError 발생 시

**증상**:
```
Error in onboarding_api: name 'run_major_recommendation' is not defined
```

**해결**:
1. `.env` 파일에 `PROJECT_ROOT` 설정 확인
2. `python-dotenv` 설치 확인: `pip install python-dotenv`
3. 서버 재시작

### 서버 시작 시 경고 메시지

**메시지**:
```
⚠️  .env file not found at: C:\Users\user\github\frontend\.env
⚠️  Using auto-detected project root: C:\Users\user\github\frontend
```

**의미**: `.env` 파일이 없지만 자동 감지된 경로를 사용 중 (정상 작동)

**권장**: `.env.example`을 복사하여 `.env` 파일 생성

## 📝 관련 파일

- `c:\Users\minek\github\frontend\.env` - 환경 변수 설정
- `c:\Users\minek\github\frontend\.env.example` - 환경 변수 템플릿
- `c:\Users\minek\github\frontend\unigo\manage.py` - 자동 로딩 로직
- `c:\Users\minek\github\frontend\unigo\unigo_app\views.py` - 에러 메시지
- `c:\Users\minek\github\frontend\docs\guide.md` - 사용자 가이드
- `c:\Users\minek\github\frontend\README.md` - 프로젝트 README

## 🎯 다음 단계

이제 다른 기기에서 프로젝트를 설정할 때:

1. `.env.example`을 `.env`로 복사
2. `PROJECT_ROOT`를 본인의 경로로 수정
3. API 키 입력
4. `python manage.py runserver` 실행

끝! 🎉
