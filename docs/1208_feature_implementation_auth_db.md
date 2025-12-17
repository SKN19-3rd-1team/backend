# 사용자 인증 및 DB 시스템 구현 완료 보고서

**작성일**: 2025-12-08  
**상태**: Phase 1, 2 구현 완료

---

## 1. 개요
기존 사용자가 요청한 "사용자 인증 시스템" 및 "대화 기록 저장" 기능을 구현하기 위해 Django의 인증 시스템을 도입하고 데이터베이스 모델을 설계/구현했습니다. 또한 기존 JSON 파일 기반의 데이터를 DB로 마이그레이션할 수 있는 기반을 마련했습니다.

## 2. 주요 구현 내용

### 2.1 데이터베이스 모델 설계 (`unigo_app/models.py`)
다음 6개의 주요 모델을 설계하고 구현했습니다.

1.  **Conversation**: 대화 세션 관리
    *   로그인 사용자 (`user` FK)와 비로그인 사용자 (`session_id`) 모두 지원
    *   대화 제목 및 생성/수정 시간 추적
2.  **Message**: 개별 메시지 저장
    *   `role` (user/assistant/system) 구분
    *   JSON 기반의 `metadata` 필드 지원
3.  **MajorRecommendation**: 전공 추천 결과 저장
    *   온보딩 답변(`onboarding_answers`)과 추천 결과(`recommended_majors`)를 JSON으로 저장
4.  **Major, University, MajorUniversity**: 전공 및 대학 데이터 관리
    *   `backend/data/major_detail.json` 데이터를 DB화하기 위한 스키마 정의

### 2.2 사용자 인증 API (`unigo_app/views.py`)
Django 기본 `User` 모델을 활용하여 다음 API를 구현했습니다.

*   `POST /api/auth/signup`: 회원가입 (자동 로그인 처리)
*   `POST /api/auth/login`: 로그인 (**Username 또는 Email** 모두 지원)
*   `POST /api/auth/logout`: 로그아웃
*   `GET /api/auth/me`: 현재 로그인한 사용자 정보 조회

### 2.3 프론트엔드 연동
*   **HTML 구조 복구**: `base.html`의 깨진 모달 구조를 재작성하고 정리했습니다.
*   **JS 로직 구현**: `static/js/script.js`에 실제 API 호출 로직을 추가하여 로그인/회원가입이 작동하도록 했습니다.
*   **UI 개선**: 입력 필드 라벨을 "이메일 또는 아이디"로 변경하여 사용자 혼동을 줄였습니다.

### 2.4 데이터 마이그레이션 도구
*   `load_major_data.py`: `backend/data`의 대용량 JSON 파일을 DB로 적재하는 관리자 커맨드 작성
    *   실행: `python manage.py load_major_data`

---

## 3. 변경된 프로젝트 구조

```
unigo/
├── unigo_app/
│   ├── models.py       # [New] DB 스키마 정의
│   ├── admin.py        # [New] 관리자 페이지 설정
│   ├── views.py        # [Updated] 인증 API 및 DB 연동 로직 추가
│   ├── urls.py         # [Updated] API 라우팅 추가
│   └── management/
│       └── commands/
│           └── load_major_data.py  # [New] 데이터 마이그레이션 스크립트
└── templates/
    └── unigo_app/
        ├── base.html   # [Updated] 인증 폼 수정 및 구조 복구
        └── auth.html   # [Updated] 리다이렉트 로직과 연동
```

## 4. 실행 방법

변경된 기능을 사용하기 위해서는 데이터베이스 마이그레이션이 필수적입니다.

```bash
# 1. 가상환경 활성화
conda activate langchain_env

# 2. 마이그레이션 실행
cd unigo
python manage.py makemigrations
python manage.py migrate

# 3. (옵션) 전공 데이터 적재
python manage.py load_major_data

# 4. 서버 실행
python manage.py runserver
```

## 5. 향후 계획 (Phase 3~)
*   **대화 기록 UI**: 채팅 화면 사이드바에 과거 대화 목록 표시
*   **전공 검색 DB 연동**: JSON 파일 대신 DB 쿼리를 이용한 전공 검색/필터링 구현
