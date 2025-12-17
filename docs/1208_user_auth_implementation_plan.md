# 사용자 인증 및 대화 기록 저장 구현 계획

**작성일**: 2025-12-08  
**목표**: 사용자 회원가입/로그인 시스템 및 대화 기록 저장 기능 구현

---

## 📋 구현 단계

### Phase 1: 데이터베이스 모델 설계 ✅
- [x] User 모델 (Django 기본 제공)
- [ ] Conversation 모델 (대화 세션)
- [ ] Message 모델 (개별 메시지)
- [ ] MajorRecommendation 모델 (전공 추천 결과)
- [ ] Major 모델 (전공 정보 - backend/data에서 마이그레이션)
- [ ] University 모델 (대학 정보)

### Phase 2: 사용자 인증 시스템
- [ ] 회원가입 API (`/api/auth/signup`)
- [ ] 로그인 API (`/api/auth/login`)
- [ ] 로그아웃 API (`/api/auth/logout`)
- [ ] 사용자 정보 조회 API (`/api/auth/me`)
- [ ] 프론트엔드 회원가입/로그인 UI

### Phase 3: 세션 관리
- [ ] 로그인 사용자 세션 관리
- [ ] 비로그인 사용자 임시 세션 (session_id)
- [ ] 세션 기반 대화 기록 조회

### Phase 4: 대화 기록 저장
- [ ] 로그인 사용자 대화 저장
- [ ] 비로그인 사용자 임시 대화 저장
- [ ] 대화 기록 조회 API
- [ ] 대화 기록 삭제 API

### Phase 5: 전공 데이터 마이그레이션
- [ ] JSON 데이터를 DB로 마이그레이션하는 스크립트
- [ ] Major, University 모델에 데이터 적재
- [ ] 기존 Pinecone 검색과 DB 조회 통합

### Phase 6: 프론트엔드 통합
- [ ] 로그인 상태 표시
- [ ] 대화 기록 목록 UI
- [ ] 이전 대화 불러오기 기능

---

## 🗄️ 데이터베이스 스키마

### 1. User (Django 기본 제공)
```python
- id: PK
- username: 사용자명 (unique)
- email: 이메일 (unique)
- password: 비밀번호 (해시)
- first_name: 이름
- last_name: 성
- is_active: 활성 여부
- date_joined: 가입일
```

### 2. Conversation (대화 세션)
```python
- id: PK
- user: FK(User, null=True)  # 로그인 사용자
- session_id: 세션 ID (unique)  # 비로그인 사용자용
- title: 대화 제목 (첫 메시지에서 자동 생성)
- created_at: 생성일
- updated_at: 수정일
```

### 3. Message (개별 메시지)
```python
- id: PK
- conversation: FK(Conversation)
- role: 역할 ('user' | 'assistant' | 'system')
- content: 메시지 내용
- metadata: JSON (tool 호출 정보 등)
- created_at: 생성일
```

### 4. MajorRecommendation (전공 추천 결과)
```python
- id: PK
- user: FK(User, null=True)
- session_id: 세션 ID
- onboarding_answers: JSON (온보딩 답변)
- recommended_majors: JSON (추천 결과)
- created_at: 생성일
```

### 5. Major (전공 정보)
```python
- id: PK
- name: 전공명
- cluster: 계열
- summary: 요약
- interest: 관심사
- property: 특성
- salary: 평균 연봉
- employment_rate: 취업률
- relate_subjects: JSON (관련 과목)
- career_activities: JSON (진로 활동)
- jobs: TEXT (진출 직업)
- qualifications: TEXT (자격증)
- enter_fields: JSON (진출 분야)
- main_subjects: JSON (주요 과목)
- chart_data: JSON (통계 데이터)
```

### 6. University (대학 정보)
```python
- id: PK
- name: 대학명
- area: 지역
- campus_name: 캠퍼스명
- school_url: 홈페이지 URL
```

### 7. MajorUniversity (전공-대학 연결)
```python
- id: PK
- major: FK(Major)
- university: FK(University)
- major_name: 해당 대학에서의 전공명
```

---

## 🔐 인증 방식

### 선택: Session 기반 인증 (Django 기본)
- **장점**: Django 기본 제공, 구현 간단
- **단점**: 확장성 제한 (모바일 앱 등)

### 대안: JWT 토큰 기반 인증
- **장점**: Stateless, 확장성 좋음
- **단점**: 구현 복잡도 증가

**결정**: Phase 1에서는 Session 기반, 추후 JWT로 전환 가능

---

## 📝 API 엔드포인트 설계

### 인증 API
```
POST   /api/auth/signup      # 회원가입
POST   /api/auth/login       # 로그인
POST   /api/auth/logout      # 로그아웃
GET    /api/auth/me          # 현재 사용자 정보
```

### 대화 API
```
GET    /api/conversations/              # 대화 목록
POST   /api/conversations/              # 새 대화 생성
GET    /api/conversations/{id}/         # 대화 상세
DELETE /api/conversations/{id}/         # 대화 삭제
GET    /api/conversations/{id}/messages # 메시지 목록
```

### 기존 API 수정
```
POST   /api/chat         # session_id 추가, 대화 저장
POST   /api/onboarding   # user_id 또는 session_id 추가
```

---

## 🚀 구현 순서 (우선순위)

### Day 1: 데이터베이스 모델 및 마이그레이션
1. ✅ models.py에 모델 정의
2. ✅ 마이그레이션 생성 및 실행
3. ✅ Admin 페이지에 모델 등록

### Day 2: 사용자 인증 API
1. ✅ 회원가입 API
2. ✅ 로그인 API
3. ✅ 로그아웃 API
4. ✅ 사용자 정보 조회 API

### Day 3: 대화 기록 저장 기능
1. ✅ chat_api 수정 (대화 저장)
2. ✅ onboarding_api 수정 (추천 결과 저장)
3. ✅ 대화 목록 조회 API
4. ✅ 대화 상세 조회 API

### Day 4: 프론트엔드 통합
1. ✅ 회원가입/로그인 UI
2. ✅ 로그인 상태 표시
3. ✅ 대화 기록 목록 UI
4. ✅ 이전 대화 불러오기

### Day 5: 전공 데이터 마이그레이션
1. ✅ JSON → DB 마이그레이션 스크립트
2. ✅ 데이터 적재 및 검증
3. ✅ 기존 코드와 통합

---

## 💾 전공 데이터 마이그레이션 전략

### 문제점
- `backend/data/major_detail.json`: 11MB, 매우 큰 파일
- 현재는 Pinecone 벡터 DB만 사용
- 전공 상세 정보는 JSON 파일에서 직접 읽음

### 해결 방안

#### 옵션 1: 전체 마이그레이션 (권장)
- JSON 데이터를 PostgreSQL/MySQL로 마이그레이션
- 장점: 쿼리 성능, 데이터 무결성, 관계 관리
- 단점: 초기 마이그레이션 시간 소요

#### 옵션 2: 하이브리드 (현재 유지 + 부분 마이그레이션)
- Pinecone: 벡터 검색 (유지)
- JSON: 전공 상세 정보 (유지)
- DB: 사용자, 대화 기록만 저장
- 장점: 빠른 구현
- 단점: 데이터 분산

#### 옵션 3: 점진적 마이그레이션
- Phase 1: 사용자/대화 기록만 DB
- Phase 2: 자주 조회되는 전공 정보만 DB
- Phase 3: 전체 마이그레이션

**결정**: 옵션 1 (전체 마이그레이션) - 장기적으로 유리

---

## 🔄 마이그레이션 스크립트 구조

```python
# unigo/unigo_app/management/commands/load_major_data.py

from django.core.management.base import BaseCommand
import json
from unigo_app.models import Major, University, MajorUniversity

class Command(BaseCommand):
    help = 'Load major data from JSON to database'
    
    def handle(self, *args, **options):
        # 1. JSON 파일 읽기
        # 2. Major 모델에 저장
        # 3. University 모델에 저장
        # 4. MajorUniversity 연결
        # 5. 진행 상황 출력
```

**실행**:
```bash
python manage.py load_major_data
```

---

## 📊 예상 소요 시간

| 단계 | 소요 시간 | 난이도 |
|------|----------|--------|
| 모델 설계 및 마이그레이션 | 4-6시간 | ⭐⭐⭐☆☆ |
| 인증 API 구현 | 6-8시간 | ⭐⭐⭐⭐☆ |
| 대화 기록 저장 | 4-6시간 | ⭐⭐⭐☆☆ |
| 프론트엔드 UI | 8-10시간 | ⭐⭐⭐⭐☆ |
| 데이터 마이그레이션 | 6-8시간 | ⭐⭐⭐⭐⭐ |
| **총계** | **28-38시간** | **약 5-7일** |

---

## 🎯 다음 단계

1. ✅ 이 계획서 검토 및 승인
2. ✅ Phase 1 시작: 모델 정의
3. ⏳ Phase 2: 인증 API 구현
4. ⏳ Phase 3: 대화 기록 저장
5. ⏳ Phase 4: 프론트엔드 통합
6. ⏳ Phase 5: 데이터 마이그레이션

---

**작성자**: AI Assistant  
**검토 필요**: 프로젝트 팀  
**시작일**: 2025-12-08
