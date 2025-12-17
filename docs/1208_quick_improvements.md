# Unigo 프로젝트 빠른 개선 가이드

**작성일**: 2025-12-08  
**대상**: 즉시 적용 가능한 개선 사항

---

## 🎯 핵심 요약

현재 Unigo 프로젝트는 **일반적인 Django-LangGraph-RAG 시스템과 비교했을 때 약 70-80% 수준**으로 잘 구현되어 있습니다.

### ✅ 잘된 점
- Backend/Frontend 분리
- LangGraph 구조
- 독창적인 차등 점수 시스템
- 문서화

### ⚠️ 개선 필요
- 데이터베이스 미활용
- 비동기 처리 부재
- 테스트 코드 없음
- 보안 강화 필요

---

## 🚀 즉시 적용 가능한 개선 사항 (Top 5)

### 1. 대화 기록 저장 (데이터베이스 모델 추가)

**소요 시간**: 4-6시간  
**난이도**: ⭐⭐☆☆☆

**구현**:

```python
# unigo/unigo_app/models.py
from django.db import models
from django.contrib.auth.models import User

class Conversation(models.Model):
    """대화 세션"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Message(models.Model):
    """개별 메시지"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]
    
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['created_at']
```

**마이그레이션**:
```bash
cd unigo
python manage.py makemigrations
python manage.py migrate
```

**views.py 수정**:
```python
import uuid
from .models import Conversation, Message

@csrf_exempt
def chat_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            message = data.get("message")
            session_id = data.get("session_id") or str(uuid.uuid4())
            
            # 대화 세션 가져오기 또는 생성
            conversation, created = Conversation.objects.get_or_create(
                session_id=session_id
            )
            
            # 사용자 메시지 저장
            Message.objects.create(
                conversation=conversation,
                role='user',
                content=message
            )
            
            # 기존 대화 기록 불러오기
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in conversation.messages.all()
            ]
            
            # AI 응답 생성
            response_content = run_mentor(
                question=message,
                chat_history=history[:-1],  # 현재 메시지 제외
                mode="react"
            )
            
            # AI 응답 저장
            Message.objects.create(
                conversation=conversation,
                role='assistant',
                content=str(response_content)
            )
            
            return JsonResponse({
                "response": str(response_content),
                "session_id": session_id
            })
            
        except Exception as e:
            print(f"Error in chat_api: {e}")
            return JsonResponse({"error": str(e)}, status=500)
```

**효과**:
- ✅ 대화 기록 영구 저장
- ✅ 세션 관리 가능
- ✅ 분석 데이터 확보

---

### 2. 로깅 시스템 구축

**소요 시간**: 2-3시간  
**난이도**: ⭐☆☆☆☆

**구현**:

```python
# unigo/unigo/settings.py
import os

# logs 디렉토리 생성
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOGS_DIR, 'unigo.log'),
            'maxBytes': 1024*1024*15,  # 15MB
            'backupCount': 10,
            'formatter': 'verbose',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
        'unigo_app': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}
```

**views.py에서 사용**:
```python
import logging

logger = logging.getLogger('unigo_app')

@csrf_exempt
def chat_api(request):
    logger.info(f"Chat API called")
    
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            message = data.get("message")
            
            logger.debug(f"User message: {message}")
            
            response_content = run_mentor(
                question=message,
                chat_history=[],
                mode="react"
            )
            
            logger.info(f"Response generated successfully")
            return JsonResponse({"response": str(response_content)})
            
        except Exception as e:
            logger.error(f"Error in chat_api: {e}", exc_info=True)
            return JsonResponse({"error": str(e)}, status=500)
```

**효과**:
- ✅ 에러 추적 용이
- ✅ 디버깅 효율 향상
- ✅ 운영 모니터링 가능

---

### 3. static_pages 디렉토리 정리

**소요 시간**: 1-2시간  
**난이도**: ⭐☆☆☆☆

**확인 사항**:
1. `static_pages/`가 실제로 사용되는지 확인
2. `unigo/templates/`와 중복되는지 확인

**조치**:

```bash
# 1. 사용되지 않는다면 삭제
# git rm -rf static_pages/

# 2. Figma 프로토타입이라면 docs로 이동
# mkdir docs/design
# mv static_pages/* docs/design/

# 3. 실제 사용된다면 README에 용도 명시
```

**README.md 업데이트**:
```markdown
## 📁 프로젝트 구조

### static_pages/
- **용도**: [명확한 용도 작성]
- **unigo/templates/와의 차이점**: [차이점 작성]
```

**효과**:
- ✅ 프로젝트 구조 명확화
- ✅ 코드 중복 제거
- ✅ 유지보수 효율 향상

---

### 4. 에러 핸들링 개선

**소요 시간**: 3-4시간  
**난이도**: ⭐⭐☆☆☆

**구현**:

```python
# unigo/unigo_app/exceptions.py (새 파일)
class UnigoException(Exception):
    """Unigo 기본 예외"""
    pass

class LLMError(UnigoException):
    """LLM 호출 실패"""
    pass

class VectorSearchError(UnigoException):
    """벡터 검색 실패"""
    pass

class InvalidInputError(UnigoException):
    """잘못된 입력"""
    pass
```

```python
# unigo/unigo_app/views.py
from .exceptions import LLMError, InvalidInputError
import logging

logger = logging.getLogger('unigo_app')

@csrf_exempt
def chat_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            message = data.get("message", "").strip()
            
            # 입력 검증
            if not message:
                raise InvalidInputError("메시지가 비어있습니다.")
            
            if len(message) > 1000:
                raise InvalidInputError("메시지가 너무 깁니다. (최대 1000자)")
            
            # LLM 호출
            try:
                response_content = run_mentor(
                    question=message,
                    chat_history=[],
                    mode="react"
                )
            except Exception as e:
                logger.error(f"LLM error: {e}", exc_info=True)
                raise LLMError("AI 응답 생성에 실패했습니다. 잠시 후 다시 시도해주세요.")
            
            return JsonResponse({"response": str(response_content)})
            
        except InvalidInputError as e:
            logger.warning(f"Invalid input: {e}")
            return JsonResponse({"error": str(e)}, status=400)
        
        except LLMError as e:
            logger.error(f"LLM error: {e}")
            return JsonResponse({"error": str(e)}, status=503)
        
        except json.JSONDecodeError:
            logger.warning("Invalid JSON")
            return JsonResponse({"error": "잘못된 요청 형식입니다."}, status=400)
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return JsonResponse({"error": "서버 오류가 발생했습니다."}, status=500)
    
    return JsonResponse({"error": "Method not allowed"}, status=405)
```

**효과**:
- ✅ 명확한 에러 메시지
- ✅ 사용자 경험 개선
- ✅ 디버깅 효율 향상

---

### 5. 환경 변수 관리 개선

**소요 시간**: 1시간  
**난이도**: ⭐☆☆☆☆

**구현**:

```bash
# .env.example 파일 생성 (Git에 포함)
```

```env
# .env.example
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# LLM 설정
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o-mini

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_environment_here
PINECONE_INDEX_NAME=your_index_name_here

# Django
SECRET_KEY=your_secret_key_here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database (선택)
# DATABASE_URL=postgresql://user:password@localhost:5432/unigo

# Redis (선택)
# REDIS_URL=redis://localhost:6379/0
```

**settings.py 수정**:
```python
# unigo/unigo/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-default-key-change-this')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'True') == 'True'

ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
```

**README.md 업데이트**:
```markdown
### 2. 환경 변수 설정

`.env.example`을 복사하여 `.env` 파일 생성:

```bash
cp .env.example .env
```

`.env` 파일을 열어 실제 API 키로 수정:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
```
```

**효과**:
- ✅ 보안 강화
- ✅ 팀원 온보딩 용이
- ✅ 환경별 설정 관리

---

## 📊 개선 효과 예상

| 개선 사항 | 소요 시간 | 효과 | 우선순위 |
|----------|----------|------|---------|
| 대화 기록 저장 | 4-6시간 | ⭐⭐⭐⭐⭐ | 🔴 높음 |
| 로깅 시스템 | 2-3시간 | ⭐⭐⭐⭐☆ | 🔴 높음 |
| static_pages 정리 | 1-2시간 | ⭐⭐⭐☆☆ | 🔴 높음 |
| 에러 핸들링 | 3-4시간 | ⭐⭐⭐⭐☆ | 🔴 높음 |
| 환경 변수 관리 | 1시간 | ⭐⭐⭐☆☆ | 🔴 높음 |
| **총계** | **11-16시간** | **약 2일** | - |

---

## 🎯 실행 순서 권장

### Day 1 (오전)
1. ✅ 환경 변수 관리 개선 (1시간)
2. ✅ 로깅 시스템 구축 (2-3시간)

### Day 1 (오후)
3. ✅ 에러 핸들링 개선 (3-4시간)

### Day 2 (오전)
4. ✅ 대화 기록 저장 (4-6시간)

### Day 2 (오후)
5. ✅ static_pages 정리 (1-2시간)
6. ✅ 테스트 및 문서 업데이트

---

## 📝 체크리스트

### 개선 전 확인
- [ ] 현재 코드 백업 (Git commit)
- [ ] 가상환경 활성화
- [ ] 의존성 설치 확인

### 개선 후 확인
- [ ] 기존 기능 정상 작동 확인
- [ ] 로그 파일 생성 확인
- [ ] 데이터베이스 마이그레이션 성공
- [ ] README.md 업데이트
- [ ] Git commit 및 push

---

## 🔗 관련 문서

- [상세 아키텍처 검토](./architecture_review.md)
- [프로젝트 계획](./plans.md)
- [실행 가이드](./guide.md)

---

**작성자**: AI Assistant  
**최종 업데이트**: 2025-12-08
