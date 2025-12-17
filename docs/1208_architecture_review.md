# Unigo 프로젝트 아키텍처 검토 및 개선 방안

**작성일**: 2025-12-08  
**검토 대상**: Django 기반 LangGraph-RAG 챗봇 시스템

---

## 📋 목차

1. [현재 구조 분석](#1-현재-구조-분석)
2. [일반적인 Django-LangGraph-RAG 시스템과의 비교](#2-일반적인-django-langgraph-rag-시스템과의-비교)
3. [강점 분석](#3-강점-분석)
4. [개선이 필요한 영역](#4-개선이-필요한-영역)
5. [구체적인 개선 방안](#5-구체적인-개선-방안)
6. [우선순위별 실행 계획](#6-우선순위별-실행-계획)

---

## 1. 현재 구조 분석

### 1.1 프로젝트 구조

```
frontend/
├── backend/                    # LangGraph RAG 백엔드
│   ├── data/                   # 데이터 및 벡터 DB
│   ├── graph/                  # LangGraph 노드 및 상태
│   ├── rag/                    # RAG 시스템 (retriever, tools, vectorstore)
│   ├── api/                    # API 관련 (현재 사용 여부 불명확)
│   ├── main.py                 # 진입점
│   ├── config.py               # 설정 관리
│   └── server.py               # 서버 (용도 확인 필요)
│
├── unigo/                      # Django 웹 애플리케이션
│   ├── manage.py
│   ├── unigo/                  # 프로젝트 설정
│   │   ├── settings.py
│   │   └── urls.py
│   ├── unigo_app/              # 메인 앱
│   │   ├── views.py            # API 엔드포인트
│   │   ├── urls.py
│   │   └── models.py
│   ├── static/                 # 정적 파일
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── templates/              # Django 템플릿
│       └── unigo_app/
│
├── static_pages/               # 정적 페이지 (용도 불명확)
│   ├── chat/
│   ├── home/
│   ├── profile/
│   └── setting/
│
├── docs/                       # 프로젝트 문서
├── assets/                     # 공통 자산
└── requirements.txt
```

### 1.2 주요 컴포넌트

#### Backend (LangGraph RAG)
- **graph/**: LangGraph 노드, 상태, 그래프 빌더
- **rag/**: RAG 시스템 (Pinecone, OpenAI Embeddings, Tools)
- **main.py**: `run_mentor()`, `run_major_recommendation()` 함수 제공

#### Frontend (Django)
- **unigo_app/views.py**: API 엔드포인트 (`/api/chat`, `/api/onboarding`)
- **static/js/chat.js**: 채팅 UI 로직, 온보딩 플로우
- **templates/**: Django 템플릿 (HTML)

---

## 2. 일반적인 Django-LangGraph-RAG 시스템과의 비교

### 2.1 표준 구조 (업계 베스트 프랙티스)

```
project_root/
├── backend/                    # AI/RAG 백엔드
│   ├── agents/                 # LangGraph 에이전트
│   │   ├── agent.py           # 그래프 정의
│   │   ├── nodes.py           # 노드 함수
│   │   ├── state.py           # 상태 정의
│   │   └── tools.py           # LangChain 툴
│   ├── rag/                    # RAG 컴포넌트
│   │   ├── retriever.py
│   │   ├── embeddings.py
│   │   └── vectorstore.py
│   ├── data/                   # 데이터 및 벡터 DB
│   ├── config.py               # 설정
│   └── main.py                 # 진입점
│
├── django_app/                 # Django 웹 애플리케이션
│   ├── manage.py
│   ├── project_name/           # 프로젝트 설정
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── asgi.py/wsgi.py
│   ├── apps/                   # Django 앱들
│   │   ├── chatbot/           # 챗봇 앱
│   │   │   ├── views.py
│   │   │   ├── urls.py
│   │   │   ├── models.py
│   │   │   └── serializers.py (DRF 사용 시)
│   │   ├── users/             # 사용자 관리 앱
│   │   └── documents/         # 문서 관리 앱
│   ├── static/
│   └── templates/
│
├── frontend/                   # 별도 프론트엔드 (선택사항)
│   └── (React/Vue/Next.js 등)
│
├── tests/                      # 테스트 코드
├── docs/                       # 문서
├── .env                        # 환경 변수
├── requirements.txt
└── docker-compose.yml          # 컨테이너화 (선택사항)
```

### 2.2 비교 분석

| 항목 | Unigo 현재 구조 | 표준 구조 | 평가 |
|------|----------------|-----------|------|
| **Backend 분리** | ✅ `backend/` 디렉토리 존재 | ✅ 별도 디렉토리 | 양호 |
| **Django 앱 구조** | ⚠️ 단일 앱 (`unigo_app`) | ✅ 기능별 다중 앱 | 개선 필요 |
| **LangGraph 구조** | ✅ `graph/` 디렉토리 | ✅ `agents/` 디렉토리 | 양호 |
| **RAG 컴포넌트** | ✅ `rag/` 디렉토리 | ✅ `rag/` 디렉토리 | 양호 |
| **API 설계** | ⚠️ 함수 기반 뷰 | ✅ DRF 클래스 기반 뷰 | 개선 권장 |
| **비동기 처리** | ❌ 없음 | ✅ Celery/Channels | 개선 필요 |
| **데이터베이스** | ⚠️ 미사용 (models.py 비어있음) | ✅ 대화 기록 저장 | 개선 필요 |
| **테스트 코드** | ❌ 없음 | ✅ 단위/통합 테스트 | 개선 필요 |
| **문서화** | ✅ 양호 (README, docs/) | ✅ 문서 존재 | 양호 |
| **환경 설정** | ✅ `.env` 사용 | ✅ `.env` 사용 | 양호 |

---

## 3. 강점 분석

### 3.1 잘 구현된 부분 ✅

1. **명확한 관심사 분리**
   - Backend (LangGraph/RAG)와 Frontend (Django)가 명확히 분리됨
   - `backend/main.py`가 깔끔한 인터페이스 제공

2. **LangGraph 구조**
   - `graph/`, `rag/` 디렉토리로 논리적 분리
   - ReAct 패턴 에이전트 구현
   - 차등 점수 시스템 (Tiered Scoring) 독창적

3. **RAG 시스템**
   - Pinecone 벡터 DB 활용
   - OpenAI Embeddings
   - 다양한 LangChain Tools 구현

4. **문서화**
   - 상세한 README.md
   - docs/ 폴더에 가이드, 계획, 로그 문서
   - 코드 주석 및 Docstring 양호

5. **프론트엔드**
   - Vanilla JS 사용 (프레임워크 의존성 없음)
   - Markdown 링크 파싱 기능
   - 온보딩 플로우 구현

### 3.2 독창적인 기능 🌟

1. **차등 점수 시스템 (Tier 1-4)**
   - 사용자 희망 전공에 대한 정확도 기반 점수 부여
   - 업계 표준에서 찾기 어려운 독창적 구현

2. **LLM 기반 전공명 정규화**
   - 줄임말/오타 자동 보정
   - 사용자 경험 개선

---

## 4. 개선이 필요한 영역

### 4.1 구조적 문제 🔴

#### 1. `static_pages/` 디렉토리의 역할 불명확
- **문제**: `unigo/templates/`와 `static_pages/`가 중복되는 것으로 보임
- **영향**: 코드 중복, 유지보수 어려움
- **확인 필요**: 
  - `static_pages/`가 실제로 사용되는지?
  > User Answer: django app 구현 이전 생성했던 각 페이지 요소들을 담은 디렉토리
  - Django 템플릿과 어떻게 다른지?

#### 2. Django 앱 구조
- **문제**: 단일 앱 (`unigo_app`)에 모든 기능 집중
- **표준**: 기능별 다중 앱 구조
  ```
  apps/
  ├── chatbot/        # 챗봇 기능
  ├── users/          # 사용자 관리
  ├── majors/         # 전공 정보
  └── recommendations/ # 추천 시스템
  ```
  > User Question: 현재 기능에서 다중 앱을 활용할 필요가 있는지?

#### 3. `backend/api/` 디렉토리
- **문제**: 존재하지만 용도 불명확
- **확인 필요**: 사용되는지, 삭제 가능한지?
> User Answer: 초기 langgraph 생성 시 만들어진 디렉토리. 불필요시 삭제

#### 4. `backend/server.py`
- **문제**: Django와 별도로 서버를 실행하는지 불명확
- **확인 필요**: 
  - FastAPI 서버인지?
  - Django와 어떻게 통합되는지?

### 4.2 기능적 문제 🟡

#### 1. 비동기 처리 부재
- **문제**: LLM 호출이 동기적으로 처리됨
- **영향**: 
  - 긴 응답 시간 동안 서버 블로킹
  - 사용자 경험 저하
- **표준 해결책**: 
  - Django Channels (WebSocket)
  - Celery (백그라운드 작업)

#### 2. 대화 기록 미저장
- **문제**: `models.py`가 비어있음, DB 미사용
- **영향**: 
  - 대화 기록 휘발성
  - 사용자 세션 관리 불가
  - 분석/개선 데이터 부족

#### 3. 사용자 인증 시스템 부재
- **문제**: 사용자 구분 없음
- **영향**: 
  - 개인화된 추천 불가
  - 대화 기록 관리 불가

#### 4. API 설계
- **문제**: 함수 기반 뷰, Django REST Framework 미사용
- **영향**: 
  - API 문서 자동 생성 불가
  - Serialization 수동 처리
  - 확장성 제한

### 4.3 품질 관리 문제 🟡

#### 1. 테스트 코드 부재
- **문제**: 단위 테스트, 통합 테스트 없음
- **영향**: 
  - 리팩토링 시 버그 위험
  - 코드 품질 보장 어려움

#### 2. 에러 핸들링
- **문제**: 기본적인 try-catch만 존재
- **개선 필요**: 
  - 구체적인 예외 처리
  - 로깅 시스템
  - 사용자 친화적 에러 메시지

#### 3. 보안
- **문제**: 
  - `@csrf_exempt` 사용 (CSRF 보호 비활성화)
  - Rate limiting 없음
  - 입력 검증 부족

### 4.4 성능 및 확장성 🟡

#### 1. 그래프 캐싱
- **현재**: 전역 변수로 캐싱 (`_graph_react`, `_graph_major`)
- **문제**: 멀티 프로세스 환경에서 작동 안 함
- **개선**: Redis 등 외부 캐시 사용

#### 2. 벡터 검색 최적화
- **현재**: Top-K=50 고정
- **개선**: 동적 조정, 하이브리드 검색

#### 3. 프론트엔드 최적화
- **현재**: 전체 페이지 새로고침
- **개선**: SPA 또는 HTMX 활용

---

## 5. 구체적인 개선 방안

### 5.1 구조 개선

#### A. `static_pages/` 정리
```bash
# 1. static_pages가 사용되지 않는다면 삭제
rm -rf static_pages/

# 2. 사용된다면 용도 명확화
# - Figma 디자인 프로토타입? → docs/design/ 으로 이동
# - 실제 페이지? → unigo/templates/ 로 통합
```

#### B. Django 앱 재구성
```
unigo/
├── manage.py
├── config/                     # 프로젝트 설정 (unigo → config 이름 변경)
│   ├── settings/
│   │   ├── base.py
│   │   ├── development.py
│   │   └── production.py
│   ├── urls.py
│   └── asgi.py
│
└── apps/
    ├── chatbot/                # 챗봇 기능
    │   ├── views.py
    │   ├── urls.py
    │   ├── serializers.py
    │   └── services.py         # 비즈니스 로직
    │
    ├── users/                  # 사용자 관리
    │   ├── models.py
    │   ├── views.py
    │   └── serializers.py
    │
    ├── majors/                 # 전공 정보
    │   ├── models.py
    │   ├── views.py
    │   └── serializers.py
    │
    └── core/                   # 공통 기능
        ├── middleware.py
        └── utils.py
```

#### C. Backend 구조 개선
```
backend/
├── agents/                     # graph → agents 이름 변경 (표준화)
│   ├── __init__.py
│   ├── graph_builder.py
│   ├── nodes.py
│   ├── state.py
│   └── prompts.py             # 프롬프트 분리
│
├── rag/
│   ├── __init__.py
│   ├── retriever.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   └── tools.py
│
├── services/                   # 비즈니스 로직 분리
│   ├── __init__.py
│   ├── mentor_service.py
│   └── recommendation_service.py
│
├── data/
├── config.py
└── main.py
```

### 5.2 기능 개선

#### A. Django REST Framework 도입

**설치**:
```bash
pip install djangorestframework
```

**구현 예시** (`apps/chatbot/views.py`):
```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ChatRequestSerializer, ChatResponseSerializer

class ChatAPIView(APIView):
    """
    챗봇 대화 API
    
    POST /api/chat
    """
    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        message = serializer.validated_data['message']
        history = serializer.validated_data.get('history', [])
        
        try:
            response_content = run_mentor(
                question=message,
                chat_history=history,
                mode="react"
            )
            
            response_serializer = ChatResponseSerializer({
                'response': str(response_content)
            })
            return Response(response_serializer.data)
            
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
```

**Serializers** (`apps/chatbot/serializers.py`):
```python
from rest_framework import serializers

class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(required=True)
    history = serializers.ListField(
        child=serializers.DictField(),
        required=False,
        default=list
    )

class ChatResponseSerializer(serializers.Serializer):
    response = serializers.CharField()
```

#### B. 데이터베이스 모델 추가

**모델 정의** (`apps/chatbot/models.py`):
```python
from django.db import models
from django.contrib.auth.models import User

class Conversation(models.Model):
    """대화 세션"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']

class Message(models.Model):
    """개별 메시지"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]
    
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    metadata = models.JSONField(null=True, blank=True)  # 추가 정보 (tool 호출 등)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']

class MajorRecommendation(models.Model):
    """전공 추천 결과"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=255)
    onboarding_answers = models.JSONField()  # 온보딩 답변
    recommended_majors = models.JSONField()  # 추천 결과
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
```

**마이그레이션**:
```bash
python manage.py makemigrations
python manage.py migrate
```

#### C. 비동기 처리 (Django Channels)

**설치**:
```bash
pip install channels channels-redis
```

**설정** (`config/settings.py`):
```python
INSTALLED_APPS = [
    'daphne',  # 맨 위에 추가
    # ...
    'channels',
]

ASGI_APPLICATION = 'config.asgi.application'

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
        },
    },
}
```

**WebSocket Consumer** (`apps/chatbot/consumers.py`):
```python
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from backend.main import run_mentor

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
    
    async def disconnect(self, close_code):
        pass
    
    async def receive(self, text_data):
        data = json.loads(text_data)
        message = data['message']
        history = data.get('history', [])
        
        # 비동기로 LLM 호출
        response = await self.get_ai_response(message, history)
        
        await self.send(text_data=json.dumps({
            'response': response
        }))
    
    async def get_ai_response(self, message, history):
        # 실제로는 async 버전의 run_mentor 필요
        # 또는 sync_to_async 사용
        from asgiref.sync import sync_to_async
        return await sync_to_async(run_mentor)(
            question=message,
            chat_history=history,
            mode="react"
        )
```

**라우팅** (`apps/chatbot/routing.py`):
```python
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/chat/$', consumers.ChatConsumer.as_asgi()),
]
```

**ASGI 설정** (`config/asgi.py`):
```python
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from apps.chatbot.routing import websocket_urlpatterns

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})
```

#### D. 로깅 시스템

**설정** (`config/settings.py`):
```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/unigo.log',
            'maxBytes': 1024*1024*15,  # 15MB
            'backupCount': 10,
            'formatter': 'verbose',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
        'apps.chatbot': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'backend': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}
```

**사용 예시**:
```python
import logging

logger = logging.getLogger(__name__)

def chat_api(request):
    logger.info(f"Chat request from {request.user}")
    try:
        # ...
        logger.debug(f"LLM response: {response_content}")
    except Exception as e:
        logger.error(f"Error in chat_api: {e}", exc_info=True)
```

#### E. 테스트 코드

**설치**:
```bash
pip install pytest pytest-django
```

**설정** (`pytest.ini`):
```ini
[pytest]
DJANGO_SETTINGS_MODULE = config.settings
python_files = tests.py test_*.py *_tests.py
```

**테스트 예시** (`apps/chatbot/tests/test_views.py`):
```python
import pytest
from django.urls import reverse
from rest_framework.test import APIClient

@pytest.mark.django_db
class TestChatAPI:
    def test_chat_api_success(self):
        client = APIClient()
        url = reverse('chat-api')
        data = {
            'message': '컴퓨터공학과에 대해 알려줘',
            'history': []
        }
        
        response = client.post(url, data, format='json')
        
        assert response.status_code == 200
        assert 'response' in response.json()
    
    def test_chat_api_empty_message(self):
        client = APIClient()
        url = reverse('chat-api')
        data = {'message': ''}
        
        response = client.post(url, data, format='json')
        
        assert response.status_code == 400
```

**Backend 테스트** (`backend/tests/test_main.py`):
```python
import pytest
from backend.main import run_mentor, run_major_recommendation

def test_run_mentor():
    response = run_mentor("컴퓨터공학과에 대해 알려줘")
    assert isinstance(response, str)
    assert len(response) > 0

def test_run_major_recommendation():
    answers = {
        'subjects': '수학, 물리',
        'interests': '코딩',
        'desired_salary': '5000만원',
        'preferred_majors': '컴퓨터공학과'
    }
    result = run_major_recommendation(answers)
    assert 'recommended_majors' in result
    assert len(result['recommended_majors']) > 0
```

### 5.3 보안 개선

#### A. CSRF 보호 활성화

**DRF 설정** (`config/settings.py`):
```python
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ],
}
```

**프론트엔드에서 CSRF 토큰 전송**:
```javascript
// chat.js
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

const csrftoken = getCookie('csrftoken');

fetch('/api/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrftoken
    },
    body: JSON.stringify(data)
});
```

#### B. Rate Limiting

**설치**:
```bash
pip install django-ratelimit
```

**사용**:
```python
from django_ratelimit.decorators import ratelimit

@ratelimit(key='ip', rate='10/m', method='POST')
def chat_api(request):
    # ...
```

#### C. 입력 검증

**Serializer 검증**:
```python
class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(
        required=True,
        max_length=1000,  # 최대 길이 제한
        trim_whitespace=True
    )
    
    def validate_message(self, value):
        if not value.strip():
            raise serializers.ValidationError("메시지가 비어있습니다.")
        
        # 악의적인 입력 차단
        forbidden_patterns = ['<script', 'javascript:', 'onerror=']
        for pattern in forbidden_patterns:
            if pattern.lower() in value.lower():
                raise serializers.ValidationError("유효하지 않은 입력입니다.")
        
        return value
```

### 5.4 성능 개선

#### A. Redis 캐싱

**설치**:
```bash
pip install django-redis
```

**설정** (`config/settings.py`):
```python
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# 그래프 캐싱
GRAPH_CACHE_TIMEOUT = 60 * 60 * 24  # 24시간
```

**사용** (`backend/main.py`):
```python
from django.core.cache import cache

def get_graph(mode: str = "react"):
    cache_key = f"graph_{mode}"
    graph = cache.get(cache_key)
    
    if graph is None:
        graph = build_graph(mode=mode)
        cache.set(cache_key, graph, timeout=settings.GRAPH_CACHE_TIMEOUT)
    
    return graph
```

#### B. 데이터베이스 인덱싱

```python
class Message(models.Model):
    # ...
    
    class Meta:
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['conversation', 'created_at']),
            models.Index(fields=['role']),
        ]
```

#### C. 쿼리 최적화

```python
# Bad: N+1 쿼리
conversations = Conversation.objects.all()
for conv in conversations:
    messages = conv.messages.all()  # 각 대화마다 쿼리 실행

# Good: select_related, prefetch_related
conversations = Conversation.objects.prefetch_related('messages').all()
for conv in conversations:
    messages = conv.messages.all()  # 캐시된 데이터 사용
```

---

## 6. 우선순위별 실행 계획

### 🔴 높은 우선순위 (즉시 실행)

#### 1. `static_pages/` 정리 (1-2시간)
- [ ] 사용 여부 확인
- [ ] 미사용 시 삭제 또는 용도 명확화

#### 2. 데이터베이스 모델 추가 (4-6시간)
- [ ] `Conversation`, `Message` 모델 생성
- [ ] 마이그레이션 실행
- [ ] 뷰에서 대화 기록 저장 로직 추가

#### 3. 로깅 시스템 구축 (2-3시간)
- [ ] `LOGGING` 설정 추가
- [ ] 주요 함수에 로깅 추가
- [ ] `logs/` 디렉토리 생성

#### 4. 에러 핸들링 개선 (3-4시간)
- [ ] 구체적인 예외 처리
- [ ] 사용자 친화적 에러 메시지
- [ ] 로깅과 통합

### 🟡 중간 우선순위 (1-2주 내)

#### 5. Django REST Framework 도입 (1-2일)
- [ ] DRF 설치 및 설정
- [ ] Serializers 작성
- [ ] 클래스 기반 뷰로 전환
- [ ] API 문서 자동 생성 (drf-spectacular)

#### 6. 테스트 코드 작성 (2-3일)
- [ ] pytest 설정
- [ ] 주요 API 엔드포인트 테스트
- [ ] Backend 함수 단위 테스트
- [ ] CI/CD 파이프라인 구축 (선택)

#### 7. 보안 강화 (1-2일)
- [ ] CSRF 보호 활성화
- [ ] Rate limiting 추가
- [ ] 입력 검증 강화

#### 8. 성능 최적화 (2-3일)
- [ ] Redis 캐싱 도입
- [ ] 데이터베이스 인덱싱
- [ ] 쿼리 최적화

### 🟢 낮은 우선순위 (장기 계획)

#### 9. Django 앱 재구성 (3-5일)
- [ ] 기능별 다중 앱 구조로 전환
- [ ] 코드 마이그레이션
- [ ] 테스트 및 검증

#### 10. 비동기 처리 (Django Channels) (5-7일)
- [ ] Channels 설치 및 설정
- [ ] WebSocket Consumer 구현
- [ ] 프론트엔드 WebSocket 연동
- [ ] Redis 메시지 브로커 설정

#### 11. 사용자 인증 시스템 (3-5일)
- [ ] Django User 모델 활용
- [ ] 회원가입/로그인 API
- [ ] JWT 토큰 인증 (선택)
- [ ] 프론트엔드 인증 플로우

#### 12. 고급 기능 (각 2-3일)
- [ ] 대화 기록 검색
- [ ] 전공 비교 기능
- [ ] 추천 결과 북마크
- [ ] 모바일 최적화

---

## 7. 추가 권장 사항

### 7.1 개발 환경

#### Docker 컨테이너화
```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
    env_file:
      - .env
  
  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=unigo
      - POSTGRES_USER=unigo
      - POSTGRES_PASSWORD=unigo
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

#### 환경 분리
```python
# config/settings/base.py
# config/settings/development.py
# config/settings/production.py

# manage.py
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')
```

### 7.2 모니터링 및 관찰성

#### Sentry 통합 (에러 추적)
```bash
pip install sentry-sdk
```

```python
# config/settings.py
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    integrations=[DjangoIntegration()],
    traces_sample_rate=1.0,
)
```

#### LangSmith 통합 (LLM 모니터링)
```python
# backend/config.py
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"
os.environ["LANGCHAIN_PROJECT"] = "unigo"
```

### 7.3 문서화

#### API 문서 자동 생성
```bash
pip install drf-spectacular
```

```python
# config/settings.py
INSTALLED_APPS = [
    # ...
    'drf_spectacular',
]

REST_FRAMEWORK = {
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
}

# config/urls.py
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

urlpatterns = [
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
]
```

---

## 8. 결론

### 8.1 현재 상태 평가

Unigo 프로젝트는 **전반적으로 잘 구조화된 Django-LangGraph-RAG 시스템**입니다. 특히 다음 부분이 우수합니다:

✅ **강점**:
- Backend와 Frontend의 명확한 분리
- LangGraph 및 RAG 시스템의 체계적 구현
- 독창적인 차등 점수 시스템
- 상세한 문서화

⚠️ **개선 필요**:
- 데이터베이스 활용 부족
- 비동기 처리 부재
- 테스트 코드 부재
- 보안 강화 필요

### 8.2 최종 권장사항

1. **단기 (1-2주)**:
   - 데이터베이스 모델 추가 (대화 기록 저장)
   - 로깅 시스템 구축
   - 에러 핸들링 개선

2. **중기 (1-2개월)**:
   - Django REST Framework 도입
   - 테스트 코드 작성
   - 보안 강화 (CSRF, Rate limiting)
   - 성능 최적화 (Redis 캐싱)

3. **장기 (3-6개월)**:
   - Django 앱 재구성 (다중 앱 구조)
   - 비동기 처리 (Django Channels)
   - 사용자 인증 시스템
   - 고급 기능 추가

### 8.3 참고 자료

- [Django Best Practices](https://docs.djangoproject.com/en/stable/misc/design-philosophies/)
- [Django REST Framework](https://www.django-rest-framework.org/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Django Channels](https://channels.readthedocs.io/)
- [Twelve-Factor App](https://12factor.net/)

---

**작성자**: AI Assistant  
**검토 필요**: 프로젝트 팀  
**다음 단계**: 우선순위 높은 항목부터 순차적 실행
