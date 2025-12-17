# 채팅 이력 관리 기능 구현 보고서

**날짜**: 2025-12-11
**작업**: 로그인 사용자의 채팅 세션을 저장·조회하여 과거 대화를 불러올 수 있는 기능 구현

## 개요
사용자가 작성한 채팅을 저장하고 과거 채팅 이력을 조회 및 불러올 수 있는 기능을 구현
- 로그인 사용자 전용 기능
- 새 채팅 생성 시 현재 세션을 DB에 자동 저장
- 과거 채팅 목록 조회 및 로드 기능

---

## 구현된 기능

### 1. 새 채팅 생성 시 현재 세션 저장
**목표**: 새로운 채팅을 시작할 때 현재까지 진행한 채팅 내용을 DB에 저장

#### 구현 내용
- **파일**: `unigo/unigo_app/views.py` - `save_chat_history()` 엔드포인트
- **엔드포인트**: `POST /api/chat/save`
- **인증**: 로그인 필수 (`@login_required`)

**요청 파라미터**:
```json
{
  "history": [
    {"role": "user", "content": "안녕하세요"},
    {"role": "assistant", "content": "반갑습니다!"}
  ]
}
```

**응답**:
```json
{
  "message": "Chat history saved successfully",
  "conversation_id": 1
}
```

**동작**:
1. 프론트엔드에서 `chatHistory` 배열 전송
2. 첫 user 메시지에서 제목 추출 (최대 50자)
3. 새로운 `Conversation` 레코드 생성 (사용자별, 고유 session_id)
4. 각 메시지를 `Message` 레코드로 저장

**프론트엔드**: `unigo/static/js/chat.js` - `resetChat()` 함수
- "새 채팅" 버튼 클릭 시 발동
- 로그인 사용자만 `/api/chat/save` 호출
- 비로그인 사용자는 저장 스킵

---

### 2. 과거 채팅 목록 조회
**목표**: 폴더 아이콘 클릭 시 우측 패널에 과거 대화 리스트 표시

#### 구현 내용
- **파일**: `unigo/unigo_app/views.py` - `list_conversations()` 엔드포인트
- **엔드포인트**: `GET /api/chat/list`
- **인증**: 로그인 필수

**응답 형식**:
```json
{
  "conversations": [
    {
      "id": 1,
      "title": "컴퓨터공학과 추천받기",
      "created_at": "2025-12-11T10:30:00.000Z",
      "updated_at": "2025-12-11T10:35:00.000Z",
      "message_count": 8,
      "last_message_preview": "감사합니다!"
    },
    ...
  ]
}
```

**리스트 항목 구성**:
- 제목 (title)
- 최근 수정 날짜
- 메시지 개수
- 마지막 메시지 미리보기

**프론트엔드**: `unigo/static/js/chat.js` - `showConversationList()` 함수
- 폴더 버튼 클릭 시 호출
- `/api/chat/list` 호출하여 리스트 조회
- 우측 `.result-card` 패널에 리스트 HTML 렌더링
- 각 항목 클릭 이벤트 바인딩

---

### 3. 과거 채팅 불러오기
**목표**: 리스트에서 항목 선택 시 해당 세션의 채팅을 좌측 영역에 로드

#### 구현 내용
- **파일**: `unigo/unigo_app/views.py` - `load_conversation()` 엔드포인트
- **엔드포인트**: `GET /api/chat/load?conversation_id=123`
- **인증**: 로그인 필수

**응답 형식**:
```json
{
  "conversation": {
    "id": 1,
    "title": "컴퓨터공학과 추천받기",
    "messages": [
      {
        "role": "user",
        "content": "컴퓨터공학과에 대해 알려주세요",
        "created_at": "2025-12-11T10:30:00.000Z"
      },
      {
        "role": "assistant",
        "content": "컴퓨터공학과는...",
        "created_at": "2025-12-11T10:30:05.000Z"
      }
    ]
  }
}
```

#### 충돌 처리 (현재 세션 vs 과거 세션)
프론트엔드에서 두 단계 confirm으로 처리:

1. **첫 번째 confirm**:
   ```
   현재 대화가 있습니다. 서버에 저장한 후 불러오시겠습니까?
   - 확인: 저장 후 불러오기
   - 취소: 저장하지 않고 불러오기 선택지로 이동
   ```

2. **두 번째 confirm** (첫 번째에서 취소 선택 시):
   ```
   저장하지 않고 불러오시겠습니까?
   - 확인: 불러오기 진행
   - 취소: 작업 중단
   ```

**동작 흐름**:
1. 리스트 항목 클릭 → `loadConversation(convId)` 호출
2. 현재 세션에 내용이 있으면 충돌 확인
3. 선택 사항에 따라:
   - 저장 선택: `/api/chat/save` 호출 후 `/api/chat/load` 호출
   - 저장 안 함: 바로 `/api/chat/load` 호출
4. 로드된 메시지들로 `chatHistory` 업데이트
5. `currentConversationId` 업데이트 (이후 메시지 추가 시 해당 세션 사용)
6. `onboardingState.isComplete = true` 설정 (온보딩 프롬프트 제외)
7. 우측 패널에 "불러온 대화" 정보 표시

**프론트엔드**: `unigo/static/js/chat.js`
- `loadConversation(convId)` 함수
- 충돌 처리 로직 구현
- UI 상태 업데이트

---

## 변경 사항 요약

| 파일 | 변경 사항 |
|------|---------|
| `unigo/unigo_app/views.py` | `save_chat_history()`, `list_conversations()`, `load_conversation()` 추가; `chat_api()` 수정 |
| `unigo/unigo_app/urls.py` | `/api/chat/save`, `/api/chat/list`, `/api/chat/load` 경로 추가 |
| `unigo/static/js/chat.js` | `currentConversationId` 변수 추가; `resetChat()`, `handleChatInput()`, `showConversationList()`, `loadConversation()` 함수 추가/수정 |
| `unigo/unigo_app/models.py` | 변경 없음 (기존 모델 사용) |

### 백엔드 (views.py)
| 함수 | 변경 내용 |
|------|---------|
| `save_chat_history()` | 새 엔드포인트 추가 (로그인 전용) |
| `list_conversations()` | 새 엔드포인트 추가 (로그인 전용) |
| `load_conversation()` | 새 엔드포인트 추가 (로그인 전용) |
| `chat_api()` | `conversation_id` 파라미터 추가 및 처리 |
| `reset_chat_history()` | 기존 유지 (로그인 사용자 세션 삭제) |

### URL 라우팅 (urls.py)
```python
path("api/chat/save", views.save_chat_history, name="save_chat_history"),
path("api/chat/list", views.list_conversations, name="list_conversations"),
path("api/chat/load", views.load_conversation, name="load_conversation"),
```

### 프론트엔드 (chat.js)
| 변경 사항 | 설명 |
|----------|------|
| `currentConversationId` 변수 추가 | 현재 대화 ID 추적 |
| `resetChat()` 수정 | 저장 로직 추가 (로그인 사용자만) |
| `handleChatInput()` 수정 | `conversation_id` 전송 |
| `showConversationList()` 추가 | 리스트 조회 및 표시 |
| `loadConversation()` 추가 | 과거 대화 불러오기 및 충돌 처리 |

---

## 버그 수정

### Issue 1: Duplicate entry for key 'session_id'
**원인**: 로그인 사용자가 새 Conversation 생성 시 `session_id`를 설정하지 않아서 빈 문자열(``'``)이 반복 저장됨

**해결**: 로그인 사용자도 `session_id=str(uuid.uuid4())`로 고유한 UUID 생성
- 파일: `views.py` - `chat_api()` 함수 수정
- 로그인/비로그인 모두 각 Conversation마다 고유한 session_id 보유

### Issue 2: 두 번째 대화가 첫 번째 대화 ID로 저장
**원인**: `chat_api`에서 항상 최근 대화(`.order_by("-updated_at").first()`)를 가져와서 재사용

**해결**: `conversation_id` 파라미터 도입
- 프론트엔드가 매 메시지마다 `conversation_id` 전송
- 없으면 새 Conversation 생성
- 새 채팅 시작 시 `currentConversationId = null` 초기화

---

## 사용자 플로우

### 1. 첫 번째 대화 작성 및 저장
```
1. 사용자 로그인
2. 온보딩 질문 응답
3. 메시지 전송
4. currentConversationId 설정 (첫 응답에서)
5. 추가 메시지 전송 (같은 conversation_id 사용)
6. "새 채팅" 버튼 클릭
   - 현재 chatHistory를 /api/chat/save로 전송 (새 Conversation 생성)
   - chatHistory 초기화
   - currentConversationId = null
```

### 2. 두 번째 대화 작성
```
1. 온보딩 다시 시작
2. 메시지 전송
3. currentConversationId 설정 (새 값)
4. 추가 메시지 전송
```

### 3. 과거 대화 불러오기
```
1. 폴더 아이콘 클릭
2. /api/chat/list 호출 → 과거 대화 리스트 표시
3. 첫 번째 대화 항목 클릭 → loadConversation(1) 호출
4. 현재 내용이 있으면:
   - confirm: "저장 후 불러오기?" → 저장 후 로드
   - 또는: "저장 안 하고 불러오기?" → 바로 로드
5. chatHistory 업데이트
6. currentConversationId = 1
7. 불러온 대화에서 추가 메시지 작성 가능
```

---

## 테스트 체크리스트

- [ ] 로그인 사용자: 첫 번째 대화 입력 및 전송
- [ ] 새 채팅 클릭 시 첫 대화가 DB에 저장되는지 확인
- [ ] DB에서 Conversation ID가 다른지 확인
- [ ] 두 번째 대화 입력 시 새로운 Conversation ID 생성되는지 확인
- [ ] 폴더 아이콘 클릭 시 과거 대화 리스트 표시
- [ ] 리스트 항목 클릭 시 해당 대화 로드
- [ ] 충돌 처리: 저장 후 불러오기 동작 확인
- [ ] 충돌 처리: 저장하지 않고 불러오기 동작 확인
- [ ] 불러온 대화에서 추가 메시지 입력 가능 확인
- [ ] 비로그인 사용자는 저장 기능 제외되는지 확인

---

## 비고

- 모든 새 엔드포인트는 로그인 필수입니다 (`@login_required`).
- 프론트엔드는 `currentConversationId`를 통해 현재 세션을 추적하고 없으면 새 세션 생성
- 과거 대화 로드 시 `onboardingState.isComplete = true`로 설정하여 온보딩 프롬프트는 보이지 않음