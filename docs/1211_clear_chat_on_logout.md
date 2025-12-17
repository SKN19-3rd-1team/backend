# Issues16

**날짜**: 2025-12-11

### 문제 상황
1. 비로그인 상태에서 이전 사용자 채팅 내역이 노출됨
2. 동일 계정 재로그인 시 DB duplicate entry 오류 발생
- 원인: 로그아웃 시 클라이언트 측 sessionStorage가 초기화되지 않아 새로운 세션을 오염시킴

### 해결 방법
- 로그아웃 완료 페이지 렌더링
    - 브라우저에서 sessionStorage를 초기화하고 로그인 페이지로 리다이렉트
- Frontend 상태 초기화
    - 로그인 사용자는 새로운 대화 세션을 시작하도록 chatHistory 초기화
    - 온보딩 완료 상태는 유지하면서, 이전 세션 데이터는 제거

### 수정 사항
- **파일**: `unigo/unigo_app/views.py`
    - 로그아웃 후 초기화 페이지 렌더링

- **파일**: `unigo/templates/unigo_app/logout.html`
    - sessionStorage를 초기화하고 로그인 페이지로 자동 리다이렉트

- **파일**: `unigo/static/js/chat.js`
    - 로그인 시 새 세션 시작 및 이전 대화 내역 제거


### 테스트 체크리스트
1. 로그아웃 → 새 게스트 세션 시작
    - [ ] 이전 사용자 채팅 내역이 표시되지 않음
    - [ ] 온보딩 다시 시작

2. 로그아웃 → 동일 계정 재로그인
    - [ ] 이전 채팅 내역이 노출되지 않음
    - [ ] DB duplicate entry 오류 없음

3. 로그인 사용자 채팅 → 저장 → 새 채팅
    - [ ] 저장 후 DB 생성 확인
    - [ ] 리셋 후 저장된 대화가 목록에 표시됨
    - [ ] 저장된 대화를 로드하여 이전 메시지들 확인 가능

---

# Issues15

**날짜**: 2025-12-12

### 문제 상황
- 과거 대화를 불러온 뒤 메시지를 추가하면 새로운 Conversation이 생성되어 기존 대화에 이어서 저장되지 않고 새로운 대화로 저장됨
- 원인: 프론트엔드에서 `conversation_id`를 지속적으로 보관하지 않아 이후 메시지 전송 시 백엔드에 전달되지 않음

### 해결 방법
- 사용자가 과거 대화를 불러오면 클라이언트에 `currentConversationId`를 sessionStorage로 보관
    - 이후 추가되는 메시지는 같은 `conversation_id`로 저장
- 새 채팅 또는 로그아웃 시에는 해당 키를 제거하여 세션 오염 방지

### 수정 사항
- **파일**: `unigo/static/js/chat.js`
    - `STORAGE_KEY_CONVERSATION_ID` 상수 추가 및 
    - `saveState()` → conversation_id 저장
    - `loadState()` → 인증된 사용자일 때 sessionStorage에 남아 있는 conversation_id를 복원
    - `resetChat()` → 저장된 conversation_id 삭제

- **파일**: `unigo/templates/unigo_app/logout.html`
    - 로그아웃 시 `currentConversationId` 삭제 코드 추가
    - 클라이언트에 대화 ID가 남지 않게 함


### 테스트 체크리스트
1. 로그인 → 과거 대화 불러오기 → 메시지 추가
    - [ ] DB에서 해당 `conversation.id`에 메시지 추가 확인

2. 불러온 대화 상태에서 페이지 새로고침
    - [ ] `conversation.id`가 복원되어 동일 대화로 이어지는지 확인

3. 새 채팅(리셋) 또는 로그아웃 후 다시 불러오기
    - [ ] 이전 세션의 `conversation_id`가 남아있지 않은지 확인