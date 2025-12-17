# 대화 삭제 기능 구현 (Conversation Delete Feature)

## 개요
사용자가 자신의 과거 대화 기록을 정리할 수 있도록 대화 삭제 기능을 추가했습니다.

## 기능 스펙
- **대상**: 로그인한 사용자 (Guest는 저장된 대화 목록이 없으므로 해당 없음)
- **UI**: 대화 목록의 각 항목 우측 상단에 '휴지통(🗑️)' 아이콘 제공
- **동작**:
    1. 휴지통 아이콘 클릭
    2. 삭제 확인 팝업 (`confirm`)
    3. 확인 시 서버로 삭제 요청 (`DELETE /api/chat/delete/{id}`)
    4. 성공 시 목록에서 즉시 제거

## 구현 상세

### 1. Backend (`unigo_app`)
- **API**: `delete_conversation` 뷰 추가 (`views.py`)
    - Method: `DELETE`, `POST` 지원
    - 권한: `@login_required`
    - 로직: 요청한 `conversation_id`가 현재 로그인한 사용자의 것인지 확인 후 삭제 (`CASCADE`로 메시지 자동 삭제)
- **URL**: `api/chat/delete/<int:conversation_id>` 추가 (`urls.py`)

### 2. Frontend
- **HTML (`chat.html`)**: `conv-item-template`에 `<button class="conv-delete-btn">🗑️</button>` 추가
- **JS (`chat.js`)**:
    - `showConversationList` 내에서 삭제 버튼 이벤트 리스너 등록
    - `e.stopPropagation()`으로 항목 열기 이벤트 방지
    - 삭제 성공 시 `li.remove()`로 DOM 업데이트 및 `currentConversationId` 체크 후 초기화
- **CSS (`chat.css`)**:
    - `.conv-list-ul li`: `position: relative` 추가
    - `.conv-delete-btn`: 우측 상단 절대 배치 및 스타일링

## 트러블슈팅 / 메모
- 삭제 버튼 클릭 시 대화가 열리는 문제 발생 방지를 위해 `stopPropagation()` 필수 적용.
- `DELETE` 메소드 사용 시 CSRF 토큰 헤더 포함 필요 (`getPostHeaders()` 활용).
