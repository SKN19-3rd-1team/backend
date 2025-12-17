# 12/14 UI/UX 개선 및 로직 수정 작업 내역

## 1. 웰컴 메시지 로직 수정 (신규 유저 구분)

### 현상

- 신규 가입 후 바로 로그인했을 때, 기존 유저에게 보여주는 "다시 만나서 반갑습니다!" 메시지가 출력되는 문제 발생.
- `chatHistory`가 비어있다는 조건만으로는 '신규 유저'와 '대화 기록이 없는 기존 유저'를 구분하지 못함.

### 수정 내용

- **Backend (`unigo_app/views.py`)**:
  - `/api/auth/me` 응답에 `has_history` 필드 추가.
  - `Conversation` 테이블에 해당 사용자의 대화 기록이 존재하는지 여부를 Boolean으로 반환.
- **Frontend (`static/js/chat.js`)**:
  - `has_history`가 `false`인 경우: "안녕하세요! 처음 뵙겠습니다. 무엇을 도와드릴까요?" (신규 유저 인사)
  - `has_history`가 `true`인 경우: "다시 만나서 반갑습니다! 무엇을 도와드릴까요?" (기존 유저 인사)

## 2. 설정 페이지(Setting Page) 레이아웃 수정

### 문제점

- `setting-container`의 좌우 여백이 비정상적으로 넓게 표시됨.
- Flex 아이템들이 내부 콘텐츠 크기만큼만 줄어드는 현상.
- 폼 입력창 크기가 들쭉날쭉함.

### 수정 내용

- **`styles.css`**:
  - `.setting-container`: `width: 100%` 추가 (화면 너비를 꽉 채우도록 변경).
  - `.setting-panel` vs `.sidebar`: 비율을 약 2.5 : 1로 조정.
  - `.form-input`: `width: 100%`, `box-sizing: border-box` 추가로 모든 입력창이 부모 컨테이너 너비를 가득 채우도록 통일.
  - 불필요한 `.input-row` 클래스 및 스타일 제거.
- **`setting.html`**:
  - `input-row` div 래퍼 제거 (수직 배치로 변경).
  - 비밀번호 변경 탭의 불필요한 "인증(더미)" 버튼 삭제.
  - 닉네임 중복 확인 버튼에 상단 여백(`margin-top`) 추가.

## 3. 캐릭터 선택 카러셀 (Carousel) 개선

### 현상

- 캐릭터 선택 화면에서 마지막 항목(거북이)에서 첫 번째(토끼)로 넘어갈 때, 회전이 자연스럽게 이어지지 않고 반대 방향으로 되감기는(Rewind) 모션 발생.

### 수정 내용

- **Circular Queue 로직 적용 (`character_select.html`)**:
  - 기존의 `0 ~ N-1` 인덱스 제한을 풀고, 무한히 증가/감소 가능한 `currentRotationIndex` 도입.
  - 회전 각도를 `-theta * currentRotationIndex`로 계산하여 끊김 없는 무한 회전 구현.
  - 클릭 이동 시 최단 거리 회전 경로를 계산하도록 로직 개선.

## 4. 기타 코드 정리

- **Backend**: `views.py` 하단에 중복 정의되어 있던 `check_username`, `change_nickname`, `change_password` 함수 제거.
