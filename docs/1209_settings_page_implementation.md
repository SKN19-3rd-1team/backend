# 설정 페이지(Settings Page) 기능 구현

**작성일**: 2025-12-09
**작성자**: Antigravity

## 1. 개요
사용자 계정 정보(닉네임, 비밀번호)를 변경할 수 있는 설정 페이지를 구현했습니다. 기존 단일 페이지 디자인을 닉네임 변경과 비밀번호 변경 탭으로 분리하여 UX를 개선하였으며, 관련 백엔드 API와 프론트엔드 검증 로직을 추가했습니다. 또한, 페이지별 스타일의 통일성을 위해 CSS 리팩토링을 진행했습니다.

## 2. 주요 수정 기능

### 2.1 닉네임 변경 기능
- **통신 방식**: AJAX (Async/Await)
- **주요 로직**:
    1. **중복 확인**: 사용자가 입력한 새 닉네임이 기존 사용자의 것과 중복되는지 `api/setting/check-username` 엔드포인트를 통해 확인합니다.
    2. **비밀번호 검증**: 보안을 위해 현재 로그인된 사용자의 비밀번호를 입력받아 본인 인증을 수행합니다.
    3. **변경**: 검증 완료 시 현재 사용자의 `username`을 업데이트합니다.

### 2.2 비밀번호 변경 기능
- **통신 방식**: AJAX (Async/Await)
- **주요 로직**:
    1. **현재 비밀번호 검증**: 서버 측에서 현재 비밀번호가 일치하는지 확인합니다.
    2. **새 비밀번호 일치 확인**: 클라이언트(프론트엔드)에서 "새 비밀번호"와 "새 비밀번호 확인" 입력값이 일치하는지 실시간으로 검증합니다.
    3. **세션 유지**: 비밀번호 변경 후 Django의 `update_session_auth_hash`를 사용하여 로그아웃되지 않고 세션을 유지하도록 처리했습니다.

### 2.3 UI/UX 개선 (탭 인터페이스)
- `setting.html`을 기존 디자인에서 **탭(Tab) 구조**로 변경하였습니다.
- **초기 진입 시**: "닉네임 변경" 탭이 활성화됩니다.
- **전환 시**: 탭 클릭 시 폼 입력값이 초기화되며, 부드러운 전환 효과를 제공합니다.

### 2.4 CSS 리팩토링
- `setting.html` 내부에 `<style>` 태그로 작성되어 있던 CSS 코드를 `unigo/static/css/styles.css`로 통합하였습니다.
- 클래스명 충돌 방지를 위해 기존 `btn-primary` 클래스를 설정 페이지 전용인 `setting-btn-primary`로 변경하여 다른 페이지(로그인/회원가입 등) 레이아웃에 영향을 주지 않도록 처리했습니다.

## 3. 기술적 변경 사항 (Code Changes)

### Backend (`unigo_app/views.py`, `urls.py`)
- **API 추가**:
    - `POST /api/setting/check-username`: 닉네임 중복 확인
    - `POST /api/setting/change-nickname`: 닉네임 변경 (비밀번호 검증 포함)
    - `POST /api/setting/change-password`: 비밀번호 변경 (세션 갱신 포함)
- **View 수정**:
    - `setting`: 비로그인 사용자가 접근 시 로그인 페이지(`auth`)로 리다이렉트 처리 (`@login_required` 대신 로직 내 처리로 유연성 확보)

### Frontend (`unigo/templates/unigo_app/setting.html`)
- 사이드바에 현재 접속자 정보(Username, Email) 표시 기능 추가.
- JavaScript `fetch` API를 활용하여 페이지 새로고침 없는 비동기 처리 구현.
- 변경 성공 시 `alert` 팝업 후 페이지 리로드(`location.reload`)로 최신 상태 반영.

## 4. 문제 해결 (Troubleshooting)
- **ISSUE**: `styles.css` 통합 과정에서 공통 클래스 `btn-primary`가 다른 페이지 스타일과 충돌할 가능성 확인.
- **SOLVED**: 설정 페이지의 버튼 클래스를 `setting-btn-primary`로 분리하여 정의함으로써 기존 디자인 시스템을 해치지 않고 통합 완료.
