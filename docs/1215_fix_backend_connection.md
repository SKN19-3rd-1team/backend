# 챗봇 백엔드 연결 오류 수정

## 개요
챗봇 사용 시 "챗봇 백엔드가 연결되지 않았습니다. 관리자에게 문의하세요."라는 오류가 발생하는 문제를 해결했습니다.

## 원인 분석
- `unigo/unigo_app/views.py` 파일은 `backend.main` 모듈에서 `run_mentor_stream` 함수를 임포트하여 사용하도록 구현되어 있었습니다.
- 그러나 `backend/main.py` 파일에는 해당 함수가 존재하지 않았습니다.
- Git 히스토리 분석 결과, `run_mentor_stream` 함수가 `backend/main.py`에서 **삭제된 것이 아니라, 애초에 추가되지 않은 상태**에서 `views.py`가 해당 함수를 사용하도록 수정된 것으로 파악됩니다.

## 작업 내용
### 1. `backend/main.py` 수정
- 누락된 `run_mentor_stream` 함수를 구현하여 추가했습니다.
- 이 함수는 `get_graph(mode="react").stream(state, stream_mode="updates")`을 호출하여 챗봇 응답을 스트리밍합니다.
- `unigo_app/views.py`와의 호환성을 확보했습니다.

## 결과
- 챗봇 백엔드 연결 오류가 해결되고, 정상적으로 대화가 스트리밍될 것으로 예상됩니다.
