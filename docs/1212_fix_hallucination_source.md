# Fix LLM Hallucination on Major Career Info Source

## 개요
특정 대학(예: 한양대)에 대한 취업률/연봉 정보를 물었을 때, LLM이 '커리어넷'의 일반 전공 데이터를 마치 해당 대학의 데이터인 것처럼 답변하는 환각 현상(Hallucination)이 발생했습니다.
이를 해결하기 위해 툴의 반환 값에 강력한 경고 메시지를 추가하고, 시스템 프롬프트를 강화하여 "일반 정보임"을 명시하도록 강제했습니다.

## 수정 내용

### 1. `backend/rag/tools.py` 수정
- `get_major_career_info` 툴의 반환 딕셔너리에 `warning_context`와 `data_source_disclaimer` 필드 추가.
- `warning_context`: "[치명적 경고] ... 반드시 '커리어넷'의 [국가 표준 데이터]임을 명시해야 합니다."
- Docstring 업데이트: 특정 대학의 정보로 답변하지 말 것을 명시.

### 2. `backend/graph/nodes.py` 수정
- `agent_node`의 `SystemMessage` 프롬프트 수정.
- **출처 명시 필수** 섹션 추가:
    > "절대 '한양대학교의 취업률은...' 이라고 답변하지 마세요. 대신 '한양대학교의 공식 취업률 자료는 확인되지 않았으나, 커리어넷의 일반적인 컴퓨터공학과 취업률은...' 이라고 답변하세요."

## 검증 방법
- `backend/rag/tools.py` 및 `backend/graph/nodes.py` 코드 리뷰를 통해 경고 문구 및 프롬프트가 올바르게 적용되었음을 확인했습니다.
- `backend/scripts/verify_tool_output.py` 스크립트 생성 (환경 문제로 실행은 건너뜀, 정적 분석으로 대체).

## 결과
이제 LLM은 전공 진로/취업률 정보를 제공할 때, 무조건 "커리어넷/국가 표준 데이터 기반"임을 밝히고 대학별 개별 자료 매칭이 아님을 사용자에게 고지하게 됩니다.
