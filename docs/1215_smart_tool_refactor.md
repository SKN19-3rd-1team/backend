# Smart Tool 리팩토링: `get_major_career_info`

## 배경
기존 `get_major_career_info` 툴은 전공에 대한 모든 정보(진로, 취업률, 연봉, 자격증 등)를 한 번에 반환했습니다.
이로 인해 사용자가 "컴공 남녀 성비 알려줘"와 같이 구체적이고 좁은 질문을 하더라도, 시스템이 너무 방대한 정보를 가져와 답변하게 되어 "과도한 정보(TMI)" 문제가 발생했습니다.

## 변경 내용

### 1. `backend/rag/tools.py`
`get_major_career_info` 함수의 시그니처를 변경하고, 필터링 로직을 추가했습니다.

```python
def get_major_career_info(major_name: str, specific_field: str = "all") -> Dict[str, Any]:
```

- **`specific_field` 파라미터**:
  - `all` (기본값): 전체 정보 반환
  - `jobs`: 진로, 직업 목록, 진출 분야만 반환
  - `stats`: 취업률, 경쟁률, 남녀 성비, 연봉, 만족도 등 통계만 반환
  - `academics`: 주요 과목, 자격증, 관련 활동만 반환

### 2. `backend/graph/nodes.py`
System Prompt를 업데이트하여 LLM이 이 파라미터를 적극적으로 사용하도록 지시했습니다.

> "사용자가 특정 정보(예: 취업률, 진로)만 물어보는 경우, `specific_field` 파라미터를 사용하여 필요한 정보만 요청하세요."

## 기대 효과
- **토큰 절약**: 불필요한 텍스트 생성을 줄임.
- **정확성 향상**: LLM이 주어진 데이터 안에서만 답변하므로, 묻지 않은 정보를 나열하는 할루시네이션/Verbosity 감소.
- **Latency 개선**: 응답 생성 시간이 단축될 가능성 있음.
