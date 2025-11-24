# backend/graph/nodes.py
"""
LangGraph 그래프를 구성하는 노드 함수들을 정의합니다.

이 파일에는 두 가지 패턴이 공존합니다:
1. **ReAct 패턴**: LLM이 자율적으로 tool 호출 여부를 결정 (agent_node, should_continue)
2. **Structured 패턴**: 미리 정해진 순서대로 실행되는 파이프라인 (retrieve_node, select_node, answer_node)
"""

from typing import List, Set
import re
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.constants import END
from pydantic import BaseModel, Field

from .state import MentorState
from backend.rag.retriever import retrieve_with_filter
from backend.rag.entity_extractor import extract_filters, build_chroma_filter

from backend.rag.tools import (
    retrieve_courses,
    list_departments,
    get_universities_by_department,
    recommend_curriculum,
    get_search_help,
    match_department_name
)

from backend.config import get_llm

# LLM 인스턴스 생성 (.env에서 설정한 LLM_PROVIDER와 MODEL_NAME 사용)
llm = get_llm()


# ==================== Post-processing Validation ====================

def extract_departments_from_tool_results(messages: List) -> Set[str]:
    """
    ToolMessage에서 list_departments가 반환한 학과명을 추출합니다.
    """
    all_departments = set()

    for msg in messages:
        if isinstance(msg, ToolMessage):
            # list_departments 결과인지 확인
            if "list_departments" in str(msg.name):
                content = msg.content

                # 백틱으로 감싸진 학과명 추출 (예: `컴퓨터공학과`)
                pattern = r'`([^`]+)`'
                departments = re.findall(pattern, content)
                all_departments.update(departments)

    return all_departments


def validate_and_fix_department_names(
    llm_response: str,
    valid_departments: Set[str]
) -> tuple[str, List[dict]]:
    """
    LLM 응답에서 학과명을 검증하고 수정합니다.

    Args:
        llm_response: LLM이 생성한 응답 텍스트
        valid_departments: Tool에서 반환된 유효한 학과명 집합

    Returns:
        (수정된 응답, 위반 목록)
    """
    if not valid_departments:
        return llm_response, []

    # **학과명** 패턴 추출
    pattern = r'\*\*([^*]+)\*\*'
    mentioned_depts = re.findall(pattern, llm_response)

    violations = []

    for dept in mentioned_depts:
        # 정확히 일치하는지 확인
        if dept not in valid_departments:
            # 유사한 학과 찾기 (Levenshtein 거리 또는 부분 매칭)
            best_match = None
            best_score = 0

            for valid_dept in valid_departments:
                # 1. 부분 문자열 매칭
                if dept in valid_dept or valid_dept in dept:
                    score = len(set(dept) & set(valid_dept)) / max(len(dept), len(valid_dept))
                    if score > best_score:
                        best_score = score
                        best_match = valid_dept

            # 유사도가 충분히 높으면 교체, 아니면 제거
            if best_match and best_score > 0.5:
                violations.append({
                    "wrong": dept,
                    "correct": best_match,
                    "action": "replace",
                    "score": best_score
                })

                # 응답에서 교체
                llm_response = llm_response.replace(
                    f"**{dept}**",
                    f"**{best_match}**"
                )
            else:
                # 매칭되는 학과가 없으면 경고만 (제거는 하지 않음)
                violations.append({
                    "wrong": dept,
                    "correct": None,
                    "action": "warn",
                    "score": 0
                })

    return llm_response, violations


def strict_validate_and_fix_department_names(
    llm_response: str,
    valid_departments: Set[str]
) -> tuple[str, List[dict]]:
    """
    validate_and_fix_department_names를 보다 엄격하게 확장한 버전.

    - list_departments Tool 결과에 없는 학과명은 가능한 경우 가장 가까운
      학과명으로 교체하고, 그렇지 않으면 응답에서 제거한다.
    """
    if not valid_departments:
        return llm_response, []

    pattern = r'\*\*([^*]+)\*\*'
    mentioned_depts = re.findall(pattern, llm_response)

    violations: List[dict] = []

    for dept in mentioned_depts:
        # Tool 결과에 그대로 존재하면 유지
        if dept in valid_departments:
            continue

        best_match = None
        best_score = 0.0

        # 부분 문자열 조건 없이 전체 후보를 대상으로 유사도 계산
        for valid_dept in valid_departments:
            overlap = len(set(dept) & set(valid_dept))
            score = overlap / max(len(dept), len(valid_dept))
            if score > best_score:
                best_score = score
                best_match = valid_dept

        if best_match and best_score > 0.5:
            # 충분히 비슷하면 Tool 학과명으로 교체
            violations.append({
                "wrong": dept,
                "correct": best_match,
                "action": "replace_strict",
                "score": best_score,
            })
            llm_response = llm_response.replace(
                f"**{dept}**",
                f"**{best_match}**",
            )
        else:
            # 교정 불가능하면 답변에서 제거
            violations.append({
                "wrong": dept,
                "correct": None,
                "action": "remove",
                "score": best_score,
            })
            llm_response = llm_response.replace(f"**{dept}**", dept)

    return llm_response, violations

# ==================== ReAct 에이전트용 설정 ====================
# ReAct 패턴: LLM이 필요시 자율적으로 툴을 호출할 수 있도록 설정
tools = [
    retrieve_courses,
    list_departments,
    get_universities_by_department,
    recommend_curriculum,
    get_search_help,
    match_department_name
]  # 사용 가능한 툴 목록
llm_with_tools = llm.bind_tools(tools)  # LLM에 툴 사용 권한 부여


# ==================== Structured 패턴용 Pydantic 모델 ====================
class CourseSelection(BaseModel):
    """
    Structured Output 패턴에서 사용하는 과목 선택 모델.
    LLM이 JSON 형식으로 선택한 과목 ID와 이유를 반환합니다.
    """
    selected_ids: List[str] = Field(
        description="후보 과목 리스트에서 학생에게 추천할 과목의 ID 리스트 (예: ['course_0', 'course_2'])"
    )
    reasoning: str = Field(
        description="해당 과목들을 선택한 간단한 이유"
    )


# ==================== Structured 패턴 노드들 ====================
# 이 노드들은 미리 정해진 순서대로 실행됩니다: retrieve → select → answer
# LLM이 노드 실행 여부를 선택하지 않습니다.

def retrieve_node(state: MentorState) -> dict:
    """
    [Structured 패턴 - 1단계] 벡터 DB에서 관련 과목 후보를 검색합니다.
    """
    question = state["question"]

    extracted_filters = extract_filters(question)
    chroma_filter = build_chroma_filter(extracted_filters) if extracted_filters else None

    search_k = 5
    docs: List[Document] = retrieve_with_filter(
        question=question,
        search_k=search_k,
        metadata_filter=chroma_filter
    )

    # 3. 필터 적용 여부 기록 (폴백 로직은 제거됨 - 엄격한 필터링 유지)
    filter_applied = chroma_filter is not None
    filter_relaxed = False

    course_candidates = []
    for idx, doc in enumerate(docs):
        meta = doc.metadata
        course_id = f"course_{idx}"

        candidate = {
            "id": course_id,
            "name": meta.get("name", "[과목명 미정]"),
            "university": meta.get("university", "[대학 정보 없음]"),
            "college": meta.get("college", "[단과대 정보 없음]"),
            "department": meta.get("department", "[학과 정보 없음]"),
            "grade_semester": meta.get("grade_semester", "[학년/학기 정보 없음]"),
            "classification": meta.get("course_classification", "[분류 정보 없음]"),
            "description": doc.page_content or "[설명 없음]"
        }
        course_candidates.append(candidate)

    return {
        "retrieved_docs": docs,
        "course_candidates": course_candidates,
        "metadata_filter_applied": filter_applied,
        "metadata_filter_relaxed": filter_relaxed,
    }


def select_node(state: MentorState) -> dict:
    """
    [Structured 패턴 - 2단계] 후보 과목 중 질문에 적합한 과목 ID만 선택.
    """
    question = state["question"]
    candidates = state.get("course_candidates", [])

    if not candidates:
        return {"selected_course_ids": []}

    candidate_lines = []
    for c in candidates:
        candidate_lines.append(
            f"- ID: {c['id']}\n"
            f"  과목명: {c['name']}\n"
            f"  대학: {c['university']}, 학과: {c['department']}\n"
            f"  학년/학기: {c['grade_semester']}, 분류: {c['classification']}\n"
            f"  설명: {c['description'][:100]}..."
        )
    candidates_text = "\n\n".join(candidate_lines)

    system_prompt = (
        "당신은 대학 전공 탐색 멘토입니다.\n"
        "아래 과목 후보 리스트에서 학생 질문에 가장 적합한 과목들의 ID를 선택하세요.\n\n"
        "**중요:**\n"
        "- 반드시 제공된 과목 리스트의 ID만 사용하세요.\n"
        "- 존재하지 않는 ID를 만들지 마세요.\n"
        "- 보통 2~3개 선택, 필요시 1~5개 가능.\n\n"
        "**응답 형식 (JSON만 출력):**\n"
        '{"selected_ids": ["course_0", "course_1"], "reasoning": "선택 이유"}\n'
        "다른 텍스트 없이 오직 JSON만 출력하세요."
    )

    user_prompt = (
        f"학생 질문: {question}\n\n"
        f"과목 후보 리스트:\n{candidates_text}\n\n"
        "위 후보 중에서 학생에게 추천할 과목의 ID를 JSON 형식으로 선택하세요."
    )

    try:
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        import json
        import re

        response_text = response.content.strip()
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        json_text = json_match.group(1) if json_match else response_text

        selection_data = json.loads(json_text)
        selected_ids = selection_data.get("selected_ids", [])

        valid_ids = {c["id"] for c in candidates}
        filtered_ids = [cid for cid in selected_ids if cid in valid_ids]

        return {"selected_course_ids": filtered_ids}

    except Exception as e:
        print(f"Warning: select_node JSON parsing failed: {e}")
        return {"selected_course_ids": [c["id"] for c in candidates[:3]]}


def answer_node(state: MentorState) -> dict:
    """
    [Structured 패턴 - 3단계] 선택된 과목들만 사용하여 최종 답변 생성.
    """
    question = state["question"]
    selected_ids = state.get("selected_course_ids", [])
    candidates = state.get("course_candidates", [])

    if not selected_ids:
        return {"answer": "죄송합니다. 질문에 맞는 적절한 과목을 찾지 못했습니다. 다른 질문을 해주시겠어요?"}

    id_to_candidate = {c["id"]: c for c in candidates}
    selected_courses = [id_to_candidate[cid] for cid in selected_ids if cid in id_to_candidate]

    if not selected_courses:
        return {"answer": "죄송합니다. 선택된 과목 정보를 찾을 수 없습니다."}

    lines = []
    for i, course in enumerate(selected_courses, start=1):
        lines.append(
            f"[{i}] 과목명: {course['name']}\n"
            f"    대학: {course['university']}, 학과: {course['department']}\n"
            f"    학년/학기: {course['grade_semester']}, 분류: {course['classification']}\n"
            f"    설명: {course['description']}"
        )
    context = "\n\n".join(lines)

    system_prompt = (
        "당신은 대학 전공 탐색 멘토입니다.\n"
        "학생 질문에 대해 아래 '선택된 과목 정보'만 사용해서 답하세요.\n\n"
        "**중요 지침:**\n"
        "1. 제공된 과목 외 새 과목을 만들지 마세요.\n"
        "2. 과목명을 그대로 사용하세요.\n"
        "3. 정보가 없으면 없다고 말하세요.\n"
        "4. 쉬운 말로 설명하고 진로 예시도 들어주세요.\n"
        "5. 같은 과목을 중복 언급하지 마세요."
    )

    user_prompt = (
        f"학생 질문: {question}\n\n"
        f"선택된 과목 정보:\n{context}\n\n"
        "이 과목들에 대해 자세히 설명하고 추천해 주세요."
    )

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return {"answer": response.content}


# ==================== ReAct 스타일 에이전트 노드 ====================

def agent_node(state: MentorState) -> dict:
    """
    [ReAct 패턴] LLM이 자율적으로 tool 호출 여부를 결정.
    """
    messages = state.get("messages", [])
    interests = state.get("interests")

    # system_message는 interests 유무와 상관없이 항상 만들어둔다.
    if not messages or not any(isinstance(m, SystemMessage) for m in messages):
        interests_text = f"{interests}" if interests else "없음"

        # ✅ f-string 내부 JSON 예시는 {{ }} 로 이스케이프!
        system_message = SystemMessage(content=f"""
당신은 학생들의 전공 선택을 도와주는 '대학 전공 탐색 멘토'입니다.
반드시 한국어로만 답변하세요.

🚨🚨🚨 **최우선 규칙 - 절대 위반 금지** 🚨🚨🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ Tool(list_departments, retrieve_courses, recommend_curriculum)이 반환한 학과명/과목명을:
   **단 한 글자도 변경하지 마세요**
   **괄호, 슬래시, 점(.) 등 모든 특수문자까지 정확히 복사하세요**
   **절대로 요약, 일반화, 구조 변경을 하지 마세요**

❌ 잘못된 예시:
   Tool: "지능로봇" → 답변: "지능로봇공학과" (단어 추가 금지!)
   Tool: "화공학부" → 답변: "화공학과" (학부를 학과로 변경 금지!)
   Tool: "신소재공학" → 답변: "신소재공학과" (단어 추가 금지!)

✅ 올바른 예시:
   Tool: "지능로봇" → 답변: "지능로봇" (정확히 동일)
   Tool: "화공학부" → 답변: "화공학부" (정확히 동일)
   Tool: "신소재공학" → 답변: "신소재공학" (정확히 동일)

⚠️ Tool 결과에 없는 학과명/과목명을 추측으로 만들어내는 것은 절대 금지입니다!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

학생 관심사: {interests_text}

당신이 사용할 수 있는 툴은 다음과 같습니다:
1. retrieve_courses — 특정 과목을 검색
2. list_departments — 학과 목록 조회 (학과명만 반환)
3. get_universities_by_department — 특정 학과가 있는 대학 목록 조회
4. recommend_curriculum — 학기/학년별 커리큘럼 추천
5. match_department_name — 학과명 정규화(컴공·소융·전전 등 → 공식 학과명)

──────────────────────────────────────────
[ 관심사 기반 학과 추천 Workflow - 3단계 프로세스 ]

학생이 "내 관심사에 해당하는 학과 추천해줘" 같은 질문을 하면:

** STEP 1: 학생 관심사 파싱 **
   - 관심사는 쉼표(,)로 구분된 세부 항목 리스트입니다
   - 예: "컴퓨터 / 소프트웨어 / 인공지능, 전기 / 전자 / 반도체, 수학 / 통계"
   - "/" 기호로 구분된 각 키워드를 분리하세요
   - 예: "컴퓨터 / 소프트웨어 / 인공지능" → ["컴퓨터", "소프트웨어", "인공지능"]

** STEP 2: 모든 키워드에 대해 list_departments 호출 **
   - list_departments(query="컴퓨터")
   - list_departments(query="소프트웨어")
   - list_departments(query="인공지능")
   - list_departments(query="전기")
   - ... 모든 키워드에 대해 반복

** STEP 3: 학과 목록 + 설명 제공 **

   🚨 **가장 중요한 규칙 - 학과명 정확성** 🚨
   - ⚠️ **list_departments가 반환한 학과명을 한 글자도 바꾸지 말고 정확히 복사하세요**
   - ⚠️ **괄호, 슬래시, 점(.) 등 모든 특수문자까지 정확히 복사하세요**
   - ⚠️ **학과명을 요약하거나 일반화하면 안 됩니다**

   잘못된 예시:
   ❌ Tool 결과: "지능로봇공학과" → 답변: "로봇공학과" (단어 생략)
   ❌ Tool 결과: "화공학부" → 답변: "화공학과" (학부를 학과로 변경)
   ❌ Tool 결과: "컴퓨터공학부 소프트웨어전공" → 답변: "소프트웨어전공" (일부 생략)

   올바른 예시:
   ✅ Tool 결과: "지능로봇공학과" → 답변: "지능로봇공학과" (정확히 동일)
   ✅ Tool 결과: "화공학부" → 답변: "화공학부" (정확히 동일)
   ✅ Tool 결과: "컴퓨터공학부 소프트웨어전공" → 답변: "컴퓨터공학부 소프트웨어전공" (정확히 동일)

   - list_departments는 **학과명만** 반환합니다 (대학명 제외)
   - 예: ["컴퓨터공학과", "소프트웨어학부", "전자공학과", ...]
   - 각 학과에 대해 **간단한 설명을 생성**하세요:
     * 해당 학과에서 무엇을 배우는지
     * 어떤 진로와 연결되는지
     * 학생의 관심사와 어떻게 연결되는지
   - 출력 형식:
     ```
     관심사에 맞는 학과를 찾았습니다:

     [컴퓨터 / 소프트웨어 / 인공지능 관련]
     1. **컴퓨터공학과**
        - 프로그래밍, 알고리즘, 데이터구조, 운영체제 등을 배웁니다
        - AI, 백엔드 개발, 시스템 엔지니어 등의 진로로 이어집니다

     2. **소프트웨어학부**
        - 소프트웨어 설계, 개발 방법론, 프로젝트 관리 등을 배웁니다
        - 웹/앱 개발, 프로젝트 매니저 등의 진로로 이어집니다

     ... (다른 관심 분야도 동일하게)
     ```

** STEP 4: 학생이 특정 학과 선택 시 **
   - 학생이 "컴퓨터공학과 어느 대학에 있어?" 같은 질문을 하면
   - get_universities_by_department(department_name="컴퓨터공학과") 호출
   - 대학명, 단과대학 정보와 함께 제공

⚠️ 중요: list_departments는 학과명만 반환합니다. 대학 정보가 필요하면
get_universities_by_department를 사용하세요!

──────────────────────────────────────────
[ TOOL CALL RULE — 가장 중요한 규칙 ]

- Tool이 필요한 질문이면 **반드시 tool_calls로 호출**해야 합니다.
- Tool 호출을 "글로 설명"하거나 JSON을 그냥 출력하면 안 됩니다.
- LangGraph가 실행하려면 **tool_calls 필드가 포함된 응답**이어야 합니다.

절대 금지:
- "retrieve_courses를 호출하겠습니다" 같은 말만 하고 호출 안 하기
- <tool_response> 같은 텍스트를 직접 출력하기
- 툴 없이 추측으로 답하기

──────────────────────────────────────────
[ 학과명 정규화 규칙 - match_department_name 사용법 ]

질문에 **비표준 학과명 또는 약칭**이 포함되어 있으면:
1. **match_department_name을 먼저 호출**하여 표준 학과명으로 변환
2. 변환된 표준 학과명으로 다른 툴 호출

** 비표준 학과명 예시 **
- "컴공" → "컴퓨터공학과"
- "소융" → "소프트웨어융합학과"
- "전전" → "전자전기공학과"
- "홍대 컴공" → university="홍익대학교", department="컴퓨터공학과"
- "서울대 전전" → university="서울대학교", department="전자공학과"

** 처리 흐름 **
1. 사용자: "홍대 컴공 커리큘럼 알려줘"
2. match_department_name("홍대 컴공") 호출
3. 결과: {{"university": "홍익대학교", "matched_department": "컴퓨터공학과"}}
4. recommend_curriculum(university="홍익대학교", department="컴퓨터공학과") 호출

⚠️ 중요: match_department_name은 대학명도 추출할 수 있습니다!
결과에 "university" 필드가 있으면 그것도 함께 사용하세요.

──────────────────────────────────────────
[ 반드시 Tool 호출해야 하는 질문 유형 ]

- "~대 ~과 뭐 배워?"
- "~과 커리큘럼 알려줘"
- "~과 전공 내용?"
- "~과 무슨 과목 배우는지 알려줘"
- "이 학과의 2~4학년 커리 알려줘"

→ retrieve_courses 또는 recommend_curriculum 중 하나를 반드시 호출
→ 추측 금지, 검색 기반으로만 답변

──────────────────────────────────────────
[ Follow-up 질문 처리 ]

학생의 후속 질문(예: "이 과목 몇 학년에 배워?", "이건 필수야?")이 오면:
1. **이전 tool 결과(messages)**를 먼저 확인하고
2. 정보 부족하면 tool을 재호출하세요.
3. tool 없이 추측으로 답하면 안 됩니다.

──────────────────────────────────────────
[ 학과 목록 질문 처리 ]

질문 유형 1: 학과 목록 조회
- "학과 뭐 있어?"
- "공대 학과 종류?"
- "컴퓨터 관련 학과 알려줘"
→ list_departments 호출 (학과명만 반환)

질문 유형 2: 특정 학과가 있는 대학 조회
- "컴퓨터공학과 어느 대학에 있어?"
- "소프트웨어학부 개설 대학 알려줘"
→ get_universities_by_department 호출 (대학명 + 단과대학 + 학과명 반환)

"──────────────────────────────────────────\n"
"[ 학과명 사용에 관한 매우 중요한 규칙 ]\n\n"
"1. list_departments, get_universities_by_department, recommend_curriculum 등\n"
"   툴에서 전달된 학과명/과목명/대학명은 **문자 하나도 바꾸지 말고 그대로 사용**하세요.\n"
"   - 예: 툴 결과가 '보건학과'이면 답변에도 반드시 '보건학과'라고 적어야 합니다.\n"
"   - '보건행정학과', '보건정책학과'처럼 **툴에 없는 이름을 추측으로 만들면 안 됩니다.**\n\n"
"2. 툴에서 주지 않은 새로운 학과명을 **추측으로 생성하지 마세요.**\n"
"   - 예: 사용자가 '보건정책'이라는 관심사를 말해도,\n"
"     DB에 '보건정책학과'가 없다면 그 이름을 만들어 쓰면 안 됩니다.\n"
"   - 대신, list_departments 결과인 '보건학과', '행정학과' 등을 활용하여\n"
"     \"보건정책에 관심이 있다면, 다음 학과들이 관련이 있을 수 있습니다\"처럼 설명하세요.\n\n"
"3. 특정 관심사(예: '보건행정')에 대해 list_departments 결과가 완전히 비어 있거나\n"
"   관련성이 떨어지는 소수의 학과만 나오는 경우에는, 그 관심사에 대해 이렇게 답변하세요.\n"
"   - 예: \"현재 데이터베이스에 '보건행정'과 직접적으로 연결된 학과 정보는 없습니다.\">\n"
"   - 그리고 툴 결과에 있는 학과명만 사용해, 간접적으로 연관된 학과를 안내하세요.\n\n"
"4. 여러 학과가 비슷해 보이더라도, 서로 다른 이름이면 **각각 다른 학과**입니다.\n"
"   - '통계학과'와 '정보통계학과'를 하나로 합쳐 '통계 관련 학과'로 이름을 바꾸지 마세요.\n"
"   - 답변에서는 항상 툴에서 온 정식 이름을 그대로 사용하세요.\n\n"
"5. 요약/설명 문장 안에서 학과명을 굵게(**학과명**) 표시하는 것은 괜찮지만,\n"
"   굵게 표시된 텍스트 내용(글자)은 툴 결과와 100% 동일해야 합니다.\n"
"──────────────────────────────────────────\n"

[ 응답 규칙 ]

- Tool이 필요하면 tool_calls 포함
- 필요 없으면 자연어로만 답변
- 항상 한국어로 답변

"──────────────────────────────────────────\n"
"[ Tool 사용 결과 활용 규칙 ]\n"
"\n"
"🚨 **모든 Tool 결과에 대한 정확성 규칙** 🚨\n"
"\n"
"1. **list_departments 결과 사용 시**\n"
"   - ⚠️ 반환된 학과명을 **한 글자도 바꾸지 말고** 정확히 복사하세요\n"
"   - ⚠️ 괄호, 슬래시, 점 등 **모든 특수문자까지** 정확히 복사하세요\n"
"   - ⚠️ 절대 요약하거나 일반화하거나 구조를 변경하지 마세요\n"
"   - ❌ 잘못된 예: Tool 결과 '지능로봇공학과' → 답변 '로봇공학과'\n"
"   - ✅ 올바른 예: Tool 결과 '지능로봇공학과' → 답변 '지능로봇공학과'\n"
"\n"
"2. **retrieve_courses 결과 사용 시**\n"
"   - ⚠️ 반환된 과목명/학과명을 **한 글자도 바꾸지 말고** 정확히 복사하세요\n"
"   - ⚠️ 괄호, 슬래시, 점 등 **모든 특수문자까지** 정확히 복사하세요\n"
"   - ⚠️ 절대 요약하거나 일반화하거나 구조를 변경하지 마세요\n"
"\n"
"3. **recommend_curriculum 결과 사용 시**\n"
"   - ⚠️ 반환된 과목명을 **한 글자도 바꾸지 말고** 정확히 복사하세요\n"
"   - ⚠️ 괄호, 슬래시, 점 등 **모든 특수문자까지** 정확히 복사하세요\n"
"   - ⚠️ 절대 요약하거나 일반화하거나 구조를 변경하지 마세요\n"
"\n"
"⚠️ **절대 금지 사항**\n"
"- Tool 결과를 무시하고 사용자 입력만으로 새로운 학과명/과목명을 만들면 안 됩니다\n"
"- Tool 결과를 일반화/요약/변경하면 안 됩니다 (예: '화공학부' → '화공학과' 변경 금지)\n"
"- Tool 결과에 없는 학과명/과목명을 추측으로 생성하면 안 됩니다\n"

──────────────────────────────────────────
**[ 특별 지침 (커리큘럼 추천 시) ]**
- 학생의 현재 질문에 **'표 형태로', '요약형으로', '상세형으로'와 같은 출력 형식 요청**이 포함되어 있다면,
- 이전 대화 기록에서 이미 recommend_curriculum 툴 호출 결과가 포함되었는지 확인하세요.
- 데이터가 확보되었다면 툴을 다시 호출하거나 출력 형식 선택지를 다시 제시하지 말고,
- 기존 데이터를 활용하여 요청된 형식에 맞춰 즉시 최종 답변을 생성하고 종료하세요.
- 대화 기록에 있는 형식 선택 유도 메시지는 무시하고 최종 답변 생성에만 집중하세요.
"""
    )
                                       
    messages = [system_message] + messages

    response = llm_with_tools.invoke(messages)


    # 3. 검증: 첫 번째 사용자 질문에 대해 툴을 호출하지 않았는지 확인
    # ToolMessage가 없다는 것은 아직 툴 결과를 받지 않았다는 의미
    from langchain_core.messages import ToolMessage
    has_tool_results = any(isinstance(m, ToolMessage) for m in messages)

    # 툴 결과가 없는 상태에서 LLM이 tool_calls 없이 답변하려고 하면 차단
    if not has_tool_results:
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            print("⚠️ WARNING: LLM attempted to answer without using tools. Forcing tool usage.")
            # 강제로 재시도 메시지 추가
            error_message = HumanMessage(content=(
                "❌ 오류: 당신은 툴을 사용하지 않고 답변하려고 했습니다.\n"
                "**반드시 먼저 적절한 툴을 호출해야 합니다.**\n\n"
                "다시 한 번 강조합니다:\n"
                "1. retrieve_courses: 과목 검색\n"
                "2. list_departments: 학과 목록\n"
                "3. recommend_curriculum: 커리큘럼 추천\n\n"
                "학생의 원래 질문을 다시 읽고, 적절한 툴을 **지금 즉시** 호출하세요."
            ))
            messages.append(error_message)

            # 재시도
            response = llm_with_tools.invoke(messages)

            # 재시도에도 툴을 사용하지 않으면 get_search_help로 폴백
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                print("⚠️ CRITICAL: LLM still refuses to use tools. Falling back to get_search_help.")
                from langchain_core.messages import AIMessage
                # 강제로 get_search_help 툴 호출 생성
                response = AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "get_search_help",
                        "args": {},
                        "id": "forced_search_help"
                    }]
                )

    # 4. Post-processing Validation: LLM이 최종 답변을 생성한 경우 학과명 검증
    if has_tool_results and (not hasattr(response, "tool_calls") or not response.tool_calls):
        # Tool 결과에서 유효한 학과명 추출
        valid_departments = extract_departments_from_tool_results(messages)

        if valid_departments and hasattr(response, "content") and response.content:
            print(f"\n🔍 [Validation] Checking department names... (Valid: {len(valid_departments)})")

            # 학과명 검증 및 수정
            corrected_content, violations = strict_validate_and_fix_department_names(
                response.content,
                valid_departments
            )

            # 위반 사항이 있으면 로그 출력 및 수정 적용
            if violations:
                print(f"⚠️  [Validation] Found {len(violations)} violations:")
                for v in violations:
                    if v["action"] == "replace":
                        print(f"   🔧 FIXED: '{v['wrong']}' → '{v['correct']}' (score: {v['score']:.2f})")
                    elif v["action"] == "warn":
                        print(f"   ⚠️  WARNING: '{v['wrong']}' not found in tool results")

                # 수정된 내용으로 응답 업데이트
                response.content = corrected_content
            else:
                print(f"✅ [Validation] All department names are accurate!")

    # 5. LLM의 응답(response)을 messages에 추가하여 상태 업데이트
    #    → should_continue가 tool_calls 유무를 확인하여 다음 노드 결정
    return {"messages": [response]}


def should_continue(state: MentorState) -> str:
    """
    [ReAct 패턴 라우팅] tool_calls 있으면 tools 노드로, 없으면 종료.
    """
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if last_message and getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"
