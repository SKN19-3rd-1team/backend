"""
전공 탐색 멘토 챗봇 - Streamlit Frontend

대학 과목 정보를 기반으로 학생들에게 맞춤 과목 추천과 진로 상담을 제공하는 챗봇 UI입니다.
백엔드의 LangGraph 기반 RAG 시스템과 연결되어 실시간으로 정보를 검색하고 답변합니다.

** 주요 기능 **
1. 채팅 기반 인터페이스 (Streamlit Chat)
2. 관심사 입력 기능 (사이드바)
3. 대화 기록 관리 (Session State)
4. 실시간 응답 (run_mentor 함수 호출)

** 실행 방법 **
```bash
streamlit run frontend/app.py
```
"""
# frontend/app.py
import streamlit as st
from pathlib import Path
import sys

# ==================== 경로 설정 ====================
# backend 모듈을 import하기 위해 프로젝트 루트를 Python 경로에 추가
ROOT_DIR = Path(__file__).resolve().parents[1]  # frontend의 부모 = 프로젝트 루트
sys.path.append(str(ROOT_DIR))

# ==================== Backend 모듈 Import ====================
from backend.main import run_mentor, run_major_recommendation  # 백엔드 메인 함수
from backend.config import get_settings  # 설정 로드

# ==================== 설정 로드 및 콘솔 출력 ====================
settings = get_settings()
print(
    f"[Mentor Console] Using provider '{settings.llm_provider}' "
    f"with model '{settings.model_name}'"
)

# ==================== 카테고리 및 온보딩 정의 ====================

ONBOARDING_QUESTIONS = [
    {
        "key": "subjects",
        "label": "선호 고교 과목",
        "prompt": "안녕하세요! 가장 좋아하거나 자신 있는 고등학교 과목은 무엇인가요? 좋아하는 이유도 함께 알려주세요.",
        "placeholder": "예: 수학과 물리를 특히 좋아하고 실험 수업을 즐깁니다."
    },
    {
        "key": "interests",
        "label": "흥미 및 취미",
        "prompt": "학교 밖에서는 어떤 주제나 취미에 가장 흥미를 느끼나요?",
        "placeholder": "예: 로봇 동아리 활동, 디지털 드로잉, 음악 감상 등"
    },
    {
        "key": "desired_salary",
        "label": "희망 연봉",
        "prompt": "졸업 후 어느 정도의 연봉을 희망하나요? 대략적인 수준을 알려주세요.",
        "placeholder": "예: 연 4천만 원 이상이면 좋겠습니다."
    },
    {
        "key": "preferred_majors",
        "label": "희망 학과",
        "prompt": "가장 진학하고 싶은 학과나 전공은 무엇인가요? 복수로 답해도 괜찮아요.",
        "placeholder": "예: 컴퓨터공학과, 데이터사이언스학과"
    },
]

# ==================== Streamlit 페이지 설정 ====================
st.set_page_config(
    page_title="전공 탐색 멘토",
    page_icon="🎓",
    layout="wide"  # 넓은 레이아웃
)

# ==================== Session State 초기화 ====================
# Streamlit Session State: 페이지 리로드 시에도 유지되는 상태 저장소

# 채팅 기록 초기화 (사용자와 챗봇의 대화 내용)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 관심사 초기화 (사용자가 입력한 관심 분야/진로 방향)
if "interests" not in st.session_state:
    st.session_state.interests = ""



if "button_prompt" not in st.session_state:
    st.session_state.button_prompt = None
if 'format_pending' not in st.session_state:
    st.session_state.format_pending = False

if "onboarding_step" not in st.session_state:
    st.session_state.onboarding_step = 0
if "onboarding_answers" not in st.session_state:
    st.session_state.onboarding_answers = {q["key"]: "" for q in ONBOARDING_QUESTIONS}
if "onboarding_complete" not in st.session_state:
    st.session_state.onboarding_complete = False
if "major_recommendations" not in st.session_state:
    st.session_state.major_recommendations = None
if "major_profile_text" not in st.session_state:
    st.session_state.major_profile_text = ""
if "major_scores" not in st.session_state:
    st.session_state.major_scores = {}
if "major_hits" not in st.session_state:
    st.session_state.major_hits = []
if "major_recommendation_error" not in st.session_state:
    st.session_state.major_recommendation_error = None
if "new_major_summary" not in st.session_state:
    st.session_state.new_major_summary = False
if "force_recalc_major" not in st.session_state:
    st.session_state.force_recalc_major = False




def ensure_onboarding_flow():
    """초기 4단계 선호도 조사가 끝날 때까지 채팅 UI를 잠시 숨긴다."""
    if st.session_state.onboarding_complete:
        return

    st.subheader("🧑‍🏫 먼저 간단한 선호도 조사를 진행해볼게요")
    st.info("아래 4가지 질문에 순서대로 답해주시면 맞춤형 전공 추천을 준비할 수 있어요.")
    step = st.session_state.onboarding_step

    # 이전 질문/답변을 간단한 대화 형태로 보여주기
    for idx in range(step):
        q = ONBOARDING_QUESTIONS[idx]
        answer = st.session_state.onboarding_answers.get(q["key"], "")
        with st.chat_message("assistant"):
            st.markdown(q["prompt"])
        if answer:
            with st.chat_message("user"):
                st.markdown(answer)

    current = ONBOARDING_QUESTIONS[step]
    with st.chat_message("assistant"):
        st.markdown(current["prompt"])

    form_key = f"onboarding_form_{step}"
    input_key = f"onboarding_input_{step}"
    with st.form(form_key, clear_on_submit=False):
        response = st.text_input(
            "답변을 입력해 주세요",
            value=st.session_state.onboarding_answers.get(current["key"], ""),
            key=input_key,
            placeholder=current.get("placeholder", "")
        )
        submitted = st.form_submit_button("다음 질문")

    if submitted:
        if not response.strip():
            st.warning("답변을 입력해 주세요.")
        else:
            st.session_state.onboarding_answers[current["key"]] = response.strip()
            st.session_state.onboarding_step += 1
            if st.session_state.onboarding_step >= len(ONBOARDING_QUESTIONS):
                st.session_state.onboarding_complete = True
            st.rerun()

    st.stop()


def ensure_major_recommendations(force: bool = False):
    """온보딩이 완료되면 Pinecone 기반 전공 추천을 호출한다."""
    if not st.session_state.onboarding_complete:
        return

    needs_fetch = force or st.session_state.major_recommendations is None
    if not needs_fetch:
        return

    st.session_state.major_recommendation_error = None
    with st.spinner("온보딩 정보를 바탕으로 전공을 분석 중입니다..."):
        try:
            result = run_major_recommendation(
                onboarding_answers=st.session_state.onboarding_answers,
                question=None
            )
        except Exception as exc:
            st.session_state.major_recommendations = []
            st.session_state.major_profile_text = ""
            st.session_state.major_scores = {}
            st.session_state.major_hits = []
            st.session_state.major_recommendation_error = str(exc)
            st.session_state.new_major_summary = False
            return

    st.session_state.major_recommendations = result.get("recommended_majors", [])
    st.session_state.major_profile_text = result.get("user_profile_text", "")
    st.session_state.major_scores = result.get("major_scores", {})
    st.session_state.major_hits = result.get("major_search_hits", [])
    st.session_state.new_major_summary = True


def render_major_recommendations_section():
    """추천된 전공을 카드 형태로 정리."""
    st.subheader("🧭 맞춤 전공 추천 결과")

    if st.session_state.major_recommendation_error:
        st.error(
            "전공 추천 중 오류가 발생했습니다. 다시 시도해 주세요.\n\n"
            f"상세 메시지: {st.session_state.major_recommendation_error}"
        )
        if st.button("🔁 다시 시도", key="retry_major_rec"):
            st.session_state.major_recommendations = None
            st.session_state.major_recommendation_error = None
            st.session_state.force_recalc_major = True
            st.rerun()
        return

    recs = st.session_state.major_recommendations
    if recs is None:
        st.info("추천 정보를 불러오는 중입니다...")
        return

    if not recs:
        st.warning("조건에 맞는 전공을 찾지 못했습니다. 답변을 조금 더 구체적으로 적어보세요.")
    else:
        if st.session_state.major_profile_text:
            st.caption("학생 프로필 요약")
            st.code(st.session_state.major_profile_text.strip())

        for idx, major in enumerate(recs[:5], start=1):
            score = major.get("score", 0.0)
            cluster = major.get("cluster") or "계열 정보 없음"
            salary = major.get("salary")
            tags = major.get("relate_subject_tags", [])[:5]
            doc_types = ", ".join(
                f"{doc_type}({doc_score:.2f})"
                for doc_type, doc_score in major.get("top_doc_types", [])
            )
            with st.container():
                st.markdown(f"**{idx}. {major['major_name']}** · 점수 {score:.2f}")
                st.write(f"- 계열: {cluster}")
                if salary is not None:
                    salary_text = f"{salary}만원" if isinstance(salary, (int, float)) else f"{salary}"
                    st.write(f"- 평균 초봉 지표: {salary_text}")
                if doc_types:
                    st.write(f"- 주요 근거: {doc_types}")
                if tags:
                    st.write(f"- 연관 과목 태그: {', '.join(tags)}")
                
                # summary 필드 표시
                summary_text = major.get("summary", "")
                if summary_text:
                    st.caption("상세 설명")
                    st.markdown(summary_text)

    rerun_col1, rerun_col2 = st.columns([1, 4])
    with rerun_col1:
        if st.button("🔁 전공 추천 다시 분석", key="rerun_major_button"):
            st.session_state.major_recommendations = None
            st.session_state.major_recommendation_error = None
            st.session_state.force_recalc_major = True
            st.rerun()
    with rerun_col2:
        st.caption("답변을 조금 더 구체적으로 수정하면 추천 정확도가 올라갑니다.")


def sync_major_summary_message():
    """새 추천 결과가 있을 때 챗봇 대화에도 요약을 남긴다."""
    if not st.session_state.get("new_major_summary"):
        return

    recs = st.session_state.major_recommendations or []
    if not recs:
        st.session_state.new_major_summary = False
        return

    lines = []
    for idx, major in enumerate(recs[:5], start=1):
        score = major.get("score", 0.0)
        lines.append(f"{idx}. {major['major_name']} (점수 {score:.2f})")
    summary_text = (
        "온보딩 답변을 바탕으로 추천 전공 TOP 5를 정리했어요:\n"
        + "\n".join(lines)
        + "\n\n필요하면 위 전공 중 궁금한 학과를 지정해서 더 물어봐도 좋아요!"
    )
    st.session_state.messages.append({"role": "assistant", "content": summary_text})
    st.session_state.new_major_summary = False


st.title("🎓 전공 탐색 멘토 챗봇")
st.write("이공계열 과목들을 기반으로, 나에게 맞는 과목과 진로를 함께 고민해보는 멘토 챗봇입니다.")

# 온보딩 설문이 끝나지 않았다면 즉시 진행
ensure_onboarding_flow()

# 온보딩 완료 후 전공 추천 실행 및 요약 표시
force_flag = st.session_state.force_recalc_major
if force_flag:
    st.session_state.force_recalc_major = False
ensure_major_recommendations(force=force_flag)
render_major_recommendations_section()
st.divider()

# 커리큘럼 키워드 감지 함수
def is_curriculum_query(text: str) -> bool:
    keywords = ["커리큘럼", "학기별", "전체 커리큘럼", "학년별", "수업 순서", "커리큘럼을"]
    return any(keyword in text for keyword in keywords)

# 버튼 렌더링 함수
def render_format_options_inline(original_question: str):
    option_labels = ["요약형", "상세형", "표 형태"]
    st.write("원하시는 출력 형식을 선택해 주세요")
    cols = st.columns(len(option_labels))
    for i, label in enumerate(option_labels):
        with cols[i]:
            st.button(label, on_click=handle_button_click, args=[label], key=f"inline_opt_{label}")

# 버튼 클릭 처리 함수
def handle_button_click(selection: str):
    original_question = ""
    for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                original_question = msg["content"]
                break

    display_prompt = f"{original_question}을 {selection}으로 보여줘"
    st.session_state.button_prompt = display_prompt


with st.sidebar:
    st.header("나에 대한 정보")
    st.info("온보딩 질문에 답변하시면 맞춤형 전공 추천을 받으실 수 있습니다.")



# ==================== 채팅 기록 표시 ====================
# Session State에 저장된 이전 대화 내용을 화면에 표시
sync_major_summary_message()
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        # "user" 또는 "assistant" 역할에 맞는 채팅 메시지 UI 생성
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = None

new_input = st.chat_input("궁금한 점을 물어보세요!")

# 버튼 클릭으로 생성된 프롬프트 처리
if st.session_state.button_prompt:
    prompt = st.session_state.button_prompt
    st.session_state.button_prompt = None
elif new_input:
    # 일반 텍스트 입력 처리
    prompt = new_input

# Chat input
if prompt:
    if is_curriculum_query(prompt) and not st.session_state.button_prompt and not st.session_state.format_pending:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        render_format_options_inline(prompt)
        st.session_state.format_pending = True
        st.stop()

    # If we are resuming after the user chose a format (button_prompt was set), avoid duplicating the user message
    if st.session_state.format_pending and st.session_state.button_prompt is None:
        pass

    # Add user message to chat history if not already added by format flow
    if not st.session_state.format_pending:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        display_content = prompt
    else:
        # We're resuming after a format selection; show the original user message
        display_content = None
        for msg in reversed(st.session_state.messages):
            if msg.get("role") == "user":
                display_content = msg.get("content")
                break

        if display_content is None:
            display_content = prompt

    # 3. 백엔드 호출하여 답변 생성
    with st.chat_message("assistant"):
        # 로딩 스피너 표시
        with st.spinner("멘토가 과목 정보를 검토 중입니다..."):
            run_question = prompt
            if st.session_state.get('internal_marker'):
                run_question = f"{prompt} {st.session_state.get('internal_marker')}"

            # 온보딩 프로필을 컨텍스트로 전달
            profile_context = (
                st.session_state.interests
                or st.session_state.major_profile_text
            )

            raw_response: str | dict = run_mentor(
                question=run_question,
                interests=profile_context or None,
                chat_history=st.session_state.messages
            )

            if st.session_state.get('internal_marker'):
                del st.session_state['internal_marker']
        
        # 일반 텍스트 응답 처리
        response_content = raw_response
        st.markdown(response_content) # 일반 텍스트는 즉시 출력

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})

    if st.session_state.format_pending:
        st.session_state.format_pending = False
        st.session_state.button_prompt = None
        if 'format_origin' in st.session_state:
            del st.session_state['format_origin']
