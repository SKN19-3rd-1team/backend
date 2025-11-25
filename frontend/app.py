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
from backend.main import run_mentor  # 백엔드 메인 함수
from backend.config import get_settings  # 설정 로드

# ==================== 설정 로드 및 콘솔 출력 ====================
settings = get_settings()
print(
    f"[Mentor Console] Using provider '{settings.llm_provider}' "
    f"with model '{settings.model_name}'"
)

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

# 대단위 카테고리 선택 초기화
if "selected_main_categories" not in st.session_state:
    st.session_state.selected_main_categories = []

# 세부 체크리스트 선택 초기화
if "selected_subcategories" not in st.session_state:
    st.session_state.selected_subcategories = {}

if "button_prompt" not in st.session_state:
    st.session_state.button_prompt = None
if 'format_pending' not in st.session_state:
    st.session_state.format_pending = False
    
st.title("🎓 전공 탐색 멘토 챗봇")
st.write("이공계열 과목들을 기반으로, 나에게 맞는 과목과 진로를 함께 고민해보는 멘토 챗봇입니다.")

# ==================== 카테고리 데이터 정의 ====================
MAIN_CATEGORIES = {
    "공학": ["컴퓨터 / 소프트웨어 / 인공지능", "전기 / 전자 / 반도체", "기계 / 자동차 / 로봇",
             "화학 / 화공 / 신소재", "산업공학 / 시스템 / 데이터분석", "건축 / 토목 / 도시",
             "에너지 / 환경 / 원자력"],
    "자연과학": ["수학 / 통계", "물리 / 천문", "화학", "생명과학 / 바이오", "지구과학 / 환경"],
    "의약·보건": ["약학", "간호", "보건행정 / 보건정책"],
    "경영·경제·회계": ["경영(마케팅, 인사, 전략 등)", "경제 / 금융 / 금융공학", "회계 / 세무"],
    "사회과학": ["행정 / 정책", "정치 / 외교 / 국제관계", "사회 / 사회복지",
                "심리 / 상담", "언론 / 미디어 / 광고 / PR"],
    "인문": ["국어 / 문학", "영어 / 외국어", "역사 / 고고학", "철학 / 인류학 / 종교학"],
    "교육": ["교육학 / 교과교육(국영수 등)", "유아교육 / 특수교육"],
    "예체능": ["미술 / 회화 / 조소", "디자인(시각, 산업, UX/UI 등)",
             "음악 / 작곡 / 연주 / 보컬", "체육 / 스포츠 / 운동재활"],
    "융합/신산업": ["데이터사이언스 / 빅데이터", "인공지능 / 로봇 / 자율주행",
                  "게임 / 인터랙티브콘텐츠", "영상 / 콘텐츠 / 유튜브 / 방송",
                  "스타트업 / 창업"]
}

# 선택된 카테고리를 텍스트로 포맷팅하는 함수
def format_interests_from_selection():
    """선택된 대단위 카테고리와 세부 항목을 구조화된 텍스트로 변환 (UI 표시용)"""
    if not st.session_state.selected_main_categories:
        return ""

    interests_parts = []
    for main_cat in st.session_state.selected_main_categories:
        subcats = st.session_state.selected_subcategories.get(main_cat, [])
        if subcats:
            interests_parts.append(f"{main_cat}: {', '.join(subcats)}")
        else:
            interests_parts.append(main_cat)

    return " | ".join(interests_parts)

def format_interests_for_llm():
    """세부 관심사만 추출하여 LLM이 파싱하기 쉬운 형태로 변환"""
    if not st.session_state.selected_main_categories:
        return ""

    all_subcats = []
    for main_cat in st.session_state.selected_main_categories:
        subcats = st.session_state.selected_subcategories.get(main_cat, [])
        if subcats:
            all_subcats.extend(subcats)
        else:
            # 세부 항목이 없으면 대단위 카테고리를 사용
            all_subcats.append(main_cat)

    # 쉼표로 구분된 리스트 형태로 반환
    return ", ".join(all_subcats)

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

    # ==================== 1. 대단위 카테고리 선택 (최대 2개) ====================
    st.subheader("1️⃣ 관심 분야 선택 (최대 2개)")
    st.caption("아래 분야에서 관심 있는 분야를 2개까지 선택해주세요.")

    selected_main = []
    # 현재 선택된 항목 수를 먼저 계산
    for category in MAIN_CATEGORIES.keys():
        if category in st.session_state.selected_main_categories:
            selected_main.append(category)

    # 체크박스 렌더링 (현재 선택 수 기준으로 비활성화)
    temp_selected = []
    for category in MAIN_CATEGORIES.keys():
        is_checked = st.checkbox(
            category,
            value=(category in st.session_state.selected_main_categories),
            key=f"main_{category}",
            disabled=(len(selected_main) >= 2 and
                     category not in st.session_state.selected_main_categories)
        )
        if is_checked:
            temp_selected.append(category)

    st.session_state.selected_main_categories = temp_selected

    # ==================== 2. 세부 체크리스트 ====================
    if st.session_state.selected_main_categories:
        st.divider()
        st.subheader("2️⃣ 세부 관심 분야 선택")
        st.caption("선택한 분야에서 구체적으로 끌리는 키워드를 골라주세요.")

        for main_cat in st.session_state.selected_main_categories:
            with st.expander(f"📌 {main_cat}", expanded=True):
                subcategories = MAIN_CATEGORIES[main_cat]
                selected_subs = []

                for subcat in subcategories:
                    if st.checkbox(
                        subcat,
                        value=(subcat in st.session_state.selected_subcategories.get(main_cat, [])),
                        key=f"sub_{main_cat}_{subcat}"
                    ):
                        selected_subs.append(subcat)

                st.session_state.selected_subcategories[main_cat] = selected_subs

    # ==================== 선택 결과 미리보기 ====================
    formatted_interests = format_interests_from_selection()
    if formatted_interests:
        st.divider()
        st.subheader("✅ 선택한 관심사")
        st.info(formatted_interests)
        # interests 필드 자동 업데이트
        st.session_state.interests = formatted_interests

    # ==================== 추가 관심사 입력 (선택) ====================
    st.divider()
    st.subheader("💬 추가 관심사 (선택)")
    additional_interests = st.text_area(
        "자유롭게 입력",
        value="" if formatted_interests else st.session_state.interests,
        placeholder="예: AI, 데이터 분석, 스타트업, 백엔드, 보안 등",
        key="additional_interests_input",
        height=80
    )

    # 추가 관심사가 있으면 기존 선택과 결합
    if additional_interests and formatted_interests:
        st.session_state.interests = f"{formatted_interests} | {additional_interests}"
    elif additional_interests:
        st.session_state.interests = additional_interests
    elif formatted_interests:
        st.session_state.interests = formatted_interests

    # 카테고리 초기화 버튼
    st.divider()
    if st.button("🔄 카테고리 초기화"):
        st.session_state.selected_main_categories = []
        st.session_state.selected_subcategories = {}
        st.session_state.interests = ""
        st.rerun()


# ==================== 채팅 기록 표시 ====================
# Session State에 저장된 이전 대화 내용을 화면에 표시
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

            # LLM이 파싱하기 쉬운 형태로 관심사 전달 (세부 항목만)
            llm_interests = format_interests_for_llm() or st.session_state.interests

            raw_response: str | dict = run_mentor(
                question=run_question,
                interests=llm_interests or None,
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
