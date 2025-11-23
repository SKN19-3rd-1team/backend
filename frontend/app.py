# frontend/app.py
import streamlit as st
from pathlib import Path
import sys

# backend 모듈 import를 위해 경로 추가
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from backend.main import run_mentor
from backend.config import get_settings

settings = get_settings()
print(
    f"[Mentor Console] Using provider '{settings.llm_provider}' "
    f"with model '{settings.model_name}'"
)

st.set_page_config(page_title="전공 탐색 멘토", page_icon="🎓", layout="wide")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize interests in session state
if "interests" not in st.session_state:
    st.session_state.interests = ""

if "button_prompt" not in st.session_state:
    st.session_state.button_prompt = None
if 'format_pending' not in st.session_state:
    st.session_state.format_pending = False
    
st.title("🎓 전공 탐색 멘토 챗봇")
st.write("컴퓨터공학 전공 과목들을 기반으로, 나에게 맞는 과목과 진로를 함께 고민해보는 멘토 챗봇입니다.")

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
    interests = st.text_area(
        "관심사 / 진로 방향 (선택)",
        value=st.session_state.interests,
        placeholder="예: AI, 데이터 분석, 스타트업, 백엔드, 보안 등",
        key="interests_input"
    )
    # Update session state when interests change
    st.session_state.interests = interests

    # Clear chat history button
    if st.button("🗑️ 대화 기록 초기화"):
        st.session_state.messages = []
        st.session_state.button_prompt = None
        st.session_state.format_pending = False
        st.stop()


# Display chat messages from history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = None

new_input = st.chat_input("궁금한 점을 물어보세요 (예: 홍익대학교 컴퓨터공학과 2학년 과목 추천해줘)")

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

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("멘토가 과목 정보를 검토 중입니다..."):
            run_question = prompt
            if st.session_state.get('internal_marker'):
                run_question = f"{prompt} {st.session_state.get('internal_marker')}"

            raw_response: str | dict = run_mentor( 
                question=run_question,
                interests=st.session_state.interests or None,
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