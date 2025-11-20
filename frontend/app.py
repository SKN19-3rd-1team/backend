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

st.title("🎓 전공 탐색 멘토 챗봇")
st.write("컴퓨터공학 전공 과목들을 기반으로, 나에게 맞는 과목과 진로를 함께 고민해보는 멘토 챗봇입니다.")

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
        st.rerun()

# Display chat messages from history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("궁금한 점을 물어보세요 (예: 홍익대학교 컴퓨터공학과 2학년 과목 추천해줘)"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("멘토가 과목 정보를 검토 중입니다..."):
            response = run_mentor(
                question=prompt,
                interests=st.session_state.interests or None
            )
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
