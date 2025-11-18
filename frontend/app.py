# frontend/app.py
import streamlit as st
from pathlib import Path
import sys

# backend 모듈 import를 위해 경로 추가
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from backend.main import run_mentor

st.set_page_config(page_title="전공 탐색 멘토", page_icon="🎓", layout="wide")

st.title("🎓 전공 탐색 멘토 챗봇")
st.write("컴퓨터공학 전공 과목들을 기반으로, 나에게 맞는 과목과 진로를 함께 고민해보는 멘토 챗봇입니다.")

with st.sidebar:
    st.header("나에 대한 정보")
    interests = st.text_area(
        "관심사 / 진로 방향 (선택)",
        placeholder="예: AI, 데이터 분석, 스타트업, 백엔드, 보안 등"
    )

question = st.text_area(
    "궁금한 점을 물어보세요",
    placeholder="예: AI 관련 과목들 중에서 2학년 때 들을만한 수업 추천해줘",
    height=150,
)

if st.button("질문하기", type="primary") and question.strip():
    with st.spinner("멘토가 과목 정보를 검토 중입니다..."):
        answer = run_mentor(question=question, interests=interests or None)
    st.markdown("### 멘토 답변")
    st.write(answer)
