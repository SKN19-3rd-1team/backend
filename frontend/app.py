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

# ==================== 메인 UI ====================
st.title("🎓 전공 탐색 멘토 챗봇")
st.write("컴퓨터공학 전공 과목들을 기반으로, 나에게 맞는 과목과 진로를 함께 고민해보는 멘토 챗봇입니다.")

# ==================== 사이드바: 사용자 정보 입력 ====================
with st.sidebar:
    st.header("나에 대한 정보")

    # 관심사 입력 영역
    interests = st.text_area(
        "관심사 / 진로 방향 (선택)",
        value=st.session_state.interests,
        placeholder="예: AI, 데이터 분석, 스타트업, 백엔드, 보안 등",
        key="interests_input"
    )
    # Session State 업데이트 (입력값 저장)
    st.session_state.interests = interests

    # 대화 기록 초기화 버튼
    if st.button("🗑️ 대화 기록 초기화"):
        st.session_state.messages = []  # 채팅 기록 삭제
        st.rerun()  # 페이지 새로고침

# ==================== 채팅 기록 표시 ====================
# Session State에 저장된 이전 대화 내용을 화면에 표시
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        # "user" 또는 "assistant" 역할에 맞는 채팅 메시지 UI 생성
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ==================== 채팅 입력 및 응답 처리 ====================
# st.chat_input(): Streamlit의 채팅 입력창 (하단 고정)
if prompt := st.chat_input("궁금한 점을 물어보세요 (예: 홍익대학교 컴퓨터공학과 2학년 과목 추천해줘)"):
    # 1. 사용자 메시지를 채팅 기록에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 사용자 메시지 화면에 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. 백엔드 호출하여 답변 생성
    with st.chat_message("assistant"):
        # 로딩 스피너 표시
        with st.spinner("멘토가 과목 정보를 검토 중입니다..."):
            # 백엔드 run_mentor() 함수 호출
            # - question: 사용자 질문
            # - interests: 사용자 관심사 (있으면 전달, 없으면 None)
            response = run_mentor(
                question=prompt,
                interests=st.session_state.interests or None
            )
        # 4. 답변 화면에 표시
        st.markdown(response)

    # 5. 챗봇 답변을 채팅 기록에 추가
    st.session_state.messages.append({"role": "assistant", "content": response})
