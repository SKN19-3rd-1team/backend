"""
Helper functions for agent_node preprocessing
"""
import re

def is_single_major_query(query: str) -> bool:
    """
    단일 학과명 질문인지 감지
    
    패턴:
    - "고분자공학과"
    - "컴퓨터공학"
    - "경영학"
    - 짧은 질문 (10자 이하)
    - 학과/공학/교육 등 키워드 포함
    """
    query = query.strip()
    
    # 너무 긴 질문은 제외
    if len(query) > 20:
        return False
    
    # 질문이 명확한 경우 제외
    question_words = ['어디', '어떤', '무엇', '왜', '어떻게', '얼마', '?', '알려', '추천', '비슷', '유사']
    if any(word in query for word in question_words):
        return False
    
    # 학과 관련 키워드 포함 여부
    major_keywords = ['과', '학과', '공학', '교육', '대학']
    has_major_keyword = any(keyword in query for keyword in major_keywords)
    
    # 짧고 학과 키워드가 있으면 단일 학과명 쿼리로 판단
    if len(query) <= 15 and has_major_keyword:
        return True
    
    return False


def enhance_single_major_query(query: str) -> str:
    """
    단일 학과명 질문을 명확한 질문으로 변환
    
    예:
    "고분자공학과" → "고분자공학과에 대해 자세히 알려주세요. 어떤 학과이고, 어디 대학에 있으며, 졸업 후 진로와 연봉은 어떤가요?"
    """
    return f"{query}에 대해 자세히 알려주세요. 어떤 학과이고, 어디 대학에 있으며, 졸업 후 진로와 연봉은 어떤가요?"
