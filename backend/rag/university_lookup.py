"""
대학 입시 정보 조회 유틸리티

Database의 University 테이블을 조회하여 대학별 KCUE 입시 정보 URL을 제공합니다.
"""

from typing import Optional, Dict

from backend.db.connection import SessionLocal
from backend.db.models import University


def lookup_university_url(university_name: str) -> Optional[Dict[str, str]]:
    """
    대학명으로 KCUE 입시 정보 URL 조회

    Args:
        university_name: 대학명 (예: "서울대학교", "연세대학교")

    Returns:
        대학 정보 딕셔너리 또는 None
        {
            "university": "서울대학교[본교]",
            "code": "0000019",
            "url": "https://www.adiga.kr/..."
        }
    """
    if not university_name:
        return None

    session = SessionLocal()
    try:
        # 1. 정확한 매칭 시도
        # University.name은 유니크 인덱스가 있으므로 빠름
        uni = (
            session.query(University).filter(University.name == university_name).first()
        )
        if uni:
            return {"university": uni.name, "code": uni.code, "url": uni.url}

        normalized_name = university_name.strip()

        # 2. [본교] 추가 시도
        if not normalized_name.endswith("]"):
            with_campus = f"{normalized_name}[본교]"
            uni = (
                session.query(University).filter(University.name == with_campus).first()
            )
            if uni:
                return {"university": uni.name, "code": uni.code, "url": uni.url}

        # 3. 부분 매칭 (대학명이 포함된 경우) - LIKE 검색
        # "서울대학교" 입력 시 "서울대학교[본교]" 매칭
        # LIMIT 1로 하나만 가져옴
        uni = (
            session.query(University)
            .filter(University.name.like(f"%{normalized_name}%"))
            .first()
        )
        if uni:
            return {"university": uni.name, "code": uni.code, "url": uni.url}

        return None

    finally:
        session.close()


def search_universities(query: str) -> list[Dict[str, str]]:
    """
    대학명으로 검색하여 매칭되는 모든 대학 반환

    Args:
        query: 검색 쿼리 (예: "서울", "연세")

    Returns:
        매칭되는 대학 정보 리스트
    """
    if not query:
        return []

    results = []

    session = SessionLocal()
    try:
        # 대학명에 쿼리가 포함되어 있으면 추가
        search_query = f"%{query.strip()}%"
        universities = (
            session.query(University).filter(University.name.like(search_query)).all()
        )

        for uni in universities:
            results.append({"university": uni.name, "code": uni.code, "url": uni.url})

        return results
    finally:
        session.close()
