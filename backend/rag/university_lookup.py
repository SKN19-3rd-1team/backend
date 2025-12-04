"""
대학 입시 정보 조회 유틸리티

university_data_cleaned.json 파일을 로드하여 대학별 KCUE 입시 정보 URL을 제공합니다.
"""

import json
from pathlib import Path
from typing import Optional, Dict

# 전역 캐시 변수
_UNIVERSITY_DATA_CACHE: Optional[Dict[str, Dict[str, str]]] = None


def _load_university_data() -> Dict[str, Dict[str, str]]:
    """
    university_data_cleaned.json 파일을 로드하여 캐싱
    
    Returns:
        대학명을 키로 하는 딕셔너리
        {
            "서울대학교[본교]": {
                "code": "0000019",
                "url": "https://www.adiga.kr/..."
            },
            ...
        }
    """
    global _UNIVERSITY_DATA_CACHE
    
    # 이미 캐시되어 있으면 반환
    if _UNIVERSITY_DATA_CACHE is not None:
        return _UNIVERSITY_DATA_CACHE
    
    try:
        # 현재 파일(university_lookup.py)의 위치: backend/rag/
        # 데이터 파일 위치: backend/data/university_data_cleaned.json
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent  # backend/
        json_path = project_root / "data" / "university_data_cleaned.json"
        
        if json_path.exists():
            _UNIVERSITY_DATA_CACHE = json.loads(json_path.read_text(encoding="utf-8"))
            print(f"✅ Loaded {len(_UNIVERSITY_DATA_CACHE)} universities from {json_path.name}")
            return _UNIVERSITY_DATA_CACHE
        
        print(f"⚠️ University data file not found at: {json_path}")
        _UNIVERSITY_DATA_CACHE = {}
        return _UNIVERSITY_DATA_CACHE
        
    except Exception as e:
        print(f"⚠️ Failed to load university data: {e}")
        _UNIVERSITY_DATA_CACHE = {}
        return _UNIVERSITY_DATA_CACHE


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
    data = _load_university_data()
    
    if not university_name:
        return None
    
    # 정확한 매칭 시도
    if university_name in data:
        return {
            "university": university_name,
            **data[university_name]
        }
    
    # [본교] 등의 캠퍼스 정보가 없는 경우 자동으로 추가하여 검색
    normalized_name = university_name.strip()
    
    # 1. [본교] 추가 시도
    if not normalized_name.endswith("]"):
        with_campus = f"{normalized_name}[본교]"
        if with_campus in data:
            return {
                "university": with_campus,
                **data[with_campus]
            }
    
    # 2. 부분 매칭 (대학명이 포함된 경우)
    for key in data.keys():
        # "서울대학교" 입력 시 "서울대학교[본교]" 매칭
        if normalized_name in key or key.startswith(normalized_name):
            return {
                "university": key,
                **data[key]
            }
    
    return None


def search_universities(query: str) -> list[Dict[str, str]]:
    """
    대학명으로 검색하여 매칭되는 모든 대학 반환
    
    Args:
        query: 검색 쿼리 (예: "서울", "연세")
        
    Returns:
        매칭되는 대학 정보 리스트
    """
    data = _load_university_data()
    
    if not query:
        return []
    
    normalized_query = query.strip().lower()
    results = []
    
    for key, value in data.items():
        # 대학명에 쿼리가 포함되어 있으면 추가
        if normalized_query in key.lower():
            results.append({
                "university": key,
                **value
            })
    
    return results
