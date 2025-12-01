import json
import re

def load_majors():
    """
    major_detail.json 파일에서 전공 및 학과 정보를 추출하여
    major_categories.json 파일로 저장하는 스크립트입니다.
    """
    try:
        # 원본 데이터 파일 로드
        with open('/home/maroco/major_mentor/backend/data/major_detail.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading json: {e}")
        return

    categories = {}
    
    # 데이터 구조: 리스트 -> 객체 -> "dataSearch" -> "content" -> 레코드 리스트
    for block in data:
        content = block.get("dataSearch", {}).get("content", [])
        if not content:
            continue
            
        for record in content:
            major = record.get("major")
            dept_str = record.get("department")
            
            if not major:
                continue
                
            major = major.strip()
            
            # 학과 정보 추출 (쉼표로 구분된 문자열)
            depts = []
            if dept_str:
                parts = dept_str.split(',')
                for p in parts:
                    p = p.strip()
                    # 자기 자신과 다른 이름만 추가하거나, 키워드 확장을 위해 모두 포함
                    if p and p != major: 
                        depts.append(p)
            
            # 학과 정보가 없으면 전공명 자체를 사용
            if not depts:
                depts = [major]
            else:
                # 검색 편의를 위해 전공명 자체도 리스트 맨 앞에 추가
                if major not in depts:
                    depts.insert(0, major)
            
            # 전공명을 키로 사용하여 매핑 저장
            categories[major] = depts

    # 키 기준으로 정렬
    sorted_cats = dict(sorted(categories.items()))

    # JSON 파일로 저장
    output_path = '/home/maroco/major_mentor/backend/data/major_categories.json'
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_cats, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved categories to {output_path}")
    except Exception as e:
        print(f"Error saving json: {e}")

if __name__ == "__main__":
    load_majors()
