"""
대학 과목 데이터 로딩 모듈

JSON 형식의 원본 데이터를 읽어서 LangChain Document 객체로 변환합니다.

** 데이터 구조 **
JSON 파일 구조: 대학 → 단과대학 → 학과 → 과목 배열
```
{
  "서울대학교": {
    "공과대학": {
      "컴퓨터공학부": [
        {
          "name": "자료구조",
          "description": "...",
          "grade_semester": "2학년 1학기",
          ...
        }
      ]
    }
  }
}
```

각 과목은 LangChain Document로 변환되어 벡터 DB에 저장됩니다.
"""

from __future__ import annotations

# backend/rag/loader.py
import json
import re
from pathlib import Path
from langchain_core.documents import Document


def load_courses(json_path: str | Path) -> list[Document]:
    """
    JSON 파일에서 대학 과목 데이터를 읽어 LangChain Document 리스트로 변환

    ** 동작 과정 **
    1. JSON 파일 읽기 (UTF-8 인코딩)
    2. 중첩된 구조 순회: 대학 → 단과대학 → 학과 → 과목
    3. 각 과목을 Document 객체로 변환
       - page_content: LLM이 읽을 텍스트 (과목명, 설명 등)
       - metadata: 검색 필터링에 사용될 구조화된 정보

    Args:
        json_path: 과목 데이터가 담긴 JSON 파일 경로

    Returns:
        list[Document]: 각 과목이 하나의 Document인 리스트

    ** Document 구조 **
    - page_content: "과목명: ...\n설명: ..." (임베딩 대상 텍스트)
    - metadata: {"university": "서울대", "department": "컴공", ...} (필터링용)
    """
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    # 변환된 Document 객체들을 저장할 리스트
    docs: list[Document] = []

    # 중첩된 4단계 구조 순회: 대학 → 단과대학 → 학과 → 과목 배열
    for university, colleges in data.items():
        for college, departments in colleges.items():
            for department, courses in departments.items():
                # 각 학과의 과목 배열 순회
                for course in courses:
                    # 1. JSON에서 과목 정보 추출
                    name = course.get("name", "")
                    grade_semester = course.get("grade_semester", "")
                    # course_classification과 category 둘 다 지원 (데이터 소스에 따라 다를 수 있음)
                    course_classification = course.get("course_classification") or course.get("category", "")
                    description = course.get("description", "")
                    name_en = course.get("name_en", "")

                    # 2. 빈 필드 처리: LLM에게 명확히 "정보 없음"을 알림
                    # 빈 문자열보다 명시적인 메시지가 LLM의 이해에 도움됨
                    grade_semester_display = grade_semester.strip() if grade_semester else "[정보 없음]"
                    description_display = description.strip() if description else "[설명 정보가 제공되지 않았습니다]"
                    name_en_display = name_en.strip() if name_en else "[정보 없음]"
                    course_classification_display = course_classification.strip() if course_classification else "[정보 없음]"

                    # 3. 학년/학기 정보 파싱
                    # 검색 필터링을 위해 "2학년 1학기"를 grade="2학년", semester="1학기"로 분리
                    grade = ""
                    semester = ""

                    if grade_semester:
                        # 한글 형식: "2학년 1학기", "3학년 2학기" 등
                        match_korean = re.search(r'([1-4])학년\s*([1-2])학기', grade_semester)
                        # 숫자-하이픈 형식: "2-1", "3-2" 등
                        match_dash = re.search(r'([0-4])\-([1-2])', grade_semester)

                        if match_korean:
                            grade = f"{match_korean.group(1)}학년"
                            semester = f"{match_korean.group(2)}학기"
                        elif match_dash:
                            # 0학년은 공통/교양 과목을 의미할 수 있음
                            g = match_dash.group(1)
                            s = match_dash.group(2)
                            grade = f"{g}학년" if g != "0" else ""
                            semester = f"{s}학기"

                    # 4. page_content 생성: LLM이 읽을 텍스트
                    # 임베딩 모델이 이 텍스트를 벡터로 변환함
                    # 중요한 정보(과목명)를 앞에 배치하여 임베딩 품질 향상
                    text = (
                        f"과목명: {name}\n"
                        f"영문명: {name_en_display}\n"
                        f"학년/학기: {grade_semester_display}\n"
                        f"분류: {course_classification_display}\n"
                        f"설명: {description_display}"
                    )

                    # 5. metadata 생성: 검색 필터링에 사용될 구조화된 정보
                    # Chroma DB는 metadata 기반 필터링을 지원함
                    # 예: "서울대" 과목만 검색, "2학년" 과목만 검색 등
                    metadata = {
                        "university": university,      # 대학명 (예: "서울대학교")
                        "college": college,            # 단과대학 (예: "공과대학")
                        "department": department,      # 학과 (예: "컴퓨터공학부")
                        "name": name,                  # 과목명 (예: "자료구조")
                        "name_en": name_en,            # 영문 과목명
                        "grade_semester": grade_semester,  # 원본 학년/학기 (예: "2학년 1학기")
                        "grade": grade,                # 파싱된 학년 (예: "2학년")
                        "semester": semester,          # 파싱된 학기 (예: "1학기")
                        "course_classification": course_classification,  # 과목 분류 (예: "전공필수")
                    }

                    # 6. Document 생성 및 리스트에 추가
                    # LangChain의 Document는 page_content(텍스트)와 metadata(메타정보)로 구성됨
                    docs.append(Document(page_content=text, metadata=metadata))

    return docs