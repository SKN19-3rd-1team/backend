"""Utilities that turn JSON records into LangChain friendly documents."""

from __future__ import annotations

# backend/rag/loader.py
import json
import re
from pathlib import Path
from langchain_core.documents import Document

# json 파일을 읽어서 데이터 로드
def load_courses(json_path: str | Path) -> list[Document]:
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    # 내용을 저장할 변수
    docs: list[Document] = []

    # 중첩된 구조 탐색: 대학 -> 단과대학 -> 학부/학과 -> 과목 배열
    for university, colleges in data.items():
        for college, departments in colleges.items():
            for department, courses in departments.items():
                for course in courses:
                    name = course.get("name", "")
                    grade_semester = course.get("grade_semester", "")
                    course_classification = course.get("course_classification") or course.get("category", "")
                    description = course.get("description", "")
                    name_en = course.get("name_en", "")

                    # Handle missing or empty fields with clear messaging
                    grade_semester_display = grade_semester.strip() if grade_semester else "[정보 없음]"
                    description_display = description.strip() if description else "[설명 정보가 제공되지 않았습니다]"
                    name_en_display = name_en.strip() if name_en else "[정보 없음]"
                    course_classification_display = course_classification.strip() if course_classification else "[정보 없음]"

                    # Extract grade (e.g., "2학년") from grade_semester (e.g., "2학년 1학기")
                    grade = ""
                    semester = ""

                    if grade_semester:
                        # "2학년 1학기"도 지원하고,
                        match_korean = re.search(r'([1-4])학년\s*([1-2])학기', grade_semester)
                        match_dash = re.search(r'([0-4])\-([1-2])', grade_semester)

                        if match_korean:
                            grade = f"{match_korean.group(1)}학년"
                            semester = f"{match_korean.group(2)}학기"
                        elif match_dash:
                            # 0은 공통/자유학점 같은 의미라면 별도 처리
                            g = match_dash.group(1)
                            s = match_dash.group(2)
                            grade = f"{g}학년" if g != "0" else ""
                            semester = f"{s}학기"

                    text = (
                        f"과목명: {name}\n"
                        f"영문명: {name_en_display}\n"
                        f"학년/학기: {grade_semester_display}\n"
                        f"분류: {course_classification_display}\n"
                        f"설명: {description_display}"
                    )

                    metadata = {
                        "university": university,
                        "college": college,
                        "department": department,
                        "name": name,
                        "name_en": name_en,
                        "grade_semester": grade_semester,
                        "grade": grade,
                        "semester": semester,
                        "course_classification": course_classification,
                    }

                    docs.append(Document(page_content=text, metadata=metadata))

    return docs