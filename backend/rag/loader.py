"""Utilities that turn JSON records into LangChain friendly documents."""

from __future__ import annotations

# backend/rag/loader.py
import json
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
                    course_classification = course.get("course_classification", "")
                    description = course.get("description", "")
                    name_en = course.get("name_en", "")

                    text = (
                        f"과목명: {name}\n"
                        f"영문명: {name_en}\n"
                        f"학년/학기: {grade_semester}\n"
                        f"분류: {course_classification}\n"
                        f"설명: {description}"
                    )

                    metadata = {
                        "university": university,
                        "college": college,
                        "department": department,
                        "name": name,
                        "name_en": name_en,
                        "grade_semester": grade_semester,
                        "course_classification": course_classification,
                    }

                    docs.append(Document(page_content=text, metadata=metadata))

    return docs