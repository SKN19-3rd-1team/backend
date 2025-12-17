import json
import re
import uuid
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 백엔드 모듈 임포트를 위해 프로젝트 루트 경로 추가
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from backend.db.connection import SessionLocal, engine, Base
from backend.db.models import Major


def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    """소스 JSON 데이터를 로드하고 파싱합니다."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any, default: float = 0.0) -> float:
    """값을 안전하게 float 타입으로 변환합니다."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def extract_chart_stats(chart_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    chart_data에서 통계 정보를 추출합니다.

    Returns:
        "employment_rate"와 "acceptance_rate"를 포함하는 딕셔너리 (값은 없을 수 있음)
    """
    stats = {"employment_rate": None, "acceptance_rate": None}

    if not chart_data:
        return stats

    data_entry = chart_data[0]  # 일반적으로 리스트로 감싸져 있음

    # 1. 취업률 (Employment Rate)
    # chartData.employment_rate -> item "전체"
    emp_rates = data_entry.get("employment_rate", [])
    if emp_rates:
        for item in emp_rates:
            if item.get("item") == "전체":
                stats["employment_rate"] = safe_float(item.get("data"), None)
                break

    # 2. 입학률 (Acceptance Rate, 경쟁률의 역수 개념)
    # chartData.applicant -> "지원자" (Applicants) vs "입학자" (Entrants)
    # 입학률 = (입학자 수 / 지원자 수) * 100
    applicants = data_entry.get("applicant", [])
    num_applicants = 0.0
    num_entrants = 0.0

    for item in applicants:
        label = item.get("item", "")
        val = safe_float(item.get("data"), 0.0)
        if label == "지원자":
            num_applicants = val
        elif label == "입학자":
            num_entrants = val

    if num_applicants > 0:
        stats["acceptance_rate"] = (num_entrants / num_applicants) * 100

    return stats


def preprocess_item(raw_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    단일 원본 JSON 항목을 Major 모델과 호환되는 딕셔너리로 변환합니다.
    """
    # 1. 중첩 구조 해제
    # 구조: { "dataSearch": { "content": [ { ... } ] } }
    ds = raw_item.get("dataSearch")
    if not ds:
        return None
    content = ds.get("content")
    if not content:
        return None

    data = content[0]
    major_name = data.get("major", "").strip()

    if not major_name:
        return None

    # 2. 결정론적 ID 생성 (Deterministic ID)
    # 스크립트 실행 시마다 동일한 전공명에 대해 항상 같은 ID가 생성되도록
    # DNS 네임스페이스와 major_name을 사용하여 UUID5를 생성합니다.
    major_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, major_name))

    # 리스트 직렬화를 위한 헬퍼 함수
    def to_json(obj):
        return json.dumps(obj, ensure_ascii=False) if obj else None

    # 별칭(aliases) 리스트 직렬화
    dept_str = data.get("department", "")
    dept_list = [d.strip() for d in dept_str.split(",") if d.strip()]

    # 차트 데이터 통계 추출
    raw_chart = data.get("chartData", [])
    stats = extract_chart_stats(raw_chart)

    return {
        "major_id": major_id,
        "major_name": major_name,
        # 텍스트/HTML 필드
        "summary": data.get("summary"),
        "interest": data.get("interest"),
        "property": data.get("property"),
        "job": data.get("job"),
        # HTML 태그 제거 (예: "<strong>60</strong>%" -> "60%")
        "employment": re.sub(r"<[^>]+>", "", data.get("employment") or ""),
        # 숫자 필드
        "salary": safe_float(data.get("salary")),
        "employment_rate": stats["employment_rate"],
        "acceptance_rate": stats["acceptance_rate"],
        # JSON 필드
        "relate_subject": to_json(data.get("relate_subject")),
        "enter_field": to_json(data.get("enter_field")),
        "department_aliases": to_json(dept_list),
        "career_act": to_json(data.get("career_act")),
        # qualifications: DB 모델에는 LONGTEXT로 정의되어 있으나, JSON 필드 그룹에 포함되어 있음.
        # 원본 데이터는 쉼표로 구분된 문자열임.
        # 검토 결과 그대로 문자열로 저장하기로 함.
        "qualifications": data.get("qualifications"),
        "main_subject": to_json(data.get("main_subject")),
        "university": to_json(data.get("university")),
        "chart_data": to_json(raw_chart),
        "raw_data": json.dumps(
            raw_item, ensure_ascii=False
        ),  # 전체 항목을 원본 백업으로 저장
    }


def seed_majors():
    json_path = project_root / "backend" / "data" / "major_detail.json"
    print(f"Loading data from {json_path}...")

    raw_list = load_json_data(json_path)
    print(f"Found {len(raw_list)} items in JSON.")

    session = SessionLocal()

    succeeded = 0
    skipped = 0
    errors = 0

    try:
        # 테이블이 없으면 생성 (일반적으로 alembic으로 처리하지만 안전을 위해 확인)
        # Base.metadata.create_all(bind=engine)

        for i, item in enumerate(raw_list):
            try:
                processed = preprocess_item(item)
                if not processed:
                    skipped += 1
                    continue

                # 업데이트 또는 삽입 (Upsert)
                # 존재 여부 확인
                existing = (
                    session.query(Major)
                    .filter_by(major_id=processed["major_id"])
                    .first()
                )

                if existing:
                    # 필드 업데이트
                    for k, v in processed.items():
                        setattr(existing, k, v)
                else:
                    # 새로 삽입
                    new_major = Major(**processed)
                    session.add(new_major)

                succeeded += 1

                if i % 100 == 0:
                    print(f"Processing... {i}/{len(raw_list)}")

            except Exception as e:
                print(f"Error processing item index {i}: {e}")
                errors += 1

        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Critical error during seeding: {e}")
        raise
    finally:
        session.close()

    print("=" * 50)
    print(f"Seeding Complete.")
    print(f"Total processed: {len(raw_list)}")
    print(f"Succeeded (Upsert): {succeeded}")
    print(f"Skipped (Invalid Data): {skipped}")
    print(f"Errors: {errors}")
    print("=" * 50)


if __name__ == "__main__":
    seed_majors()
