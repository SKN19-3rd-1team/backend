import json
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from backend.db.connection import SessionLocal, engine, Base
from backend.db.models import MajorCategory


def seed_categories():
    json_path = project_root / "backend" / "data" / "major_categories.json"
    print(f"Loading categories from {json_path}...")

    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    session = SessionLocal()
    try:
        # 테이블 생성 (안전을 위해)
        # Base.metadata.create_all(bind=engine)

        count = 0
        for category, majors in data.items():
            # Upsert
            existing = (
                session.query(MajorCategory).filter_by(category_name=category).first()
            )
            major_json = json.dumps(majors, ensure_ascii=False)

            if existing:
                existing.major_names = major_json
            else:
                new_cat = MajorCategory(category_name=category, major_names=major_json)
                session.add(new_cat)
            count += 1

        session.commit()
        print(f"✅ Successfully seeded {count} categories.")

    except Exception as e:
        session.rollback()
        print(f"❌ Error seeding categories: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    seed_categories()
