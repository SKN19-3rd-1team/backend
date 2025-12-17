import json
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from backend.db.connection import SessionLocal, engine, Base
from backend.db.models import University


def seed_universities():
    json_path = project_root / "backend" / "data" / "university_data_cleaned.json"
    print(f"Loading universities from {json_path}...")

    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    session = SessionLocal()
    try:
        # 테이블 생성
        # Base.metadata.create_all(bind=engine)

        count = 0
        for uni_name, info in data.items():
            # Info example: {"code": "0000063", "url": "..."}
            code = info.get("code")
            url = info.get("url")

            # Upsert
            existing = session.query(University).filter_by(name=uni_name).first()

            if existing:
                existing.code = code
                existing.url = url
            else:
                new_uni = University(name=uni_name, code=code, url=url)
                session.add(new_uni)
            count += 1

        session.commit()
        print(f"✅ Successfully seeded {count} universities.")

    except Exception as e:
        session.rollback()
        print(f"❌ Error seeding universities: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    seed_universities()
