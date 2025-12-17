import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from backend.db.seed_majors import seed_majors
from backend.db.seed_categories import seed_categories
from backend.db.seed_universities import seed_universities
from backend.db.connection import engine, Base


def seed_all():
    print("🚀 Starting Full Database Seeding...")
    print("=" * 50)

    # 1. 테이블 생성 (모든 모델)
    print("🛠️  Ensuring tables exist...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables checked/created.")
    print("=" * 50)

    # 2. 전공 데이터 적재
    print("\n[Step 1/3] Seeding Majors...")
    seed_majors()

    # 3. 카테고리 데이터 적재
    print("\n[Step 2/3] Seeding Major Categories...")
    seed_categories()

    # 4. 대학 데이터 적재
    print("\n[Step 3/3] Seeding Universities...")
    seed_universities()

    print("\n" + "=" * 50)
    print("🎉 All seeding processes completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    seed_all()
