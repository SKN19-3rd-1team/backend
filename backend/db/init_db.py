import sys
import os

# 현재 디렉토리를 path에 추가하여 backend 모듈을 찾을 수 있게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from backend.db.connection import engine
from backend.db.models import Base


def init_db():
    print("Creating tables for SQLAlchemy models...")
    try:
        # 모든 SQLAlchemy 모델의 테이블 생성
        Base.metadata.create_all(bind=engine)
        print("✅ Tables created successfully!")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")


if __name__ == "__main__":
    init_db()
