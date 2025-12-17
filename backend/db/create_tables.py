import sys
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from backend.db.connection import engine, Base

# Import models to ensure they are registered with Base.metadata
from backend.db import models


def create_tables():
    print("Creating SQLAlchemy tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully.")


if __name__ == "__main__":
    create_tables()
