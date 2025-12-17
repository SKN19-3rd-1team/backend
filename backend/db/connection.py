from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.config import get_settings
import json
import logging
import os
import time

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)

# ---------------------------------------------------------
# DB 쿼리 로깅 설정
# ---------------------------------------------------------


# 로그 디렉토리 생성 (backend/db/logs)
Current_Dir = os.path.dirname(os.path.abspath(__file__))
Log_Dir = os.path.join(Current_Dir, "logs")
os.makedirs(Log_Dir, exist_ok=True)

# 로거 설정
logger = logging.getLogger("sqlalchemy_custom")
logger.setLevel(logging.INFO)

# 파일 핸들러 추가
log_file_path = os.path.join(Log_Dir, "query_log.log")
file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


@event.listens_for(engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info["query_start_time"] = time.time()
    logger.info(f"📝 QUERY: {statement}")
    if parameters:
        logger.info(f"🔧 PARAMS: {parameters}")


@event.listens_for(engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - conn.info.get("query_start_time", time.time())
    logger.info(f"⏱️ EXECUTION TIME: {total:.4f}s")

    # 결과 로깅 (로우 카운트 등)
    if hasattr(cursor, "rowcount"):
        logger.info(f"🔢 ROWS AFFECTED: {cursor.rowcount}")

    logger.info("-" * 50)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """데이터베이스 세션 의존성 (FastAPI용)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
