# Python 3.10.12 슬림 이미지 사용 (Ubuntu 22.04 기반)
FROM python:3.10.12-slim

# 환경 변수 설정
# PYTHONDONTWRITEBYTECODE: 파이썬 .pyc 파일 생성 방지
# PYTHONUNBUFFERED: 버퍼링 없이 즉시 로그 출력
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (mysqlclient, build 도구 등)
# rm -rf /var/lib/apt/lists/* : 이미지 크기를 줄이기 위해 캐시 삭제
RUN apt-get update && apt-get install -y \
    pkg-config \
    default-libmysqlclient-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 패키지 목록 복사 및 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . /app/

# 작업 디렉토리 변경 (Django 프로젝트 루트)
WORKDIR /app/unigo

# Entrypoint 스크립트 복사 및 실행 권한 설정 (빌드 컨텍스트 루트에서 복사)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Entrypoint 설정
ENTRYPOINT ["/app/entrypoint.sh"]

# Gunicorn 실행 (entrypoint의 "$@"로 전달됨)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "unigo.wsgi:application"]
