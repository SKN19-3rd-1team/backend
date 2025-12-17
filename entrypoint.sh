#!/bin/sh

# 에러 발생 시 즉시 종료
set -e

# DB가 준비될 때까지 대기하는 로직이 필요할 수 있음 (depends_on이 있지만 완벽하지 않을 수 있음)
# 하지만 docker-compose의 healthcheck가 있으므로 일단 생략 가능

echo "Applying database migrations..."
python manage.py migrate

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Loading initial major data..."
# 이미 데이터가 있어도 update_or_create를 사용하므로 안전함
# python manage.py load_major_data (Deleted legacy command)

echo "Starting Gunicorn..."
exec "$@"
