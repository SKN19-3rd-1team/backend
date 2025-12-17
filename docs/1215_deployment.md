# 배포 유의사항 및 절차 (2025-12-15)

본 문서는 Unigo 프로젝트 배포 시 반드시 확인해야 할 유의사항과 절차를 정리한 문서입니다. `[.agent/workflows/plan.md]` 규칙에 따라 작업 완료 후 작성되었습니다.

## 1. 주요 변경 사항 및 검토

- **Dockerization**: `Dockerfile`, `docker-compose.yml`, `nginx/nginx.conf` 추가.
- **Database**: 외부/호스트 MySQL 사용 설정 (Docker 내부 DB 컨테이너 미사용).
- **Nginx**: 80번 포트 리버스 프록시 및 정적 파일 서빙 구성.

## 2. 배포 전 체크리스트

- [ ] **`.env` 파일 보안**: 실서버 배포 시 `DEBUG=False` 설정 및 `SECRET_KEY` 변경 필수.
- [ ] **데이터베이스 암호**: 기본 `root` 패스워드 사용 금지. 복잡한 패스워드로 변경.
- [ ] **AWS 보안 그룹**: 3306(DB) 포트는 외부 개방 금지 (필요 시 VPN이나 특정 IP만 허용). 22(SSH)는 관리자 IP만 허용.
- [ ] **볼륨 백업**: `mysql_data` 볼륨은 컨테이너 삭제 시에도 유지되지만, 별도 백업 정책 수립 필요.

## 3. 배포 절차 요약

1. EC2 인스턴스 프로비저닝 (Ubuntu 22.04).
2. Docker & Docker Compose 설치.
3. Git Clone 및 `.env` 설정.
4. `docker compose up -d --build` 실행.
5. `docker compose exec web python manage.py migrate` 실행 (DB 초기화).
6. `docker compose exec web python manage.py collectstatic` 실행 (정적 파일 모으기).

## 4. 모니터링 및 유지보수

- **로그 확인**: `docker compose logs -f [service_name]`
- **재시작**: `docker compose restart [service_name]`
- **업데이트**: `git pull` -> `docker compose up -d --build`

## 5. 향후 개선 사항

- HTTPS (SSL) 적용 (Let's Encrypt + Certbot).
- CI/CD 파이프라인 구축 (GitHub Actions).
- AWS RDS 로의 데이터베이스 이전 (트래픽 증가 시).
