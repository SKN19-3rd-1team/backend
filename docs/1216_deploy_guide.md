# AWS EC2 배포 가이드

이 문서는 Unigo 프로젝트를 AWS EC2 인스턴스에 Docker를 사용하여 배포하는 절차를 설명합니다.

## 1. AWS EC2 인스턴스 생성

1. **AWS Console 로그인**: EC2 대시보드로 이동합니다.
2. **인스턴스 시작**:
   - **OS**: Ubuntu Server 22.04 LTS (x86_64) 추천.
   - **인스턴스 유형**: t2.micro (프리티어) 또는 필요에 따라 t3.small 이상.
   - **키 페어**: 새 키 페어를 생성하고 `.pem` 파일을 안전한 곳에 저장합니다.
   - **네트워크 설정**:
     - 보안 그룹 생성:
       - SSH (22): 내 IP에서만 허용 (보안 권장)
       - HTTP (80): 위치 무관 (0.0.0.0/0)
       - HTTPS (443): 위치 무관 (0.0.0.0/0) (SSL 적용 시 필요)
3. **인스턴스 시작**: 설정을 확인하고 인스턴스를 시작합니다.

## 2. 서버 접속 및 초기 설정

터미널에서 다운로드 받은 키 페어 파일이 있는 경로로 이동하여 SSH 접속을 시도합니다.

```bash
# 키 파일 권한 변경
chmod 400 your-key-pair.pem

# 접속 (instance-public-ip는 AWS 콘솔에서 확인)
ssh -i "your-key-pair.pem" ubuntu@<instance-public-ip>
```

접속 후 시스템 패키지를 업데이트합니다.

```bash
sudo apt update && sudo apt upgrade -y
```

## 3. Docker 및 Docker Compose 설치

Docker 엔진과 Docker Compose 플러그인을 설치합니다.

```bash
# 필수 패키지 설치
sudo apt install -y ca-certificates curl gnupg lsb-release

# Docker GPG 키 추가
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Docker 리포지토리 설정
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker 설치
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 권한 설정 (sudo 없이 docker 실행)
sudo usermod -aG docker $USER
newgrp docker
```

## 4. 프로젝트 배포

### 4.1. 코드 복제 (Git Clone)

GitHub 등 리포지토리에서 코드를 가져옵니다.

```bash
git clone <repository_url>
cd <repository_directory>
```

### 4.2. 환경 변수 설정

`.env` 파일을 생성하고 필요한 값을 채워 넣습니다.

```bash
cp .env.example .env
nano .env
```

**주의**: `MYSQL_PASSWORD` 및 `SECRET_KEY` 등 민감한 정보는 강력한 값으로 변경하세요.

### 4.3. Docker 서비스 실행

```bash
docker compose up --build -d
```

- `-d`: 백그라운드에서 실행

### 4.4. 로그 확인 및 상태 점검

```bash
docker compose logs -f
docker compose ps
```

## 5. 배포 후 작업

### 5.1. 마이그레이션 및 초기 데이터

데이터베이스 테이블 생성을 위해 마이그레이션을 실행해야 할 수 있습니다.

```bash
docker compose exec web python manage.py migrate
docker compose exec web python manage.py collectstatic --noinput
```

(참고: `entrypoint.sh` 스크립트를 만들어 컨테이너 시작 시 자동 수행하게 할 수도 있습니다.)

## 6. 트러블슈팅

- **포트 충돌**: 이미 80번 포트를 사용하는 프로세스가 있는지 확인하세요.
- **DB 연결 오류**:
  - 내부 DB 사용 시: `docker compose logs db` 확인.
  - 외부/호스트 DB 사용 시: `my.ini`의 `bind-address = 0.0.0.0` 설정 및 방화벽(3306 포트) 허용 여부 확인.
- **권한 문제**: `static` 폴더 등의 쓰기 권한을 확인하세요.
