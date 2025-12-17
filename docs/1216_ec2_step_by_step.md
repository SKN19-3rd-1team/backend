# AWS EC2 배포 가이드 (Step-by-Step)

로컬에서 테스트를 마쳤으므로, AWS EC2(Ubuntu) 서버에 실배포하는 과정을 상세히 안내합니다.

## 0. AWS 인스턴스 생성 및 세팅 (AWS Console)

가장 먼저 AWS 콘솔에서 서버(컴퓨터)를 빌리고 고정 IP를 할당해야 합니다.

### 0.1. EC2 인스턴스 생성

1. **AWS Console 로그인** 후 **EC2** 서비스로 이동합니다.
2. **"인스턴스 시작 (Launch Instances)"** 버튼 클릭.
3. **이름 및 태그**: `Unigo-Web-Server` 등 식별 가능한 이름 입력.
4. **이미지(OS)**: **Ubuntu Server 22.04 LTS (HVM)** 선택 (프리티어 사용 가능).
5. **인스턴스 유형**: `t2.micro` (프리티어) 또는 필요에 따라 상위 스펙 선택.
6. **키 페어(로그인)**:
   - "새 키 페어 생성" 클릭.
   - 이름 입력 (예: `unigo-key`).
   - 유형: `RSA`, 파일 형식: `.pem` (OpenSSH용).
   - **다운로드된 .pem 파일을 안전한 곳에 보관하세요.** (분실 시 접속 불가)
7. **네트워크 설정 (보안 그룹)**:
   - "보안 그룹 생성" 선택.
   - 다음 포트 허용 (**규칙 추가**):
     - `SSH` (TCP 22): `내 IP (My IP)` 또는 `위치 무관 (Anywhere)` (보안상 내 IP 권장).
     - `HTTP` (TCP 80): `위치 무관 (Anywhere) 0.0.0.0/0`.
     - `HTTPS` (TCP 443): `위치 무관` (추후 도메인 연동 시 필요).
     - `사용자 지정 TCP` (8000): `위치 무관` (테스트용, Nginx 설정 전 Django 직접 접속 시 필요).
8. **스토리지**: 기본 8GB 이상 (넉넉하게 20GB 권장, 프리티어는 30GB까지 무료).
9. **"인스턴스 시작"** 클릭.

### 0.2. 탄력적 IP (Elastic IP) 할당

서버를 껐다 켜도 IP가 바뀌지 않도록 고정 IP를 연결합니다.

1. EC2 대시보드 왼쪽 메뉴에서 **"네트워크 및 보안" -> "탄력적 IP"** 클릭.
2. **"탄력적 IP 주소 할당"** 클릭 -> "할당" 완료.
3. 할당된 IP 선택 후 **"작업" -> "탄력적 IP 주소 연결"** 클릭.
4. **인스턴스**: 방금 만든 `Unigo-Web-Server` 선택.
5. **"연결"** 클릭.
6. 이제 이 **Public IP**가 서버의 고정 주소가 됩니다.

---

## 1. 전제 조건 (Prerequisites)

- AWS EC2 인스턴스가 생성되어 있어야 합니다 (OS: Ubuntu 22.04 LTS 권장).
- 보안 그룹(Security Group)에서 다음 포트가 열려 있어야 합니다.
  - **SSH (22)**: 관리자 접속용
  - **HTTP (80)**: 웹 서비스용

## 2. 서버 접속 및 기본 세팅

터미널(또는 PuTTY)을 열고 EC2에 SSH로 접속합니다.

```bash
# 예시
ssh -i "key.pem" ubuntu@<EC2_PUBLIC_IP>
```

접속 후, 패키지들을 최신화하고 Docker를 설치합니다.

```bash
# 1. 패키지 업데이트
sudo apt update && sudo apt upgrade -y

# 2. Docker 자동 설치 스크립트 실행
curl -fsSL https://get.docker.com | sudo sh

# 3. sudo 없이 docker 명령어 쓰도록 권한 부여 (중요)
sudo usermod -aG docker $USER

# 4. 그룹 변경 사항 적용 (로그아웃 없이 적용)
newgrp docker
```

## 3. 프로젝트 코드 가져오기

GitHub 등에서 코드를 클론합니다.

```bash
# 예시: git clone
git clone <YOUR_REPOSITORY_URL>
cd Unigo
```

## 4. 배포용 설정 (`.env`, `docker-compose.prod.yml`)

### 4.1. `.env` 파일 생성

서버에는 `.env` 파일이 없으므로 새로 만들어야 합니다.

```bash
cp .env.example .env
nano .env
```

**[서버용 .env 주의사항]**

- `DEBUG=False` 로 변경 (보안 필수)
- `MYSQL_HOST` 값은 `docker-compose.prod.yml`이 알아서 덮어쓰므로 상관없지만, 헷갈리지 않게 그냥 두거나 `db`로 적으셔도 됩니다.
- **`SECRET_KEY`**: 배포용 복잡한 키로 변경 추천.

**[보안 팁: 파일 권한 설정]**
`.env` 파일은 중요한 정보를 담고 있으므로, 나(소유자) 외에는 읽을 수 없도록 권한을 제한하는 것이 안전합니다.

```bash
chmod 600 .env
```

### 4.2. 실행 (Production 모드)

우리는 로컬과 달리 **서버에서는 DB 컨테이너가 필요**하므로, 제가 만들어드린 `docker-compose.prod.yml`을 사용해야 합니다.

```bash
# -f 옵션으로 prod 파일 지정하여 실행
docker compose -f docker-compose.prod.yml up -d --build
```

## 5. 초기 데이터 및 설정

컨테이너가 실행되었다면, 데이터베이스 초기화가 필요합니다.

```bash
# 1. DB 마이그레이션 (테이블 생성 - Django)
docker compose -f docker-compose.prod.yml exec web python manage.py migrate

# 2. [필수] AI용 데이터베이스 초기화 및 전체 데이터 적재 (All-in-One)
# - SQLAlchemy 테이블 생성
# - 전공, 카테고리, 대학 데이터 순차 적재
# - (주의) WORKDIR이 /app/unigo로 되어 있으므로 /app으로 이동 후 실행해야 함
docker compose -f docker-compose.prod.yml exec web sh -c "cd /app && python backend/db/seed_all.py"

# 4. 정적 파일 모으기 (CSS/JS 등)
docker compose -f docker-compose.prod.yml exec web python manage.py collectstatic --noinput

# 5. 관리자 계정 생성 (Admin 페이지 접속용)
docker compose -f docker-compose.prod.yml exec web python manage.py createsuperuser
```

## 6. 접속 확인

브라우저 주소창에 `http://<EC2_PUBLIC_IP>` 를 입력하여 접속되는지 확인합니다.

---

## 💡 로컬 vs 서버 차이점 요약 (중요)

| 구분          | 로컬 (Local/Laptop)                               | 서버 (EC2 Production)                              |
| :------------ | :------------------------------------------------ | :------------------------------------------------- |
| **설정 파일** | `docker-compose.yml`                              | `docker-compose.prod.yml`                          |
| **DB 위치**   | **Windows Host** (172... or host.docker.internal) | **Docker 내부 컨테이너** (`db` 서비스)             |
| **명령어**    | `docker compose up ...`                           | `docker compose -f docker-compose.prod.yml up ...` |

## 7. 서버 관리 (재시작, 중지, 초기화)

서버 운영 중 자주 사용하는 명령어 모음입니다.

```bash
# 1. [재시작] 단순 재부팅 (가장 빠름, 설정 변경 없을 때)
docker compose -f docker-compose.prod.yml restart web

# 2. [재배포] 코드 변경나 .env 수정 적용 (컨테이너 재생성)
docker compose -f docker-compose.prod.yml up -d --build

# 3. [중지] 서버 내리기 (데이터는 유지됨)
docker compose -f docker-compose.prod.yml down

# 4. [완전 초기화] 데이터까지 싹 지우고 초기화 (주의!)
# -v 옵션 때문에 DB 데이터가 모두 삭제됩니다.
docker compose -f docker-compose.prod.yml down --rmi all -v --remove-orphans
```

## 8. 💰 요금 폭탄 방지 및 보안 주의사항 (필독)

AWS 프리티어를 사용하더라도 주의하지 않으면 요금이 청구될 수 있습니다. 다음 사항을 반드시 확인하세요.

### 8.1. 비용 관련 주의사항 (Cost Management)

1. **탄력적 IP (Elastic IP) 요금**:

   - 인스턴스가 **실행 중(Running)**일 때 연결된 IP 1개는 무료입니다.
   - 하지만 **인스턴스를 중지(Stop)**했는데 IP를 반납(Release)하지 않고 가지고만 있으면 **시간당 요금이 부과**됩니다. ($0.005/시간 ≈ 월 4,000원)
   - **해결책**: 서버를 아예 안 쓸 거면 IP도 "연결 해제(Disassociate)" 후 "릴리스(Release)" 하세요.

2. **스토리지 (EBS) 용량**:

   - 프리티어는 매월 30GB까지 무료입니다.
   - 인스턴스를 "종료(Terminate)"해도 EBS 볼륨이 삭제되지 않도록 설정된 경우, 계속 요금이 나갑니다.
   - **확인**: EC2 대시보드 -> **Elastic Block Store** -> **볼륨** 메뉴에서 안 쓰는 볼륨이 있는지 주기적으로 확인하세요.

3. **데이터 전송 (Data Transfer)**:
   - 인터넷으로 나가는 데이터(Outbound)는 월 100GB까지 무료입니다.
   - 이미지/동영상을 너무 많이 서빙하면 초과될 수 있습니다. (Nginx Gzip 설정을 적용해두어 일부 절감됩니다.)

### 8.2. 보안 및 운영 주의사항 (Security)

1. **ALLOWED_HOSTS 설정 (중요)**:

   - `.env` 파일의 `ALLOWED_HOSTS`에 반드시 **EC2의 Public IP**나 **도메인**을 적어야 합니다.
   - 예: `ALLOWED_HOSTS=3.12.34.56,mydomain.com`
   - 이걸 안 하면 접속 시 `Bad Request (400)` 에러가 뜹니다.
   - **[주의]** `.env` 수정 후에는 `restart`가 아니라 `up -d`를 해야 반영됩니다!

     ```bash
     docker compose -f docker-compose.prod.yml up -d web
     ```

2. **DEBUG 모드 끄기**:

   - `.env`에서 `DJANGO_DEBUG=False`로 되어 있는지 꼭 확인하세요. 켜져 있으면 에러 발생 시 서버 내부 코드가 다 노출됩니다.

3. **SSH 포트 제한**:
   - 보안 그룹에서 22번 포트(SSH)는 가급적 **"내 IP (My IP)"**로만 열어두는 것이 안전합니다. 해커들의 무차별 대입 공격(Brute Force) 대상 1순위입니다.

## 9. [TIP] 서버 멈춤 현상 해결 (Swap 메모리)

프리티어(`t2.micro`)는 램이 1GB밖에 없어서 Docker를 돌리다 보면 **서버가 멈추거나 접속이 안 될 수 있습니다.**
이럴 때 **Swap 메모리(가상 메모리)**를 설정하면 디스크의 일부를 램처럼 사용하여 멈춤 현상을 해결할 수 있습니다.

**설정 방법 (SSH 터미널에서 실행):**

```bash
# 1. 2GB 용량의 스왑 파일 생성
sudo fallocate -l 2G /swapfile

# 2. 권한 설정 (루트만 접근 가능)
sudo chmod 600 /swapfile

# 3. 스왑 파일로 포맷
sudo mkswap /swapfile

# 4. 스왑 활성화
sudo swapon /swapfile

# 5. 재부팅 후에도 유지되도록 설정 파일 수정
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 6. 확인 (Swap 항목에 2.0G 가 잡히면 성공)
free -h
```

이 설정을 하면 느려질지언정 서버가 뻗는 일은 확실히 줄어듭니다.

## 10. [Troubleshooting] "No space left on device" 오류 해결

빌드 도중이나 패키지 설치 중 **공간 부족(No space)** 오류가 뜬다면, 8GB 하드디스크가 꽉 찬 것입니다.
(특히 여러 번 빌드를 시도하면 쓰지 않는 이미지들이 쌓여서 금방 찹니다.)

**해결 1: 불필요한 Docker 데이터 정리 (가장 효과적)**

```bash
# 중지된 컨테이너, 사용 안 하는 네트워크/이미지/캐시 모두 삭제
docker system prune -a --volumes -f
```

**해결 2: 용량 확인**

```bash
df -h
```

`/dev/root/` 의 사용량(Use%)이 90% 이상이면 정리가 필요합니다.

**해결 3: EBS 볼륨 확장 (AWS Console)**
정리를 해도 부족하다면 AWS 콘솔에서 EBS 볼륨을 8GB -> 20GB 정도로 늘려야 합니다. (프리티어는 30GB까지 무료)

1. AWS EC2 -> Elastic Block Store -> 볼륨
2. 해당 볼륨 선택 -> 작업 -> 볼륨 수정 (Modify Volume) -> 크기 변경
3. 서버 재부팅 또는 파일 시스템 확장 명령어 실행 필요

```

```
