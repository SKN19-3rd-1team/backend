# 로컬 Docker 테스트 가이드

이 문서는 AWS EC2에 배포하기 전, 로컬 환경(Windows/WSL 등)에서 Docker를 활용해 프로젝트가 정상 작동하는지 확인하는 방법을 설명합니다.

## 1. 사전 준비 (Prerequisites)

1. **Docker Desktop 설치**: Windows 환경이라면 Docker Desktop이 설치되어 있고 실행 중이어야 합니다.
2. **프로젝트 경로 이동**: 터미널(WSL, Git Bash 등)에서 프로젝트 루트 디렉토리(`Unigo/`)로 이동합니다.

## 2. 환경 설정 및 트러블슈팅 (필수)

로컬 테스트 시 **Windows 호스트의 MySQL**을 사용하므로 다음 설정이 반드시 필요합니다.

### 2.1. `.env` 파일 생성

프로젝트 실행에 필요한 환경 변수 파일을 생성해야 합니다.

```bash
# 예시 파일 복사하여 .env 생성
cp .env.example .env
```

### 2.2. `.env` 설정

Docker Desktop(Windows/Mac) 환경에서는 호스트 접근을 위해 특수한 도메인을 사용합니다.

```bash
# .env 파일 예시
MYSQL_HOST=host.docker.internal
MYSQL_PORT=3306
MYSQL_USER=unigo
MYSQL_PASSWORD=unigo
MYSQL_DB=unigodb
```

> [!NOTE] > `host.docker.internal`은 Docker 컨테이너 내부에서 호스트 머신을 가리키는 주소입니다.
> 만약 Docker가 아니라 WSL 터미널에서 직접 Python을 실행한다면 `172.x.x.x` IP가 필요할 수 있지만, `docker compose`로 실행할 때는 위 설정이 가장 확실합니다.

### 2.2. MySQL 설정 (`my.ini`)

Windows에 설치된 MySQL이 외부 접속을 허용해야 합니다.

1. `my.ini` 파일 열기 (보통 `C:\ProgramData\MySQL\MySQL Server 8.0\my.ini`).
2. `bind-address` 설정 변경:
   ```ini
   bind-address = 0.0.0.0
   ```
3. MySQL 서비스 재시작.

### 2.3. Windows 방화벽 설정 (Port 3306)

WSL/Docker에서 Windows로 접근하려면 방화벽을 열어야 합니다.

1. `고급 보안이 포함된 Windows Defender 방화벽` 실행.
2. **인바운드 규칙** -> **새 규칙**.
3. **포트** -> **TCP** -> **특정 로컬 포트: 3306**.
4. **연결 허용**.
5. 프로필: **도메인, 개인, 공용** 모두 체크.
6. 이름: `WSL MySQL 3306` 등 입력 후 마침.

## 3. Docker 실행

터미널에서 다음 명령어를 입력합니다.

```bash
# 이미지를 빌드하고 백그라운드에서 실행
docker compose up -d --build
```

- `-d`: 백그라운드 모드 (터미널을 계속 쓸 수 있음)
- `--build`: 코드가 변경되었다면 이미지를 새로 만듦

## 4. 접속 및 확인

1. **웹 브라우저 접속**:

   - 주소창에 `http://localhost` 입력을 시도합니다.
   - Nginx가 80번 포트에서 대기하고 있다가 Django(8000번)로 연결해줍니다.

2. **로그 확인**:

   - 실행 중에 에러가 없는지 실시간 확인:

```bash
docker compose logs -f
```

- `ctrl + c`를 누르면 로그 확인을 종료합니다(서버는 안 꺼짐).

3. **초기 세팅 (최초 1회)**:

   - DB 테이블 생성 및 관리자 계정 생성이 필요할 수 있습니다.

```bash
# DB 마이그레이션
docker compose exec web python manage.py migrate

# 관리자 계정 생성
docker compose exec web python manage.py createsuperuser
```

## 5. 테스트 종료

테스트가 끝났다면 다음 명령어로 컨테이너를 정리합니다.

```bash
# 컨테이너 중지 및 삭제
docker compose down
```

- 데이터(`mysql_data` 볼륨)는 `down`을 해도 지워지지 않고 남아있습니다.
- 데이터를 완전히 초기화하고 싶다면: `docker compose down -v`

### 💡 완전 초기화 (Docker 이미지, 컨테이너, 볼륨 모두 삭제)

테스트 흔적을 남기지 않고 싹 지우고 싶다면 다음 명령어를 사용하세요.

```bash
docker compose down --rmi all -v --remove-orphans
```

- `--rmi all`: 사용된 모든 이미지를 삭제
- `-v`: 생성된 볼륨(데이터)을 모두 삭제
- `--remove-orphans`: 설정 파일에서 정의되지 않은 "고아" 컨테이너도 삭제
