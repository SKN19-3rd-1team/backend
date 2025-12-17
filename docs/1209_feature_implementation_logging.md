# 로깅 시스템 구축 보고서

**작성일**: 2025-12-09  
**상태**: 구현 완료

---

## 1. 개요
운영 환경에서의 에러 추적, 디버깅 효율성 향상, 그리고 사용자 활동 모니터링을 위해 체계적인 로깅(Logging) 시스템을 구축했습니다. 기존의 `print` 문 기반 디버깅에서 벗어나 Django의 정식 Logging 기능을 활용합니다.

## 2. 구현 내용

### 2.1 로깅 설정 (`unigo/settings.py`)
Django의 `LOGGING` 설정을 통해 다음 구성을 적용했습니다.

*   **로그 저장 위치**: `unigo/logs/unigo.log`
*   **로그 레벨**:
    *   파일 저장: `INFO` 레벨 이상 (운영 기록)
    *   콘솔 출력: `DEBUG` 레벨 이상 (개발 디버깅)
*   **로그 포맷**:
    *   Verbose (파일): `[LEVEL] [TIME] [MODULE] [PROCESS] [MESSAGE]`
    *   Simple (콘솔): `[LEVEL] [MESSAGE]`
*   **로테이션**: 로그 파일 크기가 15MB를 초과하면 자동으로 회전(Rotation)하며 최대 10개까지 백업 보관 (`RotatingFileHandler` 적용).

### 2.2 애플리케이션 로깅 (`unigo_app/views.py`)
`unigo_app` 로거를 생성하여 주요 API 엔드포인트에 적용했습니다.

*   **적용된 API**:
    *   `chat_api`: 챗봇 대화 요청 및 응답
    *   `onboarding_api`: 온보딩 추천 요청
    *   Backend Import: 모듈 로드 실패 등 치명적 오류
*   **로그 기록 포인트**:
    *   `Request` 수신 시점 (INFO)
    *   사용자 입력 데이터 (DEBUG)
    *   처리 성공 시점 (INFO)
    *   예외(Exception) 발생 시 스택 트레이스 포함 (ERROR, `exc_info=True`)

## 3. 사용 방법

### 3.1 로그 확인
서버 실행 후 생성되는 로그 파일을 확인합니다.

```bash
# 실시간 로그 확인 (Linux/Mac/Git Bash)
tail -f unigo/logs/unigo.log

# 로그 파일 위치
unigo/logs/unigo.log
```

### 3.2 코드에서 로그 남기기
새로운 기능을 개발할 때 다음과 같이 로거를 사용하세요.

```python
import logging
logger = logging.getLogger('unigo_app')

def my_function():
    logger.debug("디버그용 상세 정보")
    logger.info("주요 처리 완료 알림")
    
    try:
        # ... logic ...
    except Exception as e:
        logger.error(f"에러 발생: {e}", exc_info=True)
```

## 4. 기대 효과
*   **안정성**: 서버 중단 없이 에러 원인을 파악할 수 있습니다.
*   **모니터링**: 사용자의 주요 활동(대화 시도, 로그인 등)을 추적할 수 있습니다.
*   **유지보수**: `print` 문이 콘솔에 섞이는 것을 방지하고 구조화된 로그를 관리할 수 있습니다.
