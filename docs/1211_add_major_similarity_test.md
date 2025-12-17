# 학과 유사도 검색 테스트 스크립트 추가

## 1. 개요
Vector DB에 저장된 학과 데이터의 검색 품질을 수동으로 검증하기 위한 테스트 스크립트를 추가하였습니다.

## 2. 작업 내용
- **스크립트 생성**: `backend/scripts/test_major_similarity.py`
  - 사용자의 입력을 받아 `get_major_vectorstore()`를 통해 연결된 Pinecone Vector DB에서 유사도 검색을 수행합니다.
  - 검색 결과(학과명, 유사도 점수, 메타데이터 등)를 출력하여 검색 모델의 성능을 육안으로 확인할 수 있습니다.
  - 한글 주석을 추가하여 코드 이해를 도왔습니다.

## 3. 실행 방법
```bash
python -m backend.scripts.test_major_similarity
```

## 4. 참고 사항
- 이 스크립트는 `langchain_pinecone` 및 `backend.rag` 모듈에 의존합니다.
- 실행 전 `conda activate langchain_env` 및 환경 변수 설정(`.env`)이 필요합니다.
