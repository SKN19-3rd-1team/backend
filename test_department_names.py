"""
학과 추천 시 Tool 결과의 정확한 학과명 사용 테스트

LLM이 list_departments에서 반환된 학과명을 정확히 사용하는지 확인합니다.
"""
import sys
import io
from pathlib import Path

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from backend.main import run_mentor

def test_accurate_department_names():
    """학과 추천 시 정확한 학과명 사용 확인"""
    print("=" * 80)
    print("TEST: 학과 추천 시 정확한 학과명 사용 확인")
    print("=" * 80)

    # query_answer.md와 동일한 관심사 사용
    interests = "기계 / 자동차 / 로봇, 화학 / 화공 / 신소재, 약학"
    question = "내 관심사에 맞는 과 추천해줘"

    print(f"\n[Interests] {interests}")
    print(f"[Question] {question}\n")

    try:
        # run_mentor 호출
        result = run_mentor(
            question=question,
            interests=interests,
            chat_history=[]
        )

        print("\n" + "=" * 80)
        print("[AI Response]")
        print("=" * 80)
        print(result)
        print("\n" + "=" * 80)

        # 검증: 잘못 변경된 학과명 체크
        violations = []

        # 1. "로봇공학과"가 있으면 안 됨 (정확한 이름은 "지능로봇공학과")
        if "로봇공학과**" in result and "지능로봇공학과**" not in result:
            violations.append("❌ '로봇공학과' 발견 (정확한 이름: 지능로봇공학과)")

        # 2. "화공학과"가 있으면 안 됨 (정확한 이름은 "화공학부" 등)
        if "화공학과**" in result and "화공학부**" not in result:
            violations.append("❌ '화공학과' 발견 (정확한 이름: 화공학부 또는 화공생명공학과)")

        # 3. "신소재공학과"만 있으면 체크 (정확한 이름은 "신소재공학" 또는 "신소재공학부")
        # 단, "신소재공학부", "나노신소재공학부" 등은 OK
        if "**신소재공학과**" in result:
            violations.append("❌ '신소재공학과' 발견 (정확한 이름: 신소재공학, 신소재공학부 등)")

        # 4. "화학과"가 독립적으로 있는지 체크
        # Tool 결과를 확인해야 하는데, 실제로는 "화학과"가 존재할 수도 있음
        # 이 경우는 실제 tool 호출 로그를 봐야 정확히 판단 가능

        print("\n[Validation] 학과명 정확성 체크:")
        if violations:
            print("  ⚠️  경고: 학과명이 변경/요약되었습니다:")
            for v in violations:
                print(f"     {v}")
            return False
        else:
            print("  ✅ 학과명이 정확하게 사용되었습니다!")
            return True

    except Exception as e:
        print(f"\n[ERROR] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n>> 학과명 정확성 테스트 시작...\n")

    success = test_accurate_department_names()

    print("\n" + "=" * 80)
    if success:
        print("[SUCCESS] 테스트 통과! LLM이 Tool 결과를 정확히 사용했습니다.")
    else:
        print("[FAILED] 테스트 실패 - LLM이 학과명을 변경/요약했습니다.")
    print("=" * 80)
