"""
엣지 케이스 테스트: 더 어려운 학과명 변형 감지

LLM이 미묘하게 변형하는 경우도 감지하는지 테스트합니다.
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

def test_edge_cases():
    """복잡한 학과명 변형 감지 테스트"""
    print("=" * 80)
    print("TEST: 복잡한 학과명 변형 감지 (엣지 케이스)")
    print("=" * 80)

    # 매우 긴 학과명, 괄호/슬래시 포함
    interests = "컴퓨터 / 소프트웨어 / 인공지능, 화공 / 신소재"
    question = "내 관심사에 맞는 과 추천해줘"

    print(f"\n[Interests] {interests}")
    print(f"[Question] {question}\n")

    try:
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

        # 복잡한 학과명 체크
        suspicious_patterns = [
            # 괄호 제거 체크
            ("컴퓨터공학부 소프트웨어전공**", "컴퓨터공학부**"),  # 일부 생략
            ("소프트웨어전공**", "컴퓨터공학부"),  # 앞부분 생략

            # 괄호 학과명 단순화
            ("정보소재공학**", "신소재공학부"),  # 괄호 안 내용을 메인으로
            ("금속시스템공학**", "신소재공학부"),  # 괄호 안 내용을 메인으로
        ]

        violations = []
        for shortened, original_part in suspicious_patterns:
            if shortened in result and original_part not in result:
                violations.append(f"의심: '{shortened[:-2]}' (원래 학과명의 일부만 사용)")

        if violations:
            print("\n[Validation] 의심스러운 패턴 발견:")
            for v in violations:
                print(f"  ⚠️  {v}")
            return False
        else:
            print("\n[Validation] ✅ 복잡한 학과명도 정확하게 사용되었습니다!")
            return True

    except Exception as e:
        print(f"\n[ERROR] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n>> 엣지 케이스 테스트 시작...\n")

    success = test_edge_cases()

    print("\n" + "=" * 80)
    if success:
        print("[SUCCESS] 엣지 케이스 테스트 통과!")
    else:
        print("[FAILED] 엣지 케이스 테스트 실패 - 학과명 변형 감지됨")
    print("=" * 80)
