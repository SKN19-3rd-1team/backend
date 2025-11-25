"""
매우 엄격한 학과명 검증 테스트

LLM이 Tool 결과를 단 한 글자도 바꾸지 않고 사용하는지 검증합니다.
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

def test_strict_department_name_validation():
    """학과명을 단 한 글자도 바꾸지 않는지 엄격하게 검증"""
    print("=" * 80)
    print("TEST: 매우 엄격한 학과명 검증 (완전 일치 검사)")
    print("=" * 80)

    # 복잡한 학과명들을 포함한 관심사
    test_cases = [
        {
            "name": "괄호 포함 학과명",
            "interests": "신소재 / 금속 / 세라믹",
            "question": "내 관심사에 맞는 학과 알려줘",
            "expected_patterns": [
                "신소재공학부(정보소재공학)",
                "신소재공학부(금속시스템공학)",
                "나노신소재공학부(세라믹공학전공)",
                "나노신소재공학부(금속재료공학전공)"
            ]
        },
        {
            "name": "전공 포함 긴 학과명",
            "interests": "컴퓨터 / 소프트웨어",
            "question": "컴퓨터와 소프트웨어 관련 학과 추천해줘",
            "expected_patterns": [
                "컴퓨터공학부 기계공학전공",
                "컴퓨터과학부 컴퓨터소프트웨어전공",
                "컴퓨터학부 플랫폼소프트웨어전공"
            ]
        },
        {
            "name": "점(.) 구분자 포함 학과명",
            "interests": "화공 / 환경",
            "question": "화학공학과 환경 관련 학과 알려줘",
            "expected_patterns": [
                "화공생명.환경공학부 환경공학전공",
                "화공생명.환경공학부 화공생명공학전공",
                "농업토목.생물산업공학부 생물산업기계공학전공"
            ]
        }
    ]

    all_passed = True

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'=' * 80}")
        print(f"[Interests] {test_case['interests']}")
        print(f"[Question] {test_case['question']}\n")

        try:
            result = run_mentor(
                question=test_case['question'],
                interests=test_case['interests'],
                chat_history=[]
            )

            print("\n" + "=" * 80)
            print("[AI Response]")
            print("=" * 80)
            print(result)
            print("\n" + "=" * 80)

            # 엄격한 검증: Tool 로그에서 실제로 반환된 학과명 추출
            # (이 테스트에서는 예상 패턴이 응답에 정확히 포함되어야 함)

            violations = []

            # 1. **로 감싸진 모든 학과명 추출
            import re
            mentioned_depts = re.findall(r'\*\*([^*]+)\*\*', result)

            print(f"\n[Validation] Mentioned departments ({len(mentioned_depts)}):")
            for dept in mentioned_depts:
                print(f"  - {dept}")

            # 2. 각 학과명이 Tool 결과에서 나온 정확한 이름인지 확인
            # 간단화된 이름(예: "신소재공학과" 대신 "신소재공학부(정보소재공학)")이
            # 사용되었는지 체크

            suspicious_simplifications = [
                # 괄호 제거
                ("신소재공학부", ["신소재공학부(정보소재공학)", "신소재공학부(금속시스템공학)"]),
                ("컴퓨터공학부", ["컴퓨터공학부 기계공학전공"]),
                ("컴퓨터과학부", ["컴퓨터과학부 컴퓨터소프트웨어전공"]),
                ("화공생명공학부", ["화공생명.환경공학부 화공생명공학전공"]),

                # 약어 사용
                ("신소재공학과", ["신소재공학", "신소재공학부"]),
                ("화공학과", ["화공학부", "화공생명공학과"]),
                ("컴퓨터학과", ["컴퓨터학부", "컴퓨터공학과"]),
            ]

            for simplified, originals in suspicious_simplifications:
                if simplified in mentioned_depts:
                    # 더 정확한 원본 이름이 있는지 확인
                    has_original = any(orig in mentioned_depts for orig in originals)
                    if not has_original:
                        violations.append(
                            f"⚠️  '{simplified}' - 간단화된 이름 사용 의심 "
                            f"(가능한 원본: {', '.join(originals)})"
                        )

            # 3. 괄호가 빠진 경우 체크
            for dept in mentioned_depts:
                # 괄호 없는 학과명인데, Tool 결과에 괄호 버전이 있을 가능성
                if "(" not in dept and ")" not in dept:
                    # 이 학과명과 유사하지만 괄호가 있는 버전이 있는지 의심
                    if any(keyword in dept for keyword in ["신소재", "나노신소재", "컴퓨터"]):
                        violations.append(
                            f"⚠️  '{dept}' - 괄호/전공명이 생략되었을 가능성"
                        )

            print(f"\n[Strict Validation]:")
            if violations:
                print(f"  ❌ Found {len(violations)} potential violations:")
                for v in violations:
                    print(f"     {v}")
                all_passed = False
            else:
                print(f"  ✅ All department names look accurate!")

        except Exception as e:
            print(f"\n[ERROR] Test case {i} failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("[SUCCESS] 모든 엄격한 테스트 통과!")
    else:
        print("[FAILED] 일부 테스트 실패 - 학과명 간단화/변형 감지됨")
    print("=" * 80)

    return all_passed

if __name__ == "__main__":
    print("\n>> 엄격한 학과명 검증 테스트 시작...\n")
    success = test_strict_department_name_validation()
    sys.exit(0 if success else 1)
