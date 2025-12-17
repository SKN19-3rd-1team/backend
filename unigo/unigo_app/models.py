from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


# ============================================
# 대화 관련 모델
# ============================================


class Conversation(models.Model):
    """
    대화 세션 모델

    로그인 사용자와 비로그인 사용자 모두 지원:
    - 로그인 사용자: user 필드 사용
    - 비로그인 사용자: session_id 필드 사용
    """

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="conversations",
        help_text="로그인한 사용자 (비로그인 시 null)",
    )
    session_id = models.CharField(
        max_length=255, unique=True, help_text="세션 ID (비로그인 사용자 식별용)"
    )
    title = models.CharField(
        max_length=255,
        default="새 대화",
        help_text="대화 제목 (첫 메시지에서 자동 생성)",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["user", "-updated_at"]),
            models.Index(fields=["session_id"]),
        ]

    def __str__(self):
        if self.user:
            return f"{self.user.username} - {self.title}"
        return f"Guest ({self.session_id[:8]}) - {self.title}"

    def get_message_count(self):
        """대화의 메시지 개수"""
        return self.messages.count()

    def get_last_message(self):
        """마지막 메시지"""
        return self.messages.last()


class Message(models.Model):
    """
    개별 메시지 모델

    대화 세션 내의 각 메시지를 저장
    """

    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
        ("system", "System"),
    ]

    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(
        max_length=10,
        choices=ROLE_CHOICES,
        help_text="메시지 역할 (user/assistant/system)",
    )
    content = models.TextField(help_text="메시지 내용")
    metadata = models.JSONField(
        null=True, blank=True, help_text="추가 정보 (tool 호출, 검색 결과 등)"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["conversation", "created_at"]),
        ]

    def __str__(self):
        content_preview = (
            self.content[:50] + "..." if len(self.content) > 50 else self.content
        )
        return f"{self.role}: {content_preview}"


class MajorRecommendation(models.Model):
    """
    전공 추천 결과 모델

    온보딩 질문 답변 기반 전공 추천 결과 저장
    """

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="major_recommendations",
    )
    session_id = models.CharField(
        max_length=255, help_text="세션 ID (비로그인 사용자 식별용)"
    )
    onboarding_answers = models.JSONField(
        help_text="온보딩 질문 답변 (subjects, interests, desired_salary, preferred_majors)"
    )
    recommended_majors = models.JSONField(help_text="추천된 전공 목록 및 점수")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["session_id"]),
        ]

    def __str__(self):
        if self.user:
            return f"{self.user.username} - 전공 추천 ({self.created_at.strftime('%Y-%m-%d')})"
        return f"Guest ({self.session_id[:8]}) - 전공 추천 ({self.created_at.strftime('%Y-%m-%d')})"



# ============================================
# 데이터 DB 모델 (Managed = False)
# ============================================


class Major(models.Model):
    """
    전공 정보 모델 (SQLAlchemy 관리 테이블 - 읽기 전용으로 연동)
    Table: majors
    """

    major_id = models.CharField(
        max_length=255, unique=True, help_text="전공 고유 ID"
    )
    major_name = models.CharField(max_length=255, help_text="전공명")
    
    # 텍스트 필드
    summary = models.TextField(null=True, blank=True, help_text="학과 개요")
    interest = models.TextField(null=True, blank=True, help_text="흥미와 적성")
    property = models.TextField(null=True, blank=True, help_text="학과 특성")
    job = models.TextField(null=True, blank=True, help_text="관련 직업")

    # JSON/LONGTEXT 데이터
    relate_subject = models.TextField(null=True, blank=True, help_text="관련 교과목")
    enter_field = models.TextField(null=True, blank=True, help_text="진출 분야")
    department_aliases = models.TextField(null=True, blank=True, help_text="관련 학과")
    career_act = models.TextField(null=True, blank=True, help_text="진로 탐색 활동")
    qualifications = models.TextField(null=True, blank=True, help_text="관련 자격")
    main_subject = models.TextField(null=True, blank=True, help_text="주요 교과목")
    university = models.TextField(null=True, blank=True, help_text="개설 대학")
    chart_data = models.TextField(null=True, blank=True, help_text="차트 데이터 (JSON)")
    raw_data = models.TextField(null=True, blank=True, help_text="원본 데이터 (JSON)")

    # 통계
    salary = models.FloatField(null=True, blank=True, help_text="평균 연봉 (만원)")
    employment = models.TextField(null=True, blank=True, help_text="취업률 등급")
    employment_rate = models.FloatField(null=True, blank=True, help_text="취업률 (%)")
    acceptance_rate = models.FloatField(null=True, blank=True, help_text="경쟁률")

    class Meta:
        managed = False  # Django가 테이블을 생성/삭제하지 않음
        db_table = "majors"
        verbose_name = "전공 (Major)"
        verbose_name_plural = "전공 목록"

    def __str__(self):
        return f"{self.major_name} ({self.major_id})"


class MajorCategory(models.Model):
    """
    전공 카테고리 모델 (SQLAlchemy 관리 테이블)
    Table: major_categories
    """

    category_name = models.CharField(
        max_length=255, unique=True, help_text="카테고리명 (계열)"
    )
    major_names = models.TextField(help_text="포함된 전공 목록 (JSON)")

    class Meta:
        managed = False
        db_table = "major_categories"
        verbose_name = "전공 계열 (Category)"
        verbose_name_plural = "전공 계열 목록"

    def __str__(self):
        return self.category_name


class University(models.Model):
    """
    대학 정보 모델 (SQLAlchemy 관리 테이블)
    Table: universities
    """

    name = models.CharField(max_length=255, unique=True, help_text="대학명")
    code = models.CharField(max_length=50, null=True, blank=True, help_text="대학 코드")
    url = models.CharField(max_length=500, null=True, blank=True, help_text="홈페이지 URL")

    class Meta:
        managed = False
        db_table = "universities"
        verbose_name = "대학 (University)"
        verbose_name_plural = "대학 목록"

    def __str__(self):
        return self.name


# ============================================
# User Profile
# ============================================


class UserProfile(models.Model):
    """
    사용자 추가 프로필 정보

    auth.User 모델과 1:1 연결되어 추가적인 사용자 설정(예: 캐릭터)을 저장
    """

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    character = models.CharField(
        max_length=50,
        default="rabbit",  # 기본값: 토끼
        help_text="사용자가 선택한 페르소나 캐릭터 (rabbit, bear, fox, etc.)",
    )
    custom_image = models.ImageField(
        upload_to="character_images/",
        null=True,
        blank=True,
        help_text="사용자가 업로드한 커스텀 캐릭터 이미지",
    )
    use_custom_image = models.BooleanField(
        default=False,
        help_text="커스텀 이미지 사용 여부 (캐릭터 선택 시 False, 이미지 업로드 시 True)",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}'s Profile ({self.character})"


# UserProfile 자동 생성/갱신을 위한 시그널 수신
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    # 인스턴스 저장 시 프로필도 함께 저장 (혹시 없을 경우 생성 방지 로직은 위에서 처리됨)
    # 다만, admin 등에서 프로필이 실수로 지워졌을 때를 대비해 get_or_create를 쓸 수도 있으나,
    # 여기서는 표준적인 방식인 save 호출만 수행.
    if hasattr(instance, "profile"):
        instance.profile.save()
