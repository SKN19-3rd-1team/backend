from django.contrib import admin
from .models import (
    Conversation,
    Message,
    MajorRecommendation,
    Major,
    MajorCategory,
    University,
)


# ============================================
# 대화 관련 Admin
# ============================================


class MessageInline(admin.TabularInline):
    """대화 내 메시지를 인라인으로 표시"""

    model = Message
    extra = 0
    readonly_fields = ("created_at",)
    fields = ("role", "content", "created_at")


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "get_user_display",
        "title",
        "message_count",
        "created_at",
        "updated_at",
    )
    list_filter = ("created_at", "updated_at")
    search_fields = ("title", "session_id", "user__username")
    readonly_fields = ("created_at", "updated_at", "session_id")
    inlines = [MessageInline]

    def get_user_display(self, obj):
        if obj.user:
            return obj.user.username
        return f"Guest ({obj.session_id[:8]})"

    get_user_display.short_description = "User"

    def message_count(self, obj):
        return obj.get_message_count()

    message_count.short_description = "Messages"


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "conversation", "role", "content_preview", "created_at")
    list_filter = ("role", "created_at")
    search_fields = ("content", "conversation__title")
    readonly_fields = ("created_at",)

    def content_preview(self, obj):
        return obj.content[:100] + "..." if len(obj.content) > 100 else obj.content

    content_preview.short_description = "Content"


@admin.register(MajorRecommendation)
class MajorRecommendationAdmin(admin.ModelAdmin):
    list_display = ("id", "get_user_display", "get_preferred_majors", "created_at")
    list_filter = ("created_at",)
    search_fields = ("session_id", "user__username")
    readonly_fields = ("created_at",)

    def get_user_display(self, obj):
        if obj.user:
            return obj.user.username
        return f"Guest ({obj.session_id[:8]})"

    get_user_display.short_description = "User"

    def get_preferred_majors(self, obj):
        return obj.onboarding_answers.get("preferred_majors", "N/A")

    get_preferred_majors.short_description = "Preferred Majors"


# ============================================
# 데이터 DB Admin (Read-Only)
# ============================================


@admin.register(Major)
class MajorAdmin(admin.ModelAdmin):
    list_display = (
        "major_name",
        "major_id",
        "salary",
        "employment_rate",
        "acceptance_rate",
    )
    search_fields = ("major_name", "major_id", "summary")
    list_filter = ("employment",)

    # Managed=False이므로 데이터 보호를 위해 추가/삭제 제한 (선택 사항)
    # 필요시 주석 해제하여 쓰기 권한 부여 가능
    # def has_add_permission(self, request):
    #     return False
    # def has_delete_permission(self, request, obj=None):
    #     return False


@admin.register(MajorCategory)
class MajorCategoryAdmin(admin.ModelAdmin):
    list_display = ("category_name",)
    search_fields = ("category_name",)


@admin.register(University)
class UniversityAdmin(admin.ModelAdmin):
    list_display = ("name", "code", "url")
    search_fields = ("name", "code")
