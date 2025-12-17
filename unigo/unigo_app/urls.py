from django.urls import path
from . import views

app_name = "unigo_app"

urlpatterns = [
    # 페이지
    path("", views.home, name="home"),
    path("auth/", views.auth, name="auth"),
    path("logout/", views.logout_view, name="logout"),
    path("chat/", views.chat, name="chat"),
    path("setting/", views.setting, name="setting"),
    path("setting/character/", views.character_select, name="character_select"),
    # 인증 API
    path("api/auth/signup", views.auth_signup, name="auth_signup"),
    path("api/auth/login", views.auth_login, name="auth_login"),
    path("api/auth/logout", views.auth_logout, name="auth_logout"),
    path("api/auth/me", views.auth_me, name="auth_me"),
    path("api/auth/check-email", views.auth_check_email, name="auth_check_email"),
    path(
        "api/auth/check-username", views.auth_check_username, name="auth_check_username"
    ),
    # 설정 API
    path("api/setting/check-username", views.check_username, name="check_username"),
    path("api/setting/change-nickname", views.change_nickname, name="change_nickname"),
    path("api/setting/change-password", views.change_password, name="change_password"),
    path(
        "api/setting/update-character", views.update_character, name="update_character"
    ),
    path(
        "api/setting/upload-character-image",
        views.upload_character_image,
        name="upload_character_image",
    ),
    path("api/setting/delete", views.delete_account, name="delete_account"),
    # 기능 API
    path("api/chat", views.chat_api, name="chat_api"),
    path("api/chat/history", views.chat_history, name="chat_history"),
    path("api/chat/save", views.save_chat_history, name="save_chat_history"),
    path("api/chat/list", views.list_conversations, name="list_conversations"),
    path("api/chat/load", views.load_conversation, name="load_conversation"),
    path("api/chat/reset", views.reset_chat_history, name="reset_chat_history"),
    path(
        "api/chat/summarize",
        views.summarize_conversation,
        name="summarize_conversation",
    ),
    path(
        "api/chat/delete/<int:conversation_id>",
        views.delete_conversation,
        name="delete_conversation",
    ),
    path("api/onboarding", views.onboarding_api, name="onboarding_api"),
]
