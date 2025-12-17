from django.shortcuts import render, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.models import User
import json
import sys
import os
import uuid
import logging
import time
from django.contrib.auth.decorators import login_required

logger = logging.getLogger("unigo_app")

# 모델
from .models import Conversation, Message, MajorRecommendation, UserProfile

# 백엔드 임포트를 위해 프론트엔드 루트를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if frontend_dir not in sys.path:
    sys.path.append(frontend_dir)

try:
    from backend.main import run_mentor_stream, run_major_recommendation
    from backend.rag.tools import summarize_conversation_history
except ImportError as e:
    logger.error(f"Backend import failed: {e}")
    run_mentor_stream = None
    run_major_recommendation = None
    summarize_conversation_history = None


# ============================================
# Page Views
# ============================================


def auth(request):
    """인증(로그인/가입) 페이지 렌더링"""
    if request.user.is_authenticated:
        return redirect("unigo_app:chat")
    return render(request, "unigo_app/auth.html")


def chat(request):
    """채팅 페이지 렌더링"""
    context = {}
    if request.user.is_authenticated:
        # Custom image handling
        custom_image_url = None
    if request.user.is_authenticated:
        # 커스텀 이미지 처리
        custom_image_url = None
        # [수정됨] 플래그가 설정된 경우에만 커스텀 이미지 사용
        if (
            hasattr(request.user, "profile")
            and request.user.profile.custom_image
            and request.user.profile.use_custom_image
        ):
            custom_image_url = (
                f"{request.user.profile.custom_image.url}?v={int(time.time())}"
            )

        # UserProfile에서 캐릭터 가져오기
        try:
            character = request.user.profile.character
        except Exception:
            character = "rabbit"

        # 이미지 파일명 매핑 (js/chat.js 로직과 동일하게)
        filename = character

        context["character_code"] = character
        context["character_image"] = filename
        context["custom_image_url"] = custom_image_url

    return render(request, "unigo_app/chat.html", context)


def setting(request):
    """설정 페이지 렌더링"""
    if not request.user.is_authenticated:
        return redirect("unigo_app:auth")

    context = {}

    # 커스텀 이미지 처리
    custom_image_url = None
    # [수정됨] 플래그가 설정된 경우에만 커스텀 이미지 사용
    if (
        hasattr(request.user, "profile")
        and request.user.profile.custom_image
        and request.user.profile.use_custom_image
    ):
        custom_image_url = (
            f"{request.user.profile.custom_image.url}?v={int(time.time())}"
        )

    try:
        character = request.user.profile.character
        filename = character
    except Exception:
        filename = "rabbit"

    context["character_image"] = filename
    context["custom_image_url"] = custom_image_url

    return render(request, "unigo_app/setting.html", context)


def character_select(request):
    """캐릭터 선택 페이지 렌더링"""
    if not request.user.is_authenticated:
        return redirect("unigo_app:auth")
    return render(request, "unigo_app/character_select.html")


def home(request):
    """
    홈 페이지 (루트 경로)
    - 로그인 상태: 채팅 페이지로 이동
    - 비로그인 상태: 인증(로그인) 페이지로 이동
    """
    if request.user.is_authenticated:
        return redirect("unigo_app:chat")
    return redirect("unigo_app:auth")


# ============================================
# Auth API
# ============================================


def auth_signup(request):
    """
    회원가입 API

    Args:
        request (HttpRequest): JSON 바디
            - username (str): 아이디
            - password (str): 비밀번호
            - email (str): 이메일 (선택)

    Returns:
        JsonResponse:
            - message (str): 성공 메시지
            - user (dict): 생성된 사용자 정보
            - error (str): 에러 메시지 (중복 등)
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        username = data.get("username")
        password = data.get("password")
        email = data.get("email", "")

        if not username or not password:
            return JsonResponse({"error": "Username and password required"}, status=400)

        if User.objects.filter(username=username).exists():
            return JsonResponse({"error": "Username already exists"}, status=400)

        user = User.objects.create_user(
            username=username, password=password, email=email
        )
        login(request, user)  # 가입 후 자동 로그인

        return JsonResponse(
            {
                "message": "Signup successful",
                "user": {"id": user.id, "username": user.username},
            }
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def auth_login(request):
    """
    로그인 API
    username 또는 email을 사용하여 로그인을 시도합니다.

    Args:
        request (HttpRequest): JSON 바디
            - username (str): 아이디 또는 이메일
            - password (str): 비밀번호

    Returns:
        JsonResponse:
            - message (str): 성공 메시지
            - user (dict): 사용자 정보
            - error (str): 에러 메시지
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        username_or_email = data.get("username")  # 이메일 또는 username
        password = data.get("password")

        if not username_or_email or not password:
            return JsonResponse(
                {"error": "Username/Email and password required"}, status=400
            )

        # 이메일 형식인지 확인
        user = None
        if "@" in username_or_email:
            # 이메일로 로그인 시도
            try:
                user_obj = User.objects.get(email=username_or_email)
                user = authenticate(
                    request, username=user_obj.username, password=password
                )
            except User.DoesNotExist:
                pass
        else:
            # Username으로 로그인 시도
            user = authenticate(request, username=username_or_email, password=password)

        if user is not None:
            login(request, user)
            return JsonResponse(
                {
                    "message": "Login successful",
                    "user": {
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                    },
                }
            )
        else:
            return JsonResponse({"error": "Invalid credentials"}, status=401)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def auth_logout(request):
    """로그아웃 API"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    logout(request)
    return JsonResponse({"message": "Logout successful"})


def logout_view(request):
    """
    로그아웃 뷰 (GET 요청 처리 및 리다이렉트)
    """
    # 로그아웃 처리 후 클라이언트 세션 스토리지를 정리할 수 있도록
    # 로그아웃 완료 페이지를 렌더링하여 브라우저에서 sessionStorage를 초기화하고
    # 클라이언트를 인증 페이지로 리다이렉트하게 한다.
    logout(request)
    return render(request, "unigo_app/logout.html")


def auth_me(request):
    """현재 사용자 정보 조회 API"""
    if request.user.is_authenticated:
        # 사용자의 채팅 기록 존재 여부 확인
        has_history = Conversation.objects.filter(user=request.user).exists()

        return JsonResponse(
            {
                "is_authenticated": True,
                "user": {
                    "id": request.user.id,
                    "username": request.user.username,
                    "email": request.user.email,
                    "character": request.user.profile.character
                    if hasattr(request.user, "profile")
                    else "rabbit",
                    # [수정됨] use_custom_image가 True일 때만 custom_image_url 반환
                    "custom_image_url": f"{request.user.profile.custom_image.url}?v={int(time.time())}"
                    if hasattr(request.user, "profile")
                    and request.user.profile.custom_image
                    and request.user.profile.use_custom_image
                    else None,
                },
                "has_history": has_history,
            }
        )
    return JsonResponse({"is_authenticated": False})


def auth_check_email(request):
    """이메일 중복 확인 API (Public)"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        email = data.get("email")

        if not email:
            return JsonResponse({"error": "Email required"}, status=400)

        if User.objects.filter(email=email).exists():
            return JsonResponse(
                {"exists": True, "message": "중복 이메일이 존재합니다."}
            )

        return JsonResponse({"exists": False, "message": "사용 가능한 이메일입니다."})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def auth_check_username(request):
    """아이디(닉네임) 중복 확인 API (Public)"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        username = data.get("username")

        if not username:
            return JsonResponse({"error": "Username required"}, status=400)

        if User.objects.filter(username=username).exists():
            return JsonResponse(
                {"exists": True, "message": "중복 닉네임이 존재합니다."}
            )

        return JsonResponse({"exists": False, "message": "사용 가능한 닉네임입니다."})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ============================================
# Setting API
# ============================================


@login_required
def check_username(request):
    """닉네임 중복 확인 API"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        username = data.get("username")

        if not username:
            return JsonResponse({"error": "Username required"}, status=400)

        if User.objects.filter(username=username).exists():
            return JsonResponse(
                {"exists": True, "message": "이미 사용 중인 닉네임입니다."}
            )

        return JsonResponse({"exists": False, "message": "사용 가능한 닉네임입니다."})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def change_nickname(request):
    """닉네임 변경 API"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        new_username = data.get("username")
        password = data.get("password")

        if not new_username or not password:
            return JsonResponse({"error": "Username and password required"}, status=400)

        # 현재 비밀번호 검증
        user = authenticate(request, username=request.user.username, password=password)
        if user is None:
            return JsonResponse({"error": "비밀번호가 일치하지 않습니다."}, status=400)

        # 중복 확인
        if User.objects.filter(username=new_username).exists():
            return JsonResponse({"error": "이미 사용 중인 닉네임입니다."}, status=400)

        # 닉네임 변경
        user.username = new_username
        user.save()

        return JsonResponse({"message": "내용이 변경되었습니다."})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def change_password(request):
    """비밀번호 변경 API"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        current_password = data.get("current_password")
        new_password = data.get("new_password")

        if not current_password or not new_password:
            return JsonResponse({"error": "All fields required"}, status=400)

        # 현재 비밀번호 검증
        user = authenticate(
            request, username=request.user.username, password=current_password
        )
        if user is None:
            return JsonResponse(
                {"error": "현재 비밀번호가 일치하지 않습니다."}, status=400
            )

        # 비밀번호 변경
        user.set_password(new_password)
        user.save()

        # 세션 유지
        update_session_auth_hash(request, user)

        return JsonResponse({"message": "내용이 변경되었습니다."})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def update_character(request):
    """캐릭터 변경 API"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        character = data.get("character")

        if not character:
            return JsonResponse({"error": "Character required"}, status=400)

        # 프로필 가져오기 (없으면 생성)
        profile, created = UserProfile.objects.get_or_create(user=request.user)

        # [수정됨] 캐릭터 변경 시 커스텀 이미지 사용 해제
        profile.character = character
        profile.use_custom_image = False
        profile.save()

        return JsonResponse({"message": f"Character updated to {character}"})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def upload_character_image(request):
    """커스텀 캐릭터 이미지 업로드 API"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        if "image" not in request.FILES:
            return JsonResponse({"error": "No image file provided"}, status=400)

        image_file = request.FILES["image"]

        # 프로필 가져오기
        profile, created = UserProfile.objects.get_or_create(user=request.user)

        # [수정됨] 기존 이미지가 있다면 삭제하여 파일명 중복 시 덮어쓰기 유도
        if profile.custom_image:
            profile.custom_image.delete(save=False)

        # 이미지 저장
        profile.custom_image = image_file
        # [수정됨] 이미지 업로드 시 커스텀 이미지 사용 설정
        profile.use_custom_image = True
        profile.save()

        return JsonResponse(
            {
                "message": "Image uploaded successfully",
                "image_url": f"{profile.custom_image.url}?v={int(time.time())}",
            }
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ============================================
# Setting API
# ============================================


def delete_account(request):
    """
    계정탈퇴 API
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        username = data.get("username")
        password = data.get("password")
        email = data.get("email", "")

        # required fields 작성 여부 확인
        if not username or not password:
            return JsonResponse({"error": "Username and password required"}, status=400)

        # 사용자 존재 여부 확인
        if not User.objects.filter(username=username).exists():
            return JsonResponse({"error": "Username doesn't exist"}, status=400)

        # 현재 비밀번호 검증
        user = authenticate(request, username=request.user.username, password=password)
        if user is None:
            return JsonResponse(
                {"error": "현재 비밀번호가 일치하지 않습니다."}, status=400
            )

        # 로그아웃
        logout(request)

        # 계정 삭제
        user.delete()

        # 삭제 성공
        return JsonResponse(
            {
                "message": "Deletion successful",
                "user": {"id": user.id, "username": user.username},
            }
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ============================================
# Chat & Feature API
# ============================================


def stream_chat_responses(conversation, message_text, chat_history_for_ai):
    """채팅 응답을 스트리밍하는 제너레이터"""

    if not run_mentor_stream:
        error_msg = "챗봇 백엔드가 연결되지 않았습니다. 관리자에게 문의하세요."

        yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"

        return

    full_response_content = ""

    try:
        # [수정] stream_mode=["messages", "updates"] 로 토큰 스트리밍과 상태 업데이트를 모두 받음
        stream = run_mentor_stream(
            question=message_text,
            chat_history=chat_history_for_ai,
            mode="react",
            stream_mode=["messages", "updates"],
        )

        for mode, chunk in stream:
            # 1. 메시지 스트리밍 (토큰 단위)
            if mode == "messages":
                message, metadata = chunk
                # 에이전트 노드에서 생성된 AIMessageChunk인 경우에만 처리
                if (
                    metadata.get("langgraph_node") == "agent"
                    and hasattr(message, "content")
                    and message.content
                ):
                    # 토큰 전송
                    # [2025-12-16] Fix: Handle list-type content (e.g. from Anthropic/OpenAI multimodal outputs)
                    # to prevent "[object Object]" in frontend.
                    content_str = ""
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, str):
                                content_str += block
                            elif isinstance(block, dict) and "text" in block:
                                content_str += block["text"]
                    else:
                        content_str = str(message.content)

                    # 빈 문자열이면 스킵 (불필요한 패킷 방지)
                    if not content_str:
                        continue

                    data = {"type": "delta", "content": content_str}
                    yield f"data: {json.dumps(data)}\n\n"

            # 2. 상태 업데이트 (툴 호출 등 확인)
            elif mode == "updates":
                step_name = list(chunk.keys())[0]

                if step_name == "agent":
                    agent_messages = chunk["agent"].get("messages", [])
                    if agent_messages:
                        last_ai_message = agent_messages[-1]

                        # 도구 사용 결정 시 상태 업데이트
                        if (
                            hasattr(last_ai_message, "tool_calls")
                            and last_ai_message.tool_calls
                        ):
                            tool_names = [
                                call["name"] for call in last_ai_message.tool_calls
                            ]
                            status_message = f"Tool: {', '.join(tool_names)}"
                            data = {"type": "status", "content": status_message}
                            yield f"data: {json.dumps(data)}\n\n"

                        # [중요] DB 저장을 위해 최종 답변 업데이트 (마지막 메시지 기준)
                        if last_ai_message.content:
                            full_response_content = last_ai_message.content

    except Exception as e:
        logger.error(f"AI Stream Error: {e}", exc_info=True)

        data = {"type": "error", "content": "AI 서버에서 오류가 발생했습니다."}

        yield f"data: {json.dumps(data)}\n\n"

        return

    # 전체 응답 DB 저장

    if full_response_content:
        Message.objects.create(
            conversation=conversation, role="assistant", content=full_response_content
        )

        logger.info(f"Streamed response saved to DB for conversation {conversation.id}")


def chat_api(request):
    """
    챗봇 대화 API (DB 저장 및 RAG 답변 생성)

    사용자의 메시지를 받아 DB에 저장하고, Backend RAG 엔진(`run_mentor`)을 호출하여
    답변을 생성한 뒤, 이를 다시 DB에 저장하고 프론트엔드에 반환합니다.

    Args:
        request (HttpRequest): JSON 바디를 포함한 POST 요청
            - message (str): 사용자 질문
            - history (list): (Optional) 프론트엔드에서 관리하는 대화 내역
            - session_id (str): 비로그인 사용자 식별용
            - conversation_id (int): 로그인 사용자 대화방 ID

    Returns:
        JsonResponse:
            - response (str): AI 답변
            - session_id (str): 세션 ID (비로그인 시)
            - conversation_id (int): 대화방 ID
            - error (str): 에러 메시지 (실패 시)
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        message_text = data.get("message")
        session_id = data.get("session_id")  # 비로그인 사용자용 세션 ID
        conversation_id = data.get("conversation_id")  # 로그인 사용자용: 현재 대화 ID

        if not message_text:
            return JsonResponse({"error": "Empty message"}, status=400)

        # 1. 대화 세션 찾기 또는 생성
        conversation = None
        if request.user.is_authenticated:
            if conversation_id:
                try:
                    conversation = Conversation.objects.get(
                        id=conversation_id, user=request.user
                    )
                except Conversation.DoesNotExist:
                    conversation = Conversation.objects.create(
                        user=request.user,
                        session_id=str(uuid.uuid4()),
                        title=message_text[:20],
                    )
            else:
                conversation = Conversation.objects.create(
                    user=request.user,
                    session_id=str(uuid.uuid4()),
                    title=message_text[:20],
                )
        else:
            if not session_id:
                session_id = str(uuid.uuid4())
            conversation, _ = Conversation.objects.get_or_create(
                session_id=session_id, defaults={"title": message_text[:20]}
            )

        # 2. 사용자 메시지 DB 저장
        Message.objects.create(
            conversation=conversation, role="user", content=message_text
        )

        # 3. DB 기반 히스토리 구성
        db_messages = conversation.messages.order_by("created_at").all()
        chat_history_for_ai = [
            {"role": msg.role, "content": msg.content} for msg in db_messages
        ]

        # 4. 스트리밍 응답 생성 및 반환
        response = StreamingHttpResponse(
            stream_chat_responses(conversation, message_text, chat_history_for_ai),
            content_type="text/event-stream",
        )
        response["Cache-Control"] = "no-cache"

        # conversation_id를 헤더로 전달 (클라이언트가 첫 메시지 후 ID를 알 수 있도록)
        response["X-Conversation-Id"] = conversation.id
        if not request.user.is_authenticated:
            response["X-Session-Id"] = conversation.session_id

        return response

    except Exception as e:
        logger.error(f"Error in chat_api (stream): {e}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def chat_history(request):
    """
    사용자 대화 기록 조회 API
    """
    try:
        # 최근 대화 세션 가져오기
        conversation = (
            Conversation.objects.filter(user=request.user)
            .order_by("-updated_at")
            .first()
        )

        if not conversation:
            return JsonResponse({"history": []})

        messages = conversation.messages.order_by("created_at")

        history_data = [
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
            }
            for msg in messages
        ]

        return JsonResponse({"history": history_data})

    except Exception as e:
        logger.error(f"Error in chat_history: {e}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def save_chat_history(request):
    """
    현재 세션의 채팅 내용을 새로운 Conversation으로 저장하는 API
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        chat_history = data.get("history", [])

        if not chat_history:
            return JsonResponse({"message": "No chat history to save"}, status=200)

        title = "새 대화"
        for msg in chat_history:
            if msg.get("role") == "user":
                title = msg.get("content", "새 대화")[:50]
                break

        new_conversation = Conversation.objects.create(
            user=request.user,
            session_id=str(uuid.uuid4()),
            title=title,
        )

        for msg in chat_history:
            Message.objects.create(
                conversation=new_conversation,
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                metadata=msg.get("metadata"),
            )

        logger.info(f"Chat history saved for user={request.user.username}")

        return JsonResponse(
            {
                "message": "Chat history saved successfully",
                "conversation_id": new_conversation.id,
            }
        )

    except Exception as e:
        logger.error(f"Error in save_chat_history: {e}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def list_conversations(request):
    """
    로그인 사용자의 과거 대화 세션 리스트 반환
    """
    try:
        convs = Conversation.objects.filter(user=request.user).order_by("-updated_at")

        data = []
        for c in convs:
            last_msg = c.get_last_message()
            preview = last_msg.content[:80] if last_msg else ""
            data.append(
                {
                    "id": c.id,
                    "title": c.title,
                    "created_at": c.created_at.isoformat(),
                    "updated_at": c.updated_at.isoformat(),
                    "message_count": c.get_message_count(),
                    "last_message_preview": preview,
                }
            )

        return JsonResponse({"conversations": data})

    except Exception as e:
        logger.error(f"Error in list_conversations: {e}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def delete_conversation(request, conversation_id):
    """
    대화 삭제 API
    로그인 사용자의 특정 대화 세션을 삭제합니다. 삭제 시 관련 메시지도 함께 삭제됩니다(CASCADE).

    Args:
        request (HttpRequest): 요청 객체
        conversation_id (int): 삭제할 대화 ID

    Returns:
        JsonResponse: 성공/실패 메시지
    """
    if request.method != "DELETE" and request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        conversation = Conversation.objects.get(id=conversation_id, user=request.user)
        conversation.delete()
        logger.info(
            f"Conversation {conversation_id} deleted by user {request.user.username}"
        )
        return JsonResponse({"message": "Conversation deleted successfully"})

    except Conversation.DoesNotExist:
        return JsonResponse(
            {"error": "Conversation not found or permission denied"}, status=404
        )
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def load_conversation(request):
    """
    특정 conversation의 메시지들을 반환
    """
    try:
        conv_id = request.GET.get("conversation_id")
        if not conv_id:
            return JsonResponse({"error": "conversation_id required"}, status=400)

        try:
            conv = Conversation.objects.get(id=conv_id, user=request.user)
        except Conversation.DoesNotExist:
            return JsonResponse({"error": "Conversation not found"}, status=404)

        msgs = conv.messages.order_by("created_at")
        messages = [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in msgs
        ]

        return JsonResponse(
            {"conversation": {"id": conv.id, "title": conv.title, "messages": messages}}
        )

    except Exception as e:
        logger.error(f"Error in load_conversation: {e}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def reset_chat_history(request):
    """
    사용자 대화 기록 초기화 API (새 채팅)
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    # 이 API는 더 이상 클라이언트에서 직접 호출하지 않음.
    # 클라이언트가 '새 채팅' 버튼을 누르면, 클라이언트 자체적으로 UI를 리셋하고
    # conversation_id를 null로 설정하여 다음 메시지 전송 시 서버에서
    # 새 대화를 시작하도록 유도.
    return JsonResponse({"message": "Client-side reset is preferred."})


def onboarding_api(request):
    """
    온보딩 질문 답변 API (전공 추천 실행)

    사용자가 입력한 온보딩 정보(선호 과목, 흥미 등)를 바탕으로
    `run_major_recommendation`을 실행하여 맞춤형 전공을 추천하고 결과를 저장합니다.

    Args:
        request (HttpRequest): JSON 바디
            - answers (dict): 온보딩 답변 딕셔너리
            - session_id (str): 세션 ID

    Returns:
        JsonResponse:
            - recommended_majors (list): 추천 전공 목록
            - user_profile_text (str): 사용자 페르소나 분석 결과
            - session_id (str): 세션 ID
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        answers = data.get("answers")
        session_id = data.get("session_id")
        conversation_id = data.get(
            "conversation_id"
        )  # [MODIFIED] Receive existing conversation ID
        history = data.get("history", [])

        if not answers:
            return JsonResponse({"error": "Empty answers"}, status=400)

        if not run_major_recommendation:
            return JsonResponse({"error": "Backend not available"}, status=503)

        result = run_major_recommendation(onboarding_answers=answers)

        # 1. Conversation 생성 또는 검색
        user = request.user if request.user.is_authenticated else None
        if not session_id:
            session_id = str(uuid.uuid4())

        title = "전공 추천 상담"
        if history and len(history) > 0:
            first_answer = next(
                (h["content"] for h in history if h["role"] == "user"), "전공 추천 상담"
            )
            title = first_answer[:30]

        conversation = None
        created = True  # reuse flag

        # [MODIFIED] Try to reuse existing conversation
        if user and conversation_id:
            try:
                conversation = Conversation.objects.get(id=conversation_id, user=user)
                created = False
            except Conversation.DoesNotExist:
                pass

        if not conversation:
            # Create new
            if user:
                conversation = Conversation.objects.create(
                    user=user, session_id=session_id, title=title
                )
            else:
                # Session based
                conversation, created = Conversation.objects.get_or_create(
                    session_id=session_id, defaults={"title": title}
                )

        # 2. Onboarding History 저장 (Message 모델)
        if conversation and history:
            # If reusing conversation, we must be careful not to duplicate existing messages.
            # Simple heuristic: If we reused (not created), assume previous messages are safe types.
            # But 'history' passed from frontend includes EVERYTHING.
            # We should only append the *tail* that is not in DB.
            # Since Onboarding follows a linear flow without API saves (except the start 'Hello'?),
            # We can count existing messages in DB.

            existing_count = conversation.messages.count()

            # If newly created, existing_count is 0. All history is new.
            # If reused, existing_count might be e.g. 2 ("Hello", "AI response").
            # Frontend history might be 6 ("Hello", "AI", "Start", "Q1", "A1"...).
            # So we take history[2:] ?

            # Safety check: Verify timestamps or content? Hard without IDs.
            # Let's trust the count for linear append.

            msgs_to_save = history[existing_count:]

            for msg in msgs_to_save:
                Message.objects.create(
                    conversation=conversation,
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                )

            # 3. 추천 결과(Summary)도 Assistant Message로 저장
            recs = result.get("recommended_majors", [])
            summary_text = "온보딩 답변을 바탕으로 추천 전공 TOP 5를 정리했어요:\n"
            for idx, major in enumerate(recs[:5]):
                summary_text += f"{idx + 1}. {major.get('major_name')} (점수 {major.get('score', 0):.2f})\n"
            summary_text += (
                "\n필요하면 위 전공 중 궁금한 학과를 지정해서 더 물어봐도 좋아요!"
            )

            Message.objects.create(
                conversation=conversation, role="assistant", content=summary_text
            )

        MajorRecommendation.objects.create(
            user=user,
            session_id=session_id if not user else "",
            onboarding_answers=answers,
            recommended_majors=result.get("recommended_majors", []),
        )

        result["session_id"] = session_id
        if conversation:
            result["conversation_id"] = conversation.id

        return JsonResponse(result)

    except Exception as e:
        logger.error(f"Error in onboarding_api: {e}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


# ============================================
# Chat Summarization API
# ============================================


def summarize_conversation(request):
    """
    대화 요약 API
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        chat_history = data.get("history", [])

        if not chat_history:
            return JsonResponse({"error": "Empty chat history"}, status=400)

        if not summarize_conversation_history:
            return JsonResponse({"error": "Backend not available"}, status=503)

        summary = summarize_conversation_history(chat_history)
        return JsonResponse({"summary": summary})

    except Exception as e:
        logger.error(f"Error in summarize_chat: {e}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)
