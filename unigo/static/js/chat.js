const chatInput = document.querySelector('.chat-input');
const sendBtn = document.getElementById('chat-send-btn');
const chatCanvas = document.querySelector('.chat-canvas');

const STORAGE_KEY_HISTORY = 'unigo.app.chatHistory';
const STORAGE_KEY_ONBOARDING = 'unigo.app.onboarding';
const STORAGE_KEY_CONVERSATION_ID = 'unigo.app.currentConversationId';
const STORAGE_KEY_RESULT_PANEL = 'unigo.app.resultPanel';


// -- Initialization Handling for Server Data --
document.addEventListener("DOMContentLoaded", () => {
    const config = document.getElementById('chat-config');
    if (config) {
        // Read data attributes
        window.USER_CHARACTER = config.dataset.userCharacter || 'rabbit';
        window.USER_CUSTOM_IMAGE_URL = config.dataset.userCustomImage || '';

        // Sync to localStorage
        localStorage.setItem('user_character', window.USER_CHARACTER);
        if (window.USER_CUSTOM_IMAGE_URL) {
            localStorage.setItem('user_custom_image', window.USER_CUSTOM_IMAGE_URL);
        } else {
            localStorage.removeItem('user_custom_image');
        }
    }

    // Start Init
    init();
});

// Configure marked.js globally
const markedOptions = {
    renderer: new marked.Renderer(),
    gfm: true, // Enable GitHub Flavored Markdown
    breaks: true // Enable GFM line breaks
};
markedOptions.renderer.link = function(href, title, text) {
    return `<a target="_blank" rel="noopener noreferrer" href="${href}" title="${title}">${text}</a>`;
};

// APIs
const API_CHAT_URL = '/api/chat';
const API_ONBOARDING_URL = '/api/onboarding';

// Onboarding Questions Definition
const ONBOARDING_QUESTIONS = [
    {
        key: "subjects",
        label: "선호 교과목",
        prompt: "안녕하세요! 고등학교 과목 중 가장 자신 있거나 흥미로운 과목은 무엇인가요? (예: 수학, 물리를 잘하고 과학 실험을 좋아합니다)",
        placeholder: "예: 수학, 영어, 사회문화"
    },
    {
        key: "interests",
        label: "관심사 및 활동",
        prompt: "평소 즐겨 하는 활동이나 관심 있는 주제는 무엇인가요? 동아리 활동이나 취미도 좋아요.",
        placeholder: "예: 코딩 동아리, 역사 소설 읽기, 유튜브 영상 편집"
    },
    {
        key: "career_goal",
        label: "관심 활동 유형",
        prompt: "구체적인 직업은 몰라도 좋아요. 나중에 어떤 **스타일의 일**을 하고 싶으신가요? (예: 남을 돕는 일, 무언가를 분석하는 일, 창의적인 것을 만드는 일, 몸을 움직이는 일 등)",
        placeholder: "예: 사람들과 소통하며 돕는 일, 데이터를 분석해서 문제를 해결하는 일"
    },
    {
        key: "strengths",
        label: "선호 환경 및 성향",
        prompt: "어떤 상황에서 가장 즐거움이나 보람을 느끼나요? (예: 어려운 문제를 풀었을 때, 친구의 고민을 해결해줬을 때, 조립 설명서를 보고 완벽하게 만들었을 때)",
        placeholder: "예: 혼자 조용히 깊게 생각할 때, 팀원들과 함께 목표를 달성했을 때"
    },
    {
        key: "career_field",
        label: "중요하게 생각하는 가치",
        prompt: "대학 생활이나 미래 직업에서 가장 중요하게 생각하는 것은 무엇인가요? (예: 안정적인 삶, 높은 연봉, 새로운 도전, 사회적 기여)",
        placeholder: "예: 안정적인 직업이 최고예요, 돈을 많이 벌고 싶어요, 사회에 도움이 되고 싶어요"
    },
    {
        key: "topics",
        label: "평소 관심 주제",
        prompt: "평소 유튜브나 뉴스, 책에서 어떤 **주제**를 주로 찾아보시나요? (예: 역사, 우주, 연예, 환경, 기계 등)",
        placeholder: "예: 우주 다큐멘터리, 최신 IT 기기 리뷰, 심리 테스트"
    },
    {
        key: "learning_style",
        label: "이론 탐구 vs 실전 활동",
        prompt: "책상에 앉아 깊게 **이론을 파고드는 것**과, 몸을 움직이며 **실험/실습하는 것** 중 무엇을 더 선호하시나요?",
        placeholder: "예: 원리는 책으로 배우는 게 좋아요, 직접 해봐야 직성이 풀려요"
    },
];

let chatHistory = [];
let currentConversationId = null; // 현재 Conversation ID (로그인 사용자용)
let onboardingState = {
    isComplete: false,
    step: 0,
    answers: {}
};

// -- CSRF Token Helper --
function getPostHeaders() {
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
    return {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrfToken,
    };
}


// -- Initialization --

const init = async () => {
    await loadState();
    detectReloadAndReset(); // Reset on full reload if needed, or keep persistence
    renderHistory();
    restoreResultPanel(); // Restore right panel state

    if (!onboardingState.isComplete && onboardingState.step > 0) {
        // 이미 진행 중이던 온보딩이 있다면 계속 진행
        startOnboardingStep();
    } else {
        // 처음 접속했거나 온보딩이 완료된 상태라면 일반 대화 모드로 시작
        // 단, 진행 중이 아니면 isComplete를 true로 간주하여 placeholder 등이 일반 대화용으로 나오게 함
        onboardingState.isComplete = true;

        // 온보딩 완료 상태라면 Placeholder 업데이트
        if (chatInput) chatInput.placeholder = "궁금한 점을 물어보세요!";

        // 초기 환영 메시지 (채팅 기록이 비어 있을 때만)
        if (chatHistory.length === 0) {
            try {
                const authResponse = await fetch('/api/auth/me');
                const authData = await authResponse.json();

                if (authData.is_authenticated) {
                    // Check 'has_history' from backend
                    if (authData.has_history) {
                        // [MODIFIED] New greeting for returning users
                        await appendBubbleWithTyping("다시 만나서 반갑습니다! 이전의 대화 내용은 좌측의 폴더 아이콘을 눌러 가져올 수 있습니다! 아니면 지금 저와 새로운 대화를 시작하실까요?", 'ai', false, 20);
                    } else {
                        // New User Greeting
                        await appendBubbleWithTyping("안녕하세요! 처음 뵙겠습니다. 무엇을 도와드릴까요?", 'ai', false, 20);
                    }
                }
            } catch (e) {
                console.error("Auth check in init failed:", e);
            }
        }
    }

};

const loadState = async () => {
    // 1. 로그인된 사용자 확인
    try {
        const authResponse = await fetch('/api/auth/me');
        const authData = await authResponse.json();

        if (authData.is_authenticated) {
            console.log("User is authenticated. Fetching server history...");
            // [NEW] Update character image
            if (authData.user) {
                // [NEW] Handle custom image vs character persistence
                if (authData.user.custom_image_url) {
                    console.log("Loading custom image from server:", authData.user.custom_image_url);
                    updateCharacterImage(authData.user.character, authData.user.custom_image_url);
                    localStorage.setItem('user_custom_image', authData.user.custom_image_url);
                } else {
                    // If server says no custom image (or use_custom_image is False), remove from local
                    localStorage.removeItem('user_custom_image');

                    if (authData.user.character) {
                        console.log("Loading character from server:", authData.user.character);
                        updateCharacterImage(authData.user.character);
                    }
                }

                if (authData.user.character) {
                    localStorage.setItem('user_character', authData.user.character);
                }
            } else {
                // Fallback: check local storage just in case server returned nothing (guest/error?)
                const saved = localStorage.getItem('user_character');
                const savedCustom = localStorage.getItem('user_custom_image');
                if (savedCustom) updateCharacterImage(saved, savedCustom);
                else if (saved) updateCharacterImage(saved);
            }
            await fetchHistory();
            console.log("User is authenticated. Initializing fresh chat session...");

            // 로그인 사용자는 기본적으로 새로운 대화 세션으로 시작
            // 단, 이전에 불러온 conversation id가 sessionStorage에 있으면 복원
            try {
                const storedConvId = sessionStorage.getItem(STORAGE_KEY_CONVERSATION_ID);
                if (storedConvId) {
                    currentConversationId = storedConvId;
                    console.log('Restored conversation id from sessionStorage:', currentConversationId);
                } else {
                    currentConversationId = null;
                }
            } catch (e) {
                currentConversationId = null;
            }

            // 클라이언트 세션 상태도 초기화 (온보딩은 완료 상태로 유지)
            chatHistory = [];
            onboardingState = {
                isComplete: true,  // 이미 온보딩을 완료한 사용자로 간주
                step: 0,
                answers: {}
            };
            saveState();

            return; // Skip loading from local storage if logged in
        }
    } catch (e) {
        console.error("Auth check failed:", e);
    }

    // 2. Fallback: Load Chat History from SessionStorage (Guest)
    try {
        const storedHistory = sessionStorage.getItem(STORAGE_KEY_HISTORY);
        if (storedHistory) chatHistory = JSON.parse(storedHistory);
    } catch { chatHistory = []; }

    // Load Onboarding State
    try {
        const storedOnboarding = sessionStorage.getItem(STORAGE_KEY_ONBOARDING);
        if (storedOnboarding) {
            onboardingState = JSON.parse(storedOnboarding);
        }
    } catch {
        // Default state
    }
};

const fetchHistory = async () => {
    // 로그인 사용자의 최근 대화 목록을 조회 (참고용)
    // 이 함수는 현재 대화 화면에는 영향을 주지 않고, 
    // 사용자가 이전 대화를 확인하고 싶을 때 "폴더" 버튼을 통해 호출됨
    try {
        const response = await fetch('/api/chat/history');
        if (!response.ok) throw new Error('Failed to fetch history');

        const data = await response.json();
        if (data.history && Array.isArray(data.history)) {
            // Server history format match: { role, content, ... }
            // NOTE: 이 함수는 대화 화면을 업데이트하지 않음.
            // 필요한 경우에만 폴더 버튼 등에서 호출되어 이전 대화를 복원함
            console.log("Fetched server history (for reference):", data.history);
        }
    } catch (e) {
        console.error("Error fetching history:", e);
    }
};

const saveState = () => {
    sessionStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(chatHistory));
    sessionStorage.setItem(STORAGE_KEY_ONBOARDING, JSON.stringify(onboardingState));
    try {
        if (currentConversationId) sessionStorage.setItem(STORAGE_KEY_CONVERSATION_ID, String(currentConversationId));
        else sessionStorage.removeItem(STORAGE_KEY_CONVERSATION_ID);
    } catch (e) {
        console.warn('Failed to save conversation id to sessionStorage', e);
    }
};

const detectReloadAndReset = () => {
    const navEntry = performance.getEntriesByType('navigation')[0];
    const isReload = navEntry
        ? navEntry.type === 'reload'
        : performance.navigation && performance.navigation.type === 1;

    // Optional: Clear session on reload if strictly desired. 
    // Usually users prefer persistence. Let's keep persistence for now.
    // if (isReload) { ... }
};

// -- UI Rendering --

// Helper to get avatar URL
const getAvatarUrl = (type) => {
    const customImg = (window.USER_CUSTOM_IMAGE_URL && window.USER_CUSTOM_IMAGE_URL !== '')
        ? window.USER_CUSTOM_IMAGE_URL
        : localStorage.getItem('user_custom_image');

    if (customImg) return customImg;

    const savedChar = (window.USER_CHARACTER && window.USER_CHARACTER !== 'None')
        ? window.USER_CHARACTER
        : localStorage.getItem('user_character');
    let filename = savedChar || 'rabbit';
    return `/static/images/${filename}.png`;
};

const createBubble = (content, type, isHtml = false) => {
    const container = document.createElement('div');
    container.classList.add('bubble-container', type);

    const avatar = document.createElement('img');
    avatar.classList.add('chat-avatar');
    avatar.src = getAvatarUrl(type);
    avatar.alt = type === 'ai' ? 'AI' : 'User';
    container.appendChild(avatar);

    const bubble = document.createElement('div');
    bubble.classList.add('bubble');

    if (isHtml) {
        bubble.innerHTML = content;          // ✅ 스피너
    } else {
        bubble.innerHTML = marked.parse(content); // ✅ 일반 메시지
    }

    container.appendChild(bubble);
    return container;
};


const appendBubble = (content, type, shouldPersist = true, isHtml = false) => {
    if (!chatCanvas) return null;

    const bubbleContainer = createBubble(content, type, isHtml);
    chatCanvas.appendChild(bubbleContainer);
    chatCanvas.scrollTop = chatCanvas.scrollHeight;

    if (shouldPersist) {
        const role = type === 'user' ? 'user' : 'assistant';
        chatHistory.push({ role, content });
        saveState();
    }

    return bubbleContainer.querySelector('.bubble');
};


const appendBubbleWithTyping = async (text, type, shouldPersist = true, speed = 20) => {
    if (!chatCanvas) return;

    const container = document.createElement('div');
    container.classList.add('bubble-container', type);

    const avatar = document.createElement('img');
    avatar.classList.add('chat-avatar');
    avatar.src = getAvatarUrl(type);
    container.appendChild(avatar);

    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    container.appendChild(bubble);

    chatCanvas.appendChild(container);

    const startTime = Date.now();
    let charIndex = 0;
    bubble.innerHTML = '';

    while (charIndex < text.length) {
        const elapsed = Date.now() - startTime;
        let targetCount = Math.max(1, Math.floor(elapsed / speed));

        // If tab was backgrounded, elapsed might be huge, so we catch up instantly.
        if (targetCount > text.length) targetCount = text.length;

        if (targetCount > charIndex) {
            charIndex = targetCount;
            const currentText = text.substring(0, charIndex);
            bubble.innerHTML = marked.parse(currentText, markedOptions);
            chatCanvas.scrollTop = chatCanvas.scrollHeight;
        }

        if (charIndex >= text.length) break;
        await new Promise(resolve => setTimeout(resolve, speed));
    }

    if (shouldPersist) {
        const role = type === 'user' ? 'user' : 'assistant';
        chatHistory.push({ role, content: text });
        saveState();
    }

    return container;
};

const renderHistory = () => {
    if (!chatCanvas) return;
    chatCanvas.innerHTML = ''; // Clear existing
    chatHistory.forEach(msg => {
        const type = msg.role === 'user' ? 'user' : 'ai';
        chatCanvas.appendChild(createBubble(msg.content, type));
    });
    chatCanvas.scrollTop = chatCanvas.scrollHeight;
};

// -- Onboarding Logic --

const startOnboardingStep = async () => {
    if (onboardingState.step >= ONBOARDING_QUESTIONS.length) {
        finishOnboarding();
        return;
    }

    const currentQ = ONBOARDING_QUESTIONS[onboardingState.step];
    const lastAiMsg = chatHistory.slice().reverse().find(m => m.role === 'assistant');
    if (!lastAiMsg || lastAiMsg.content !== currentQ.prompt) {
        await appendBubbleWithTyping(currentQ.prompt, 'ai', true, 15);
    }
    if (chatInput) chatInput.placeholder = currentQ.placeholder || "답변을 입력하세요...";
};

const handleOnboardingInput = async (text) => {
    appendBubble(text, 'user');
    const currentQ = ONBOARDING_QUESTIONS[onboardingState.step];
    onboardingState.answers[currentQ.key] = text;
    onboardingState.step++;
    saveState();

    if (onboardingState.step < ONBOARDING_QUESTIONS.length) {
        startOnboardingStep();
    } else {
        await finishOnboarding();
    }
};

// 스피너 생성 함수
const createSpinner = (message = '응답 생성 중...') => {
    return `
        <div class="bubble-spinner-container">
            <div class="bubble-spinner"></div>
            <span>${message}</span>
        </div>
    `;
};

const finishOnboarding = async () => {
    onboardingState.isComplete = true;
    saveState();

    const loadingBubble = appendBubble(createSpinner('추천 분석 중...'), 'ai', false, true);

    try {
        const response = await fetch(API_ONBOARDING_URL, {
            method: 'POST',
            headers: getPostHeaders(),
            body: JSON.stringify({ 
                answers: onboardingState.answers, 
                history: chatHistory,
                conversation_id: currentConversationId // [MODIFIED] Send current ID to prevent duplication
            })
        });
        if (!response.ok) throw new Error("Onboarding API failed");
        const result = await response.json();
        if (loadingBubble) loadingBubble.parentElement.remove();

        // [MODIFIED] Update conversation ID from result to enable list updates
        if (result.conversation_id) {
            currentConversationId = result.conversation_id;
            console.log("Onboarding saved. Conversation ID:", currentConversationId);
            saveState();
        }

        const recs = result.recommended_majors || [];
        let summaryText = "온보딩 답변을 바탕으로 추천 전공 TOP 5를 정리했어요:\n";
        recs.slice(0, 5).forEach((major, idx) => {
            summaryText += `${idx + 1}. ${major.major_name} (점수 ${major.score.toFixed(2)})\n`;
        });
        summaryText += "\n필요하면 위 전공 중 궁금한 학과를 지정해서 더 물어봐도 좋아요!";

        // [MODIFIED] Manually push the summary to local history so it syncs with what backend just saved
        // Backend saved it as assistant message, so we reflect it here.
        await appendBubbleWithTyping(summaryText, 'ai', true, 15);
        
        updateResultPanel(result);
    } catch (e) {
        console.error(e);
        if (loadingBubble) loadingBubble.parentElement.remove();
        await appendBubbleWithTyping("죄송합니다. 추천 정보를 불러오는데 실패했습니다.", 'ai', true, 20);
    }
    if (chatInput) chatInput.placeholder = "궁금한 점을 물어보세요!";
};

const updateResultPanel = (result) => {
    const resultCard = document.querySelector('.result-card');
    if (!resultCard) return;

    const recs = result.recommended_majors || [];
    if (recs.length === 0) {
        resultCard.innerHTML = "추천 결과가 없습니다.";
        sessionStorage.setItem(STORAGE_KEY_RESULT_PANEL, "추천 결과가 없습니다.");
        return;
    }

    let html = "<strong>추천 전공 결과:</strong><br><br>";
    recs.slice(0, 5).forEach((major, idx) => {
        html += `${idx + 1}. ${major.major_name}<br>`;
        html += `<small style="color:#666">${major.cluster || ''} - 추천 점수: ${major.score ? major.score.toFixed(2) : 'N/A'}</small><br><br>`;
    });

    resultCard.innerHTML = html;
    sessionStorage.setItem(STORAGE_KEY_RESULT_PANEL, html);
};

const restoreResultPanel = () => {
    const resultCard = document.querySelector('.result-card');
    if (!resultCard) return;

    const savedContent = sessionStorage.getItem(STORAGE_KEY_RESULT_PANEL);
    if (savedContent) {
        resultCard.innerHTML = savedContent;
        return;
    }

    const defaultHtml = `
        제가 당신에게 추천드리는 학과들로는 생명공학, 컴퓨터공학, AI융합전공, 데이터사이언스과, 소프트웨어공학과 등이 있으며 추가적으로 물리학과, 천문학 등도 고려하실 수 있습니다.
        <br><br>
        이외 더 자세한 학과정보 및 진로상담이 필요하시면 채팅창에 추가 질문을 해주세요.
    `;
    resultCard.innerHTML = defaultHtml;
    sessionStorage.setItem(STORAGE_KEY_RESULT_PANEL, defaultHtml);
};

// -- Main Chat Logic (STREAMING) --

const handleChatInput = async (text) => {
    // 1. Show user message
    appendBubble(text, 'user', true);

    // 2. Create a new AI bubble with spinner
    const aiBubble = appendBubble(createSpinner(), 'ai', false, true);
    if (!aiBubble) return;

    let finalResponse = "";
    try {
        const historyToSend = chatHistory.slice(0, -1);
        const requestBody = {
            message: text,
            history: historyToSend,
            conversation_id: currentConversationId,
        };

        const response = await fetch(API_CHAT_URL, {
            method: 'POST',
            headers: getPostHeaders(),
            body: JSON.stringify(requestBody)
        });

        // 3. Update Conversation ID from header
        const convId = response.headers.get('X-Conversation-Id');
        if (convId && !currentConversationId) {
            currentConversationId = convId;
            saveState();
            console.log('Conversation started with ID:', currentConversationId);
        }

        if (!response.ok) throw new Error(`Network error: ${response.statusText}`);
        if (!response.body) throw new Error("No response body");

        // 4. Read the stream
        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
        let buffer = '';
        let isFirstContent = true; // 첫 컨텐츠 수신 시 스피너 제거용

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += value;
            let boundary = buffer.indexOf('\n\n');

            while (boundary !== -1) {
                const chunk = buffer.substring(0, boundary);
                buffer = buffer.substring(boundary + 2);

                if (chunk.startsWith('data: ')) {
                    const jsonString = chunk.substring(6);
                    try {
                        const data = JSON.parse(jsonString);

                        if (data.type === 'status') {
                            // 상태 메시지도 스피너와 함께 표시
                            aiBubble.innerHTML = createSpinner(data.content);
                        } else if (data.type === 'delta') {
                            // 첫 델타 수신 시 스피너 제거
                            if (isFirstContent) {
                                isFirstContent = false;
                            }
                            finalResponse += data.content;
                            aiBubble.innerHTML = marked.parse(finalResponse, markedOptions);
                        } else if (data.type === 'content') {
                            finalResponse = data.content;
                            aiBubble.innerHTML = marked.parse(finalResponse);

                        } else if (data.type === 'error') {
                            finalResponse = data.content;
                            aiBubble.innerHTML = `<span style="color:red;">${data.content}</span>`;
                        }
                    } catch (e) {
                        console.error('Failed to parse JSON chunk:', jsonString, e);
                    }
                }
                boundary = buffer.indexOf('\n\n');
            }
            chatCanvas.scrollTop = chatCanvas.scrollHeight;
        }

    } catch (error) {
        console.error('Chat stream failed:', error);
        finalResponse = "오류가 발생했습니다.";
        aiBubble.innerHTML = `<span style="color:red;">${finalResponse}</span>`;
    }

    // 5. Persist the final AI response
    if (finalResponse) {
        chatHistory.push({ role: 'assistant', content: finalResponse });
        saveState();
    } else {
        aiBubble.parentElement.remove();
    }
};

// [Modified] Prevent double submission
let isProcessing = false;

const handleSubmit = async () => {
    if (!chatInput || isProcessing) return;
    const text = chatInput.value.trim();
    if (!text) return;

    // Lock
    isProcessing = true;
    chatInput.disabled = true; // Optional: disable input
    chatInput.value = '';

    const cleanedText = text.replace(/\s+/g, '');
    if (cleanedText === '추천시작' || text.includes('추천 시작')) {
        appendBubble(text, 'user');
        onboardingState = { isComplete: false, step: 0, answers: {} };
        saveState();
        await startOnboardingStep();
        
        // Unlock
        isProcessing = false;
        chatInput.disabled = false;
        chatInput.focus();
        return;
    }

    try {
        if (!onboardingState.isComplete) {
            await handleOnboardingInput(text);
        } else {
            await handleChatInput(text);
        }
    } catch (e) {
        console.error("Error in handleSubmit:", e);
    } finally {
        // Unlock
        isProcessing = false;
        chatInput.disabled = false;
        chatInput.focus();
    }
};

// -- Reset Chat Logic --

const resetChat = async () => {
    if (!confirm("대화 내용을 모두 지우고 처음부터 다시 시작하시겠습니까?")) return;

    // 1. Save chat history for logged-in users
    // [MODIFIED] Only save if we don't have a conversation ID (i.e., unsaved local history).
    // If currentConversationId exists, it means messages are already auto-saved to backend.
    if (chatHistory.length > 0 && !currentConversationId) {
        try {
            const authRes = await fetch('/api/auth/me');
            const authData = await authRes.json();
            if (authData?.is_authenticated) {
                await fetch('/api/chat/save', {
                    method: 'POST',
                    headers: getPostHeaders(),
                    body: JSON.stringify({ history: chatHistory })
                });
            }
        } catch (e) { console.error('Error saving chat history:', e); }
    }

    // 2. Clear Local State
    chatHistory = [];
    currentConversationId = null;
    onboardingState = { isComplete: true, step: 0, answers: {} };
    sessionStorage.removeItem(STORAGE_KEY_HISTORY);
    sessionStorage.removeItem(STORAGE_KEY_ONBOARDING);
    sessionStorage.removeItem(STORAGE_KEY_RESULT_PANEL);
    sessionStorage.removeItem(STORAGE_KEY_CONVERSATION_ID);
    saveState();

    // 3. UI Reset
    if (chatCanvas) chatCanvas.innerHTML = '';
    restoreResultPanel();

    // 4. Show Greeting
    await appendBubbleWithTyping("새로운 대화를 시작합니다! 무엇을 도와드릴까요?", 'ai', false, 20);

    if (chatInput) chatInput.placeholder = "궁금한 점을 물어보세요!";
};

// -- Conversation List Logic --

const showConversationList = async () => {
    const resultCard = document.querySelector('.result-card');
    if (!resultCard) return;

    try {
        const authResp = await fetch('/api/auth/me');
        const authData = await authResp.json();
        if (!authData.is_authenticated) {
            resultCard.innerHTML = '<p>게스트 사용자는 저장된 대화를 불러올 수 없습니다. 로그인 후 이용해주세요.</p>';
            return;
        }

        const resp = await fetch('/api/chat/list');
        if (!resp.ok) throw new Error('Failed to fetch conversations');
        const data = await resp.json();
        const convs = data.conversations || [];

        if (convs.length === 0) {
            resultCard.innerHTML = '<p>저장된 과거 대화가 없습니다.</p>';
            return;
        }

        const tpl = document.getElementById('conv-list-template');
        const clone = tpl.content.cloneNode(true);
        const ul = clone.querySelector('.conv-list-ul');

        convs.forEach(c => {
            const itemTpl = document.getElementById('conv-item-template');
            const li = itemTpl.content.cloneNode(true).querySelector('li');

            li.setAttribute('data-id', c.id);
            li.querySelector('.conv-title').textContent = c.title || '(제목 없음)';
            li.querySelector('.conv-meta').textContent = `${c.updated_at.split('T')[0]} · ${c.message_count} messages`;
            li.querySelector('.conv-preview').textContent = c.last_message_preview || '';

            ul.appendChild(li);
        });

        resultCard.innerHTML = '';
        resultCard.appendChild(clone);

        // 각 내역 항목 클릭 리스너
        const items = resultCard.querySelectorAll('li[data-id]');
        items.forEach(it => {
            // 항목 자체 클릭 시 대화 불러오기
            it.addEventListener('click', async (e) => {
                const convId = it.getAttribute('data-id');
                await loadConversation(convId);
            });

            // [NEW] 삭제 버튼 클릭 리스너
            const deleteBtn = it.querySelector('.conv-delete-btn');
            if (deleteBtn) {
                deleteBtn.addEventListener('click', async (e) => {
                    e.stopPropagation(); // 부모 항목 클릭 이벤트 전파 방지
                    const convId = it.getAttribute('data-id');
                    const convTitle = it.querySelector('.conv-title').textContent;

                    if (confirm(`'${convTitle}' 대화를 정말로 삭제하시겠습니까?\n삭제된 대화는 복구할 수 없습니다.`)) {
                        try {
                            const response = await fetch(`/api/chat/delete/${convId}`, {
                                method: 'DELETE',
                                headers: getPostHeaders() // CSRF 토큰 포함
                            });

                            if (response.ok) {
                                // UI에서 제거
                                it.remove();
                                alert('대화가 삭제되었습니다.');

                                // 만약 현재 열려있는 대화라면 초기화 (선택 사항)
                                if (String(currentConversationId) === String(convId)) {
                                    resetChat();
                                }
                            } else {
                                const err = await response.json();
                                alert(`삭제 실패: ${err.error || '알 수 없는 오류'}`);
                            }
                        } catch (err) {
                            console.error('삭제 요청 중 오류 발생:', err);
                            alert('오류가 발생했습니다.');
                        }
                    }
                });
            }
        });
        resultCard.querySelector('.conv-back-btn')?.addEventListener('click', (e) => { e.preventDefault(); restoreResultPanel(); });
    } catch (e) { console.error('Error loading conversation list:', e); }
};

const loadConversation = async (convId) => {
    if (!convId) return;

    // [MODIFIED] Removed save confirmation prompt. 
    // Backend automatically saves messages during chat, so we can safely switch contexts.
    // if (chatHistory.length > 0) { ... }

    try {
        const resp = await fetch(`/api/chat/load?conversation_id=${encodeURIComponent(convId)}`);
        if (!resp.ok) throw new Error('Failed to load conversation');

        const data = await resp.json();
        const conv = data.conversation;
        if (!conv) throw new Error('Invalid conversation data');

        chatHistory = conv.messages.map(m => ({ role: m.role, content: m.content }));
        currentConversationId = conv.id;
        onboardingState.isComplete = true;
        saveState();
        renderHistory(); // UI 업데이트

    } catch (e) {
        console.error('Error loading conversation:', e);
        alert('세션 불러오기에 실패했습니다.');
    }
};

const summarizeConversation = async () => {
    if (chatHistory.length === 0) {
        alert('요약할 대화 내역이 없습니다.');
        return;
    }
    const resultCard = document.querySelector('.result-card');
    try {
        if (resultCard) {
            const loadingTpl = document.getElementById('summary-loading-template');
            resultCard.innerHTML = '';
            resultCard.appendChild(loadingTpl.content.cloneNode(true));
        }

        const resp = await fetch('/api/chat/summarize', {
            method: 'POST',
            headers: getPostHeaders(),
            body: JSON.stringify({ history: chatHistory })
        });
        if (!resp.ok) throw new Error('Failed to summarize conversation');
        const data = await resp.json();

        if (resultCard) {
            const resultTpl = document.getElementById('summary-result-template');
            const resultContent = resultTpl.content.cloneNode(true);

            const contentDiv = resultContent.querySelector('.result-content');
            contentDiv.innerHTML = data.summary.replace(/\n/g, '<br>');

            resultCard.innerHTML = '';
            resultCard.appendChild(resultContent);
            sessionStorage.setItem(STORAGE_KEY_RESULT_PANEL, resultCard.innerHTML);
        }
    } catch (e) {
        console.error('Error summarizing conversation:', e);
        alert('요약에 실패했습니다.');
    }
};

// -- Event Listeners --
document.querySelector('.action-btn[aria-label="새 채팅"]')?.addEventListener('click', resetChat);
document.querySelector('.action-btn[aria-label="폴더"]')?.addEventListener('click', showConversationList);
document.querySelector('.action-btn[aria-label="정보"]')?.addEventListener('click', summarizeConversation);

if (sendBtn) sendBtn.addEventListener('click', handleSubmit);
if (chatInput) {
    chatInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.isComposing) {
            event.preventDefault();
            handleSubmit();
        }
    });
}

// Start
function updateCharacterImage(characterId, customImageUrl = null) {
    console.log("updateCharacterImage called with:", characterId, customImageUrl);
    const imgEl = document.querySelector('.result-rabbit');
    if (!imgEl) return;

    if (customImageUrl) {
        imgEl.src = customImageUrl;
        imgEl.alt = 'User Custom Character';
        imgEl.style.borderRadius = '20px';
        return;
    }

    let filename = characterId || 'rabbit';

    imgEl.src = `/static/images/${filename}.png`;
    imgEl.alt = `${Object.keys({
        'rabbit': '토끼', 'bear': '곰', 'fox': '여우', 'hedgehog': '고슴도치', 'koala': '코알라',
        'otter': '수달', 'penguin': '펭귄', 'raccoon': '너구리', 'sloth': '나무늘보', 'turtle': '거북이'
    }).find(key => key === characterId) || 'User Character'}`;
}