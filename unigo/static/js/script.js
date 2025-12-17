// 메인 콘텐츠 로그인/회원가입 버튼
const signInBtn = document.getElementById("btn-signin");
const logInBtn = document.getElementById("btn-login");
const signUpBtn = document.getElementById("btn-signup");

// 로그인 팝업
const modalLogin = document.getElementById("login-modal");
const closeBtnLogin = document.getElementById("btn-login-close");
const overlayLogIn = document.getElementById("login-overlay");

const linkForgotPassword = document.getElementById("link-forgot-password");
const linkSignup = document.getElementById("link-signup");

if (signInBtn) {
    signInBtn.addEventListener("click", () => {
        modalLogin.classList.add("show");
    });
}

const loginBtns = document.querySelectorAll(".btn-login");

loginBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        modalLogin.classList.add("show");
    });
});


closeBtnLogin.addEventListener("click", () => modalLogin.classList.remove("show"));
overlayLogIn.addEventListener("click", () => modalLogin.classList.remove("show"));


const loginConfirmBtn = document.getElementById("btn-login-confirm");

if (loginConfirmBtn) {
    loginConfirmBtn.addEventListener("click", () => {
        const emailOrId = document.getElementById("login-email").value;
        const password = document.getElementById("login-password").value;

        if (!emailOrId || !password) {
            alert("이메일/아이디와 비밀번호를 입력해주세요.");
            return;
        }

        fetch("/api/auth/login", {
            method: "POST",
            headers: getPostHeaders(),
            body: JSON.stringify({
                username: emailOrId,
                password: password
            })
        })
            .then(response => response.json())
            .then(data => {
                if (data.user) {
                    // 로그인 성공
                    window.location.href = "/chat/";
                } else {
                    alert(data.error || "로그인 실패");
                }
            })
            .catch(error => {
                console.error("Error:", error);
                alert("서버 오류가 발생했습니다.");
            });
    });
}


// 회원가입 팝업
const modalSignup = document.getElementById("signup-modal");
const closeBtnSignup = document.getElementById("btn-signup-close");
const overlaySignup = document.getElementById("signup-overlay");

const emailConfirmBtn = document.getElementById("btn-email-confirm"); // 이메일 인증 전송 버튼
const signUpConfirmBtn = document.getElementById("btn-signup-confirm");

const inputs = document.querySelectorAll("#signup-modal input[required]");
const passwordInput = document.getElementById("signup-password");
const passwordCheckInput = document.getElementById("signup-password-check");

if (linkSignup) {
    linkSignup.addEventListener("click", () => {
        modalSignup.classList.add("show");
        modalLogin.classList.remove("show"); // 로그인 창 닫기
    });
}

if (signUpBtn) {
    signUpBtn.addEventListener("click", () => {
        modalSignup.classList.add("show");
    });
}

closeBtnSignup.addEventListener("click", () => modalSignup.classList.remove("show"));
overlaySignup.addEventListener("click", () => modalSignup.classList.remove("show"));


// 회원가입 종료 팝업
const modalSignupComplete = document.getElementById("signup-complete-modal");
const closeBtnSignupComplete = document.getElementById("btn-signup-complete-close");
const overlaySignupComplete = document.getElementById("signup-complete-overlay");

if (signUpConfirmBtn) {
    signUpConfirmBtn.addEventListener("click", () => {
        // 모든 input 값이 채워져 있는지 확인
        const allFilled = Array.from(inputs).every(input => input.value.trim() !== "");

        if (!allFilled) {
            alert("모든 필수 항목을 입력해주세요.");
            return;
        }

        // 비밀번호와 비밀번호 확인이 같은지 확인
        if (passwordInput.value !== passwordCheckInput.value) {
            alert("비밀번호와 비밀번호 확인이 일치하지 않습니다.");
            return;
        }

        const username = document.getElementById("signup-username").value;
        const email = document.getElementById("signup-email").value;
        const password = passwordInput.value;

        // 회원가입 API 호출
        fetch("/api/auth/signup", {
            method: "POST",
            headers: getPostHeaders(),
            body: JSON.stringify({
                username: username,
                email: email,
                password: password
            })
        })
            .then(response => response.json())
            .then(data => {
                if (data.user) {
                    // 회원가입 성공 -> 완료 모달 띄우기
                    modalSignup.classList.remove("show");
                    modalSignupComplete.classList.add("show");

                    // 완료 모달 닫으면 채팅 페이지로 이동 (자동 로그인 됨)
                    closeBtnSignupComplete.onclick = () => {
                        window.location.href = "/chat/";
                    };
                    overlaySignupComplete.onclick = () => {
                        window.location.href = "/chat/";
                    };
                } else {
                    alert(data.error || "회원가입 실패");
                }
            })
            .catch(error => {
                console.error("Error:", error);
                alert("서버 오류가 발생했습니다.");
            });
    });
}

closeBtnSignupComplete.addEventListener("click", () => modalSignupComplete.classList.remove("show"));
overlaySignupComplete.addEventListener("click", () => modalSignupComplete.classList.remove("show"));

// 비밀번호 찾기 팝업
const modalFindPassword = document.getElementById("find-password-modal");
const closeBtnFindPassword = document.getElementById("btn-find-password-close");
const overlayFindPassword = document.getElementById("find-password-overlay");

const tempPasswordBtn = document.getElementById("btn-temp-password"); // 임시 비밀번호 받기 버튼
const returnLoginBtn = document.getElementById("btn-return-login");

if (linkForgotPassword) {
    linkForgotPassword.addEventListener("click", () => {
        modalFindPassword.classList.add("show");
    });
}

closeBtnFindPassword.addEventListener("click", () => modalFindPassword.classList.remove("show"));
overlayFindPassword.addEventListener("click", () => modalFindPassword.classList.remove("show"));
returnLoginBtn.addEventListener("click", () => modalFindPassword.classList.remove("show"));


// ==========================================
// [REMOVED] Hero Image Sync for Auth Page
// (User requested to always show Rabbit on login page)
// ==========================================
// ==========================================
// [FIX] Hero Image Force Rabbit
// Always show Rabbit on Auth page, ignoring localStorage
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
    const heroImg = document.querySelector(".hero-img");
    if (heroImg) {
        // Force reset to Rabbit in case some other script tried to change it
        heroImg.src = "/static/images/rabbit.png";
    }
});

// [REMOVED] deleteAccountBtn from global scope. Logic is in setting.js

// ==========================================
// [MOVED] Duplicate Checks (Email/Username) from base.html
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    const showWarning = (inputElement, message) => {
        // Use native browser validation bubble
        inputElement.setCustomValidity(message);
        inputElement.reportValidity();
    };

    const showSuccess = (inputElement, message) => {
        inputElement.setCustomValidity(""); // Clear error
        alert(message);
    };

    // Clear custom validity on input so user can try again
    const clearValidity = (e) => {
        e.target.setCustomValidity("");
    };

    // Email Check
    const btnEmailConfirm = document.getElementById('btn-email-confirm');
    const emailInput = document.getElementById('signup-email');
    if (btnEmailConfirm && emailInput) {
        emailInput.addEventListener('input', clearValidity);

        btnEmailConfirm.addEventListener('click', async () => {
            const email = emailInput.value.trim();
            if (!email) {
                showWarning(emailInput, "이메일을 입력해주세요.");
                return;
            }
            try {
                const res = await fetch('/api/auth/check-email', {
                    method: 'POST',
                    headers: getPostHeaders(),
                    body: JSON.stringify({ email })
                });
                const data = await res.json();
                if (data.exists) {
                    showWarning(emailInput, data.message);
                } else {
                    showSuccess(emailInput, data.message);
                }
            } catch (e) {
                console.error(e);
            }
        });
    }

    // Username Check
    const btnUserConfirm = document.getElementById('btn-username-confirm');
    const userInput = document.getElementById('signup-username');
    if (btnUserConfirm && userInput) {
        userInput.addEventListener('input', clearValidity);

        btnUserConfirm.addEventListener('click', async () => {
            const username = userInput.value.trim();
            if (!username) {
                showWarning(userInput, "닉네임을 입력해주세요.");
                return;
            }
            try {
                const res = await fetch('/api/auth/check-username', {
                    method: 'POST',
                    headers: getPostHeaders(),
                    body: JSON.stringify({ username })
                });
                const data = await res.json();
                if (data.exists) {
                    showWarning(userInput, data.message);
                } else {
                    showSuccess(userInput, data.message);
                }
            } catch (e) {
                console.error(e);
            }
        });
    }
});
