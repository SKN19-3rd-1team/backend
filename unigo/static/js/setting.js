document.addEventListener('DOMContentLoaded', () => {

    // Tab Switching
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {

        tab.addEventListener('click', () => {

            // Remove active class from all
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            // Add active to clicked

            tab.classList.add('active');
            const target = tab.dataset.tab;
            document.getElementById(`tab-${target}`).classList.add('active');

            // Reset forms when switching tabs? (Optional)
            resetForms();
        });
    });

    function resetForms() {
        document.getElementById('form-nickname').reset();
        document.getElementById('form-password').reset();
        document.getElementById('form-dangerzone').reset();
        document.querySelectorAll('.error-msg').forEach(el => el.style.display = 'none');
        document.querySelectorAll('.success-msg').forEach(el => el.style.display = 'none');

        isNicknameChecked = false;

    }


    // ==========================================
    // Nickname Change Logic
    // ==========================================

    let isNicknameChecked = false;
    document.getElementById('btn-check-duplicate').addEventListener('click', async () => {
        const username = document.getElementById('nickname-input').value;
        const errorEl = document.getElementById('nickname-error');
        const successEl = document.getElementById('nickname-success');

        errorEl.style.display = 'none';
        successEl.style.display = 'none';
        isNicknameChecked = false;

        if (!username) {
            errorEl.textContent = '닉네임을 입력해주세요.';
            errorEl.style.display = 'block';
            return;
        }

        try {
            const response = await fetch('/api/setting/check-username', {
                method: 'POST',
                headers: getPostHeaders(),
                body: JSON.stringify({ username })
            });

            const data = await response.json();

            if (data.exists) {
                errorEl.textContent = data.message;
                errorEl.style.display = 'block';

            } else {
                successEl.textContent = data.message;
                successEl.style.display = 'block';
                isNicknameChecked = true;
            }

        } catch (error) {
            console.error('Error:', error);
            alert('중복 확인 중 오류가 발생했습니다.');
        }
    });

    document.getElementById('btn-save-nickname').addEventListener('click', async () => {
        const newUsername = document.getElementById('nickname-input').value;
        const password = document.getElementById('nickname-password').value;

        if (!isNicknameChecked) {
            alert('닉네임 중복 확인을 해주세요.');
            return;
        }

        if (!password) {
            alert('비밀번호를 입력해주세요.');
            return;
        }

        try {
            const response = await fetch('/api/setting/change-nickname', {
                method: 'POST',
                headers: getPostHeaders(),
                body: JSON.stringify({ username: newUsername, password })
            });

            const data = await response.json();

            if (response.ok) {
                alert('내용이 변경되었습니다.');
                location.reload(); // Reload to refresh user state

            } else {
                alert(data.error || '변경 실패');
            }

        } catch (error) {
            console.error('Error:', error);
            alert('오류가 발생했습니다.');
        }
    });


    // ==========================================
    // Password Change Logic
    // ==========================================
    document.getElementById('btn-save-password').addEventListener('click', async () => {
        const currentPassword = document.getElementById('current-password').value;
        const newPassword = document.getElementById('new-password').value;
        const confirmPassword = document.getElementById('new-password-confirm').value;
        const matchError = document.getElementById('password-match-error');

        matchError.style.display = 'none';

        if (!currentPassword || !newPassword || !confirmPassword) {
            alert('모든 필드를 입력해주세요.');
            return;
        }

        if (newPassword !== confirmPassword) {
            matchError.style.display = 'block';
            return;
        }

        try {
            const response = await fetch('/api/setting/change-password', {
                method: 'POST',
                headers: getPostHeaders(),
                body: JSON.stringify({
                    current_password: currentPassword,
                    new_password: newPassword
                })
            });

            const data = await response.json();

            if (response.ok) {
                alert('내용이 변경되었습니다.');
                location.reload(); // Reload

            } else {
                alert(data.error || '변경 실패');
            }

        } catch (error) {
            console.error('Error:', error);
            alert('오류가 발생했습니다.');
        }
    });


    // ==========================================
    // Account Deletion Logic
    // ==========================================

    document.getElementById('btn-delete-account').addEventListener('click', async () => {
        const confirmText = document.getElementById('delete-confirm').value;
        const password = document.getElementById('delete-password').value;
        const username = document.querySelector('.sidebar-username').textContent;
        const confirmError = document.getElementById('delete-confirm-error');
        const passwordError = document.getElementById('delete-password-error');

        confirmError.style.display = 'none';
        passwordError.style.display = 'none';

        if (confirmText !== '삭제하겠습니다') {
            confirmError.textContent = "'삭제하겠습니다'를 정확히 입력해주세요.";
            confirmError.style.display = 'block';
            return;
        }

        if (!password) {
            passwordError.textContent = '본인 확인을 위해 비밀번호를 입력해주세요.';
            passwordError.style.display = 'block';
            return;
        }

        try {
            const response = await fetch('/api/setting/delete', {
                method: 'POST',
                headers: getPostHeaders(),
                body: JSON.stringify({
                    username: username,
                    password: password
                })
            });

            const data = await response.json();

            if (response.ok) {
                alert('계정이 성공적으로 삭제되었습니다.');
                window.location.href = '/auth'; // Redirect to login page

            } else {
                alert(data.error || '계정 삭제에 실패했습니다.');
            }

        } catch (error) {
            console.error('Error:', error);
            alert('계정 삭제 중 오류가 발생했습니다.');
        }
    });

    // ==========================================
    // Custom Image Upload Logic
    // ==========================================
    const uploadInput = document.getElementById('custom-image-upload');
    if (uploadInput) {
        uploadInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('image', file);

            try {
                // For FormData, we do NOT set Content-Type header (browser sets it with boundary).
                // We ONLY set the CSRF token.
                const headers = {
                    'X-CSRFToken': getCookie('csrftoken')
                };

                const response = await fetch('/api/setting/upload-character-image', {
                    method: 'POST',
                    headers: headers,
                    body: formData
                });
                const data = await response.json();

                if (response.ok) {
                    alert('이미지가 변경되었습니다.');
                    location.reload();
                } else {
                    alert(data.error || '업로드 실패');
                }
            } catch (err) {
                console.error(err);
                alert('업로드 중 오류 발생');
            }
        });
    }
});
