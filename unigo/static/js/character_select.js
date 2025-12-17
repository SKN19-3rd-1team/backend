/*
 * Extracted from unigo/templates/unigo_app/character_select.html
 */
document.addEventListener("DOMContentLoaded", () => {
    // ==========================================
    // Character Carousel Logic (Circular Queue Style)
    // ==========================================
    const characters = [
        {
            name: "토끼",
            id: "rabbit",
            image: "/static/images/rabbit.png",
            description:
                "토끼와 함께 활발하고 에너지 있게 오늘의 공부를 시작해봐요.",
        },
        {
            name: "곰",
            id: "bear",
            image: "/static/images/bear.png",
            description:
                "곰과 함께 든든한 마음으로 차분하게 목표를 향해 나아가봐요.",
        },
        {
            name: "여우",
            id: "fox",
            image: "/static/images/fox.png",
            description: "여우와 함께 영리하고 똑똑하게 오늘의 미션을 해결해봐요.",
        },
        {
            name: "고슴도치",
            id: "hedgehog",
            image: "/static/images/hedgehog.png",
            description:
                "고슴도치와 함께 조심스럽지만 꾸준하게 한 걸음씩 나아가봐요.",
        },
        {
            name: "코알라",
            id: "koala",
            image: "/static/images/koala.png",
            description: "코알라와 함께 여유롭고 안정감 있게 공부 시간을 채워봐요.",
        },
        {
            name: "수달",
            id: "otter",
            image: "/static/images/otter.png",
            description: "수달과 함께 즐겁고 가볍게 학습 루틴을 만들어봐요.",
        },
        {
            name: "펭귄",
            id: "penguin",
            image: "/static/images/penguin.png",
            description: "펭귄과 함께 꾸준하게, 한 걸음씩 오늘의 할 일을 채워봐요.",
        },
        {
            name: "너구리",
            id: "raccoon",
            image: "/static/images/raccoon.png",
            description: "너구리와 함께 호기심을 가지고 새로운 공부를 시도해봐요.",
        },
        {
            name: "나무늘보",
            id: "sloth",
            image: "/static/images/sloth.png",
            description:
                "나무늘보와 함께 너무 서두르지 말고, 나만의 속도로 공부를 이어가봐요.",
        },
        {
            name: "거북이",
            id: "turtle",
            image: "/static/images/turtle.png",
            description: "거북이와 함께 긴 호흡으로 차근차근 공부 습관을 쌓아봐요.",
        },
    ];

    // "currentRotationIndex" allows unbounded values (negative or > length)
    // for continuous rotation without rewinding.
    let currentRotationIndex = 0;
    const totalItems = characters.length;
    const theta = 360 / totalItems;
    const radius = 400;

    initCarousel();

    function initCarousel() {
        const carousel = document.getElementById("carousel3d");
        carousel.innerHTML = "";

        characters.forEach((character, index) => {
            const item = document.createElement("div");
            item.className = `carousel-item ${index === 0 ? "active" : ""}`;
            item.innerHTML = `<img src="${character.image}" alt="${character.name}" class="character-image">`;

            const angle = theta * index;
            item.style.transform = `rotateY(${angle}deg) translateZ(${radius}px)`;

            item.addEventListener("click", () => {
                // Click logic: find shortest path from current wrapped index
                const currentWrapped = getWrappedIndex();
                let diff = index - currentWrapped;

                // Shortest path adjustment
                if (diff > totalItems / 2) diff -= totalItems;
                else if (diff < -totalItems / 2) diff += totalItems;

                rotateCarousel(diff);
            });

            carousel.appendChild(item);
        });

        loadUserCharacter();
    }

    // Helper to get simple 0..N-1 index
    function getWrappedIndex() {
        return ((currentRotationIndex % totalItems) + totalItems) % totalItems;
    }

    async function loadUserCharacter() {
        try {
            const response = await fetch("/api/auth/me");
            const data = await response.json();

            if (data.user && data.user.character) {
                const charId = data.user.character;
                const idx = characters.findIndex((c) => c.id === charId);
                if (idx !== -1) {
                    currentRotationIndex = idx; // Set initial state
                    rotateCarousel(0); // Update view
                }
            }
        } catch (error) {
            console.error("Failed to load character:", error);
        }
        updateCharacterInfo();
    }

    function rotateCarousel(direction) {
        // Simply add direction to the unbounded counter
        currentRotationIndex += direction;

        const carousel = document.getElementById("carousel3d");
        const angle = -theta * currentRotationIndex;
        carousel.style.transform = `rotateY(${angle}deg)`;

        // Update UI based on wrapped index
        const wrappedIndex = getWrappedIndex();

        const items = document.querySelectorAll(".carousel-item");
        items.forEach((item, index) => {
            item.classList.toggle("active", index === wrappedIndex);
        });

        updateCharacterInfo();
    }

    function updateCharacterInfo() {
        const wrappedIndex = getWrappedIndex();
        const character = characters[wrappedIndex];
        document.getElementById("characterName").textContent = character.name;
        document.getElementById("characterDesc").textContent =
            character.description;
    }

    document
        .getElementById("prevBtn")
        .addEventListener("click", () => rotateCarousel(-1));
    document
        .getElementById("nextBtn")
        .addEventListener("click", () => rotateCarousel(1));

    // Note: onclick handler for cancelBtn is handled in the template or should be added here if no longer inline
    // In previous step we added an event listener for cancelBtn. We should keep it.
    const cancelBtn = document.getElementById("cancelBtn");
    if (cancelBtn) {
        cancelBtn.addEventListener("click", () => {
            // Need to handle the URL. Since this is a static file, we can't use {% url %}.
            // We usually pass the URL via data attribute or just hardcode if it's simple.
            // But wait, the original code had: window.location.href = "{% url 'unigo_app:setting' %}";
            // We have a problem here. Static JS cannot render Django template tags.

            // Strategy: Read data-url from the button.
            // I will update the HTML to include data-url attribute.
            const url = cancelBtn.dataset.url;
            if (url) window.location.href = url;
            else window.history.back(); // Fallback
        });
    }

    document.getElementById("saveBtn").addEventListener("click", async () => {
        const wrappedIndex = getWrappedIndex();
        const character = characters[wrappedIndex];

        try {
            const response = await fetch("/api/setting/update-character", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-CSRFToken": getCookie("csrftoken")
                },
                body: JSON.stringify({ character: character.id }),
            });

            if (response.ok) {
                alert(`${character.name} 캐릭터가 선택되었습니다.`);
                localStorage.setItem("user_character", character.id);
                // Redirect back to settings
                const saveBtn = document.getElementById("saveBtn");
                const nextUrl = saveBtn ? saveBtn.dataset.nextUrl : "/setting/";
                window.location.href = nextUrl;
            } else {
                alert("저장에 실패했습니다.");
            }
        } catch (error) {
            console.error("Error saving character:", error);
            alert("오류가 발생했습니다.");
        }
    });
});
