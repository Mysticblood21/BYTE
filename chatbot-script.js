// Global state
let currentFile = null;
let currentFilePreviewUrl = null;
let isRecording = false;
let recognition = null;
let canvas = null;
let ctx = null;
let currentDrawMode = 'draw'; // 'draw' or 'erase'
let brushSize = 3;
let currentChatId = null;
let currentUser = null;
let chats = [];
let chatToDelete = null;
let chatToRename = null;
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Init
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() { init(); });
} else {
    // DOM already ready
    init();
}

async function init() {
    await checkAuthentication();
    setupEventListeners();
    setupCanvas();
    setupSpeechRecognition();
    await loadChatHistory();

    const sidebar = document.getElementById('sidebar');
    if (sidebar) {
        sidebar.classList.add('collapsed');
        const toggleBtn = document.getElementById('sidebarToggle');
        if (toggleBtn) toggleBtn.textContent = '☰';
    }
    // Wire header buttons defensively in case inline handlers aren't bound
    wireHeaderButtons();
    showWelcomeScreen();
}

function wireHeaderButtons() {
    try {
        const clearBtn = document.getElementById('btnClearChat') || document.querySelector('.chat-actions button[title="Clear Chat"]');
        const clockBtn = document.getElementById('btnClock') || document.querySelector('.chat-actions button[title="Clock"]');
        const calcBtn = document.getElementById('btnCalculator') || document.querySelector('.chat-actions button[title="Calculator"]');
        const todoBtn = document.getElementById('btnTodo') || document.querySelector('.chat-actions button[title="To-Do List"]');
        if (clearBtn) clearBtn.addEventListener('click', clearCurrentChat);
        if (clockBtn) clockBtn.addEventListener('click', openClock);
        if (calcBtn) calcBtn.addEventListener('click', openCalculator);
        if (todoBtn) todoBtn.addEventListener('click', openTodoPage);
        const fileBtn = document.querySelector('.action-btn[title="Upload Image"]');
        if (fileBtn) fileBtn.addEventListener('click', toggleFileUpload);
        const voiceBtn = document.getElementById('voiceBtn');
        if (voiceBtn) voiceBtn.addEventListener('click', toggleVoiceInput);
        const drawBtn = document.querySelector('.action-btn[title="Draw"]');
        if (drawBtn) drawBtn.addEventListener('click', openDrawingPad);
    } catch (e) {
        console.warn('Header wiring error', e);
    }
}

// Auth
async function checkAuthentication() {
    try {
        const response = await fetch('/api/auth/me', { credentials: 'include' });
        if (response.ok) {
            const data = await response.json();
            currentUser = data.user;
            updateUserInterface();
        } else {
            window.location.href = 'auth.html';
        }
    } catch (error) {
        showNotification('Authentication failed. Please login again.', 'error');
        setTimeout(() => window.location.href = 'auth.html', 1000);
    }
}

function updateUserInterface() {
    if (!currentUser) return;
    const userInitial = document.getElementById('userInitial');
    const userName = document.getElementById('userName');
    const userEmail = document.getElementById('userEmail');
    if (userInitial) userInitial.textContent = currentUser.name.charAt(0).toUpperCase();
    if (userName) userName.textContent = currentUser.name;
    if (userEmail) userEmail.textContent = currentUser.email;
}

async function logout() {
    try {
        await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' });
    } catch (e) {}
    currentChatId = null;
    currentUser = null;
    chats = [];
    cleanupFilePreview();
    showNotification('Logged out successfully!', 'success');
    setTimeout(() => window.location.href = 'index.html', 500);
}

// Chats
async function loadChatHistory() {
    try {
        const response = await fetch('/api/chats', { credentials: 'include' });
        if (response.ok) {
            const data = await response.json();
            chats = data.chats;
            updateChatList();
        } else {
            showNotification('Failed to load chat history', 'error');
        }
    } catch (e) {
        showNotification('Error loading chat history', 'error');
    }
}

function updateChatList() {
    const chatList = document.getElementById('chatList');
    if (!chatList) return;
    chatList.innerHTML = '';
    if (chats.length === 0) {
        chatList.innerHTML = '<div class="no-chats">No chats yet. Start a new conversation!</div>';
        return;
    }
    chats.forEach(chat => {
        const item = document.createElement('div');
        item.className = `chat-item ${currentChatId === chat.id ? 'active' : ''}`;
        item.onclick = () => loadChat(chat.id);
        item.innerHTML = `
            <div class="chat-item-content">
                <div class="chat-title" title="${chat.title || `Chat ${chat.id}`}">${truncateText(chat.title || `Chat ${chat.id}`, 25)}</div>
                <div class="chat-preview">${truncateText(chat.preview || 'No messages yet', 40)}</div>
                <div class="chat-date">${formatDate(chat.updated_at)}</div>
            </div>
            <div class="chat-item-actions">
                <button class="chat-action-btn" onclick="event.stopPropagation(); startRenameChat(${chat.id}, '${escapeString(chat.title || `Chat ${chat.id}`)}')" title="Rename Chat">✏️</button>
                <button class="chat-action-btn" onclick="event.stopPropagation(); deleteChat(${chat.id})" title="Delete Chat">🗑️</button>
            </div>`;
        chatList.appendChild(item);
    });
}

async function createNewChat() {
    currentChatId = null;
    cleanupFilePreview();
    showWelcomeScreen();
    updateChatList();
    showNotification('New chat started!', 'success');
}

async function loadChat(chatId) {
    try {
        const res = await fetch(`/api/chats/${chatId}`, { credentials: 'include' });
        if (!res.ok) throw new Error('Failed');
        const data = await res.json();
        currentChatId = chatId;
        displayChatMessages(data.messages);
        updateChatList();
    } catch (e) {
        showNotification('Failed to load chat', 'error');
    }
}

function displayChatMessages(messages) {
    const chatMessages = document.getElementById('chatMessages');
    const welcomeSection = document.getElementById('welcomeSection');
    if (messages.length > 0) {
        hideWelcomeSection();
        if (chatMessages) {
            chatMessages.innerHTML = '';
            messages.forEach(m => {
                if (m.message_type === 'image') addImageMessageToUI(m.sender, m.content); else addMessageToUI(m.sender, m.content);
            });
            scrollToBottom();
        }
    } else {
        showWelcomeScreen();
    }
}

function showWelcomeScreen() {
    const welcomeSection = document.getElementById('welcomeSection');
    const chatMessages = document.getElementById('chatMessages');
    if (welcomeSection) welcomeSection.style.display = 'flex';
    if (chatMessages) { chatMessages.innerHTML = ''; chatMessages.style.display = 'none'; }
}

async function clearCurrentChat() {
    if (!currentChatId) { showWelcomeScreen(); showNotification('Chat cleared!', 'success'); return; }
    try {
        const res = await fetch(`/api/chats/${currentChatId}/clear`, { method: 'POST', credentials: 'include' });
        if (!res.ok) throw new Error();
        showWelcomeScreen();
        await loadChatHistory();
        showNotification('Chat cleared!', 'success');
    } catch (e) { showNotification('Failed to clear chat', 'error'); }
}

function deleteChat(chatId) {
    chatToDelete = chatId;
    const modal = document.getElementById('deleteModal');
    if (modal) modal.style.display = 'flex';
}

function closeDeleteModal() {
    chatToDelete = null;
    const modal = document.getElementById('deleteModal');
    if (modal) modal.style.display = 'none';
}

async function confirmDeleteChat() {
    if (!chatToDelete) return;
    try {
        const res = await fetch(`/api/chats/${chatToDelete}`, { method: 'DELETE', credentials: 'include' });
        if (!res.ok) throw new Error();
        if (chatToDelete === currentChatId) { currentChatId = null; showWelcomeScreen(); }
        await loadChatHistory();
        showNotification('Chat deleted successfully!', 'success');
    } catch (e) { showNotification('Failed to delete chat', 'error'); }
    closeDeleteModal();
}

function startRenameChat(chatId, currentTitle) {
    chatToRename = chatId;
    const chatItems = document.querySelectorAll('.chat-item');
    chatItems.forEach(item => {
        const titleElement = item.querySelector('.chat-title');
        if (titleElement && titleElement.textContent.includes(currentTitle.substring(0, 20))) {
            const input = document.createElement('input');
            input.type = 'text';
            input.value = currentTitle;
            input.className = 'chat-rename-input';
            input.style.cssText = 'width:100%;background:#e8e8e8;border:1px solid #999;border-radius:4px;padding:4px 8px;font-size:0.9rem;font-weight:600;color:#333;';
            input.onblur = () => finishRenameChat(chatId, input.value);
            input.onkeydown = (e) => { if (e.key === 'Enter') finishRenameChat(chatId, input.value); else if (e.key === 'Escape') cancelRenameChat(); };
            titleElement.innerHTML = '';
            titleElement.appendChild(input);
            input.focus();
            input.select();
        }
    });
}

async function finishRenameChat(chatId, newTitle) {
    if (!newTitle.trim()) { cancelRenameChat(); return; }
    try {
        const res = await fetch(`/api/chats/${chatId}/rename`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, credentials: 'include', body: JSON.stringify({ title: newTitle.trim() }) });
        if (!res.ok) {
            const err = await res.json();
            showNotification(err.error || 'Failed to rename chat', 'error');
        }
        await loadChatHistory();
        showNotification('Chat renamed successfully!', 'success');
    } catch (e) { showNotification('Failed to rename chat', 'error'); }
    chatToRename = null;
}

function cancelRenameChat() { chatToRename = null; loadChatHistory(); }
function escapeString(str) { return str.replace(/'/g, "\\'").replace(/"/g, '\\"'); }

// Messaging
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput?.value.trim();
    if (!message && !currentFile) return;
    hideWelcomeSection();
    const formData = new FormData();
    if (currentFile) {
        formData.append('image', currentFile);
        if (currentChatId) formData.append('chat_id', currentChatId);
        if (message) { formData.append('query', message); addMessageToUI('user', message); }
        addImageMessageToUI('user', currentFilePreviewUrl);
        if (messageInput) { messageInput.value = ''; autoResize(messageInput); }
        removeFile();
        showTypingIndicator();
        try {
            const res = await fetch('/api/recognize_handwriting', { method: 'POST', credentials: 'include', body: formData });
            hideTypingIndicator();
            if (!res.ok) throw await res.json();
            const data = await res.json();
            currentChatId = data.chat_id;
            addMessageToUI('bot', data.response);
            await loadChatHistory();
        } catch (e) {
            hideTypingIndicator();
            addMessageToUI('bot', 'Sorry, I had trouble processing your image. Please try again with a clearer image.');
        }
    } else {
        formData.append('query', message);
        if (currentChatId) formData.append('chat_id', currentChatId);
        addMessageToUI('user', message);
        if (messageInput) { messageInput.value = ''; autoResize(messageInput); }
        showTypingIndicator();
        try {
            const res = await fetch('/api/ask', { method: 'POST', credentials: 'include', body: formData });
            hideTypingIndicator();
            if (!res.ok) throw await res.json();
            const data = await res.json();
            currentChatId = data.chat_id;
            addMessageToUI('bot', data.response);
            await loadChatHistory();
        } catch (e) {
            hideTypingIndicator();
            addMessageToUI('bot', 'Sorry, I had trouble processing your message. Please try again.');
        }
    }
}

// Sidebar
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('sidebarToggle');
    if (!sidebar) return;
    sidebar.classList.toggle('collapsed');
    if (toggleBtn) toggleBtn.textContent = sidebar.classList.contains('collapsed') ? '☰' : '✕';
}

// Utils
function truncateText(text, maxLength) { return text.length <= maxLength ? text : text.substr(0, maxLength - 3) + '...'; }
function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
}

function setupEventListeners() {
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.addEventListener('keydown', handleKeyPress);
        messageInput.addEventListener('input', () => autoResize(messageInput));
    }
    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.addEventListener('change', handleFileSelect);
    document.addEventListener('click', function(e) {
        const drawingModal = document.getElementById('drawingModal');
        const deleteModal = document.getElementById('deleteModal');
        if (drawingModal && e.target === drawingModal) closeDrawingPad();
        if (deleteModal && e.target === deleteModal) closeDeleteModal();
    });
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const drawingModal = document.getElementById('drawingModal');
            const deleteModal = document.getElementById('deleteModal');
            if (drawingModal && drawingModal.classList.contains('active')) closeDrawingPad();
            else if (deleteModal && deleteModal.style.display === 'flex') closeDeleteModal();
            else if (chatToRename) cancelRenameChat();
        }
        if (e.ctrlKey && e.key === 'n') { e.preventDefault(); createNewChat(); }
        if (e.ctrlKey && e.key === '/') { e.preventDefault(); toggleSidebar(); }
    });
}

function handleKeyPress(event) { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); } }
function autoResize(textarea) { if (textarea) { textarea.style.height = 'auto'; textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px'; } }

// Message display
function addMessageToUI(sender, content) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    chatMessages.style.display = 'flex';
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    const ts = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    div.innerHTML = `<div class="message-avatar">${sender === 'user' ? '👤' : '🤖'}</div><div class="message-content">${content}<div class="message-time">${ts}</div></div>`;
    chatMessages.appendChild(div);
    scrollToBottom();
}

function addImageMessageToUI(sender, imageUrl) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    chatMessages.style.display = 'flex';
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    const ts = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    div.innerHTML = `<div class="message-avatar">${sender === 'user' ? '👤' : '🤖'}</div><div class="message-content image-message"><img src="${imageUrl}" alt="Uploaded image" class="chat-image" onclick="openImageModal('${imageUrl}')"><div class="message-time">${ts}</div></div>`;
    chatMessages.appendChild(div);
    scrollToBottom();
}

function openImageModal(imageUrl) {
    const modal = document.createElement('div');
    modal.className = 'image-modal-overlay';
    modal.onclick = () => closeImageModal(modal);
    modal.innerHTML = `<div class="image-modal-content" onclick="event.stopPropagation()"><button class="image-modal-close" onclick="closeImageModal(this.closest('.image-modal-overlay'))">✕</button><img src="${imageUrl}" alt="Full size image" class="image-modal-img"></div>`;
    modal.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.8);display:flex;align-items:center;justify-content:center;z-index:10000;cursor:pointer;';
    document.body.appendChild(modal);
}
function closeImageModal(modal) { if (modal && document.body.contains(modal)) document.body.removeChild(modal); }
function hideWelcomeSection() { const w = document.getElementById('welcomeSection'); const m = document.getElementById('chatMessages'); if (w) w.style.display = 'none'; if (m) m.style.display = 'flex'; }
function scrollToBottom() { const m = document.getElementById('chatMessages'); if (m) m.scrollTop = m.scrollHeight; }
function showTypingIndicator() { const t = document.getElementById('typingIndicator'); if (t) { t.style.display = 'flex'; scrollToBottom(); } }
function hideTypingIndicator() { const t = document.getElementById('typingIndicator'); if (t) t.style.display = 'none'; }
function showNotification(message, type = 'info') {
    const n = document.createElement('div');
    n.className = `notification ${type}`;
    n.textContent = message;
    n.style.cssText = `position:fixed;top:20px;right:20px;padding:12px 20px;background:${type==='success'?'#4CAF50':type==='error'?'#f44336':'#2196F3'};color:white;border-radius:8px;z-index:10000;font-weight:500;box-shadow:0 4px 15px rgba(0,0,0,0.2);`;
    document.body.appendChild(n);
    setTimeout(() => { n.style.opacity = '0'; n.style.transform = 'translateX(100%)'; setTimeout(() => { if (document.body.contains(n)) document.body.removeChild(n); }, 300); }, 3000);
}

// File handling
function toggleFileUpload() { const fi = document.getElementById('fileInput'); if (fi) fi.click(); }
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    const valid = ['image/jpeg','image/jpg','image/png','image/gif','image/bmp','image/webp'];
    if (!valid.includes(file.type)) { showNotification('Please select a valid image file (JPEG, PNG, GIF, BMP, WebP)', 'error'); event.target.value = ''; return; }
    if (file.size > 16 * 1024 * 1024) { showNotification('File is too large. Maximum size is 16MB.', 'error'); event.target.value = ''; return; }
    currentFile = file;
    cleanupFilePreview();
    const reader = new FileReader();
    reader.onload = (e) => { currentFilePreviewUrl = e.target.result; showFilePreview(file, currentFilePreviewUrl); };
    reader.readAsDataURL(file);
}
function cleanupFilePreview() { if (currentFilePreviewUrl && currentFilePreviewUrl.startsWith('blob:')) URL.revokeObjectURL(currentFilePreviewUrl); currentFilePreviewUrl = null; }
function showFilePreview(file, previewUrl) {
    const area = document.getElementById('fileUploadArea');
    const text = document.getElementById('filePreviewText');
    const img = document.getElementById('filePreviewImage');
    if (area && text && img) { area.style.display = 'flex'; text.textContent = `📷 ${file.name} (${formatFileSize(file.size)})`; img.src = previewUrl; img.style.display = 'block'; }
}
function removeFile() { cleanupFilePreview(); currentFile = null; const a = document.getElementById('fileUploadArea'); const t = document.getElementById('filePreviewText'); const i = document.getElementById('filePreviewImage'); const fi = document.getElementById('fileInput'); if (a) a.style.display='none'; if (t) t.textContent=''; if (i){ i.src=''; i.style.display='none'; } if (fi) fi.value=''; }
function formatFileSize(bytes) { if (bytes===0) return '0 Bytes'; const k=1024; const sizes=['Bytes','KB','MB','GB']; const i=Math.floor(Math.log(bytes)/Math.log(k)); return parseFloat((bytes/Math.pow(k,i)).toFixed(2))+' '+sizes[i]; }

// Voice input (MediaRecorder + optional SpeechRecognition)
function toggleVoiceInput() { if (isRecording) { stopVoiceInput(); } else { startMediaRecorderFallback(); } }
function startVoiceInput() { startMediaRecorderFallback(); }
function stopVoiceInput() {
    isRecording = false;
    const voiceBtn = document.getElementById('voiceBtn');
    const voiceIndicator = document.getElementById('voiceIndicator');
    if (voiceBtn) { voiceBtn.classList.remove('active'); voiceBtn.innerHTML = '🎤'; }
    if (voiceIndicator) voiceIndicator.style.display = 'none';
    if (recognition) { try { recognition.stop(); } catch (e) {} }
    try { if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop(); } catch (e) {}
}

function setupSpeechRecognition() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = false; recognition.interimResults = false; recognition.lang = 'en-US';
        recognition.onresult = (event) => {
            let interim = '', finalText = '';
            for (let i = event.resultIndex; i < event.results.length; i++) { const res = event.results[i]; if (res.isFinal) finalText += res[0].transcript; else interim += res[0].transcript; }
            const messageInput = document.getElementById('messageInput');
            if (messageInput) { messageInput.value = (finalText || interim).trim(); autoResize(messageInput); }
            if (finalText.trim()) stopVoiceInput();
        };
        recognition.onerror = (e) => { const benign = ['no-speech','aborted','audio-capture']; stopVoiceInput(); if (!benign.includes(e.error)) showNotification('Voice recognition error. Please try again.', 'error'); };
        recognition.onend = () => { if (isRecording) stopVoiceInput(); };
    }
}

let mediaRecorder = null; let recordedChunks = [];
async function startMediaRecorderFallback() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: { noiseSuppression: true, echoCancellation: true, autoGainControl: true, channelCount: 1, sampleRate: 48000 } });
        recordedChunks = [];
        const preferredType = (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) ? 'audio/webm;codecs=opus' : 'audio/webm';
        mediaRecorder = new MediaRecorder(stream, { mimeType: preferredType, audioBitsPerSecond: 128000 });
        const voiceBtn = document.getElementById('voiceBtn'); const voiceIndicator = document.getElementById('voiceIndicator');
        if (voiceBtn) { voiceBtn.classList.add('active'); voiceBtn.innerHTML = '🔴'; }
        if (voiceIndicator) voiceIndicator.style.display = 'flex';
        mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) recordedChunks.push(e.data); };
        mediaRecorder.onstop = async () => { const blob = new Blob(recordedChunks, { type: 'audio/webm' }); await uploadAudioForTranscription(blob); stopVoiceUI(); stream.getTracks().forEach(t => t.stop()); };
        mediaRecorder.start();
    } catch (e) {
        showNotification('Microphone access denied or unavailable', 'error');
        stopVoiceUI();
    }
}
function stopVoiceUI() { isRecording = false; const voiceBtn = document.getElementById('voiceBtn'); const voiceIndicator = document.getElementById('voiceIndicator'); if (voiceBtn) { voiceBtn.classList.remove('active'); voiceBtn.innerHTML = '🎤'; } if (voiceIndicator) voiceIndicator.style.display = 'none'; }
async function uploadAudioForTranscription(blob) {
    try { const formData = new FormData(); formData.append('audio', blob, 'voice.webm'); const res = await fetch('/api/voice_to_text', { method: 'POST', body: formData, credentials: 'include' }); if (!res.ok) { showNotification('Audio upload failed', 'error'); return; } const data = await res.json(); const transcript = (data.transcript || '').trim(); const messageInput = document.getElementById('messageInput'); if (transcript && transcript.length >= 3 && messageInput) { messageInput.value = transcript; autoResize(messageInput); } else { showNotification('Could not transcribe clearly. Please try again.', 'error'); } } catch (e) { showNotification('Transcription error. Please try again.', 'error'); }
}

// Drawing
function setupCanvas() {
    canvas = document.getElementById('drawingCanvas');
    if (canvas) {
        ctx = canvas.getContext('2d');
        canvas.width = 800; canvas.height = 600;
        ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = 'black'; ctx.lineWidth = brushSize; ctx.fillStyle = 'white'; ctx.fillRect(0, 0, canvas.width, canvas.height);
        addCanvasEventListeners();
    }
}
function addCanvasEventListeners() {
    if (!canvas) return;
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    // Use passive:false so preventDefault works on mobile to stop page scroll while drawing
    canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    canvas.addEventListener('touchend', stopDrawing, { passive: false });
}
function getCanvasCoordinates(e) {
    const rect = canvas.getBoundingClientRect(); const scaleX = canvas.width / rect.width; const scaleY = canvas.height / rect.height;
    if (e.touches && e.touches[0]) return { x: (e.touches[0].clientX - rect.left) * scaleX, y: (e.touches[0].clientY - rect.top) * scaleY };
    return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
}
function startDrawing(e) { e.preventDefault(); isDrawing = true; const c = getCanvasCoordinates(e); lastX = c.x; lastY = c.y; if (currentDrawMode === 'erase') drawEraserCursor(c.x, c.y); }
function draw(e) {
    if (!isDrawing) return; e.preventDefault(); const c = getCanvasCoordinates(e); ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(c.x, c.y);
    if (currentDrawMode === 'erase') { ctx.globalCompositeOperation = 'destination-out'; ctx.lineWidth = brushSize * 2; } else { ctx.globalCompositeOperation = 'source-over'; ctx.strokeStyle = 'black'; ctx.lineWidth = brushSize; }
    ctx.stroke(); lastX = c.x; lastY = c.y; if (currentDrawMode === 'erase') drawEraserCursor(c.x, c.y);
}
function stopDrawing(e) { if (isDrawing) { e.preventDefault(); isDrawing = false; ctx.globalCompositeOperation = 'source-over'; removeEraserPreview(); } }
function handleTouchStart(e) { e.preventDefault(); startDrawing(e); }
function handleTouchMove(e) { e.preventDefault(); draw(e); }
function openDrawingPad() {
    const modal = document.getElementById('drawingModal');
    if (modal) {
        modal.classList.add('active');
        clearCanvas();
        currentDrawMode = 'draw'; brushSize = 3; updateDrawingControls();
        if (canvas && ctx) { ctx.fillStyle = 'white'; ctx.fillRect(0, 0, canvas.width, canvas.height); }
        resizeCanvasToFitModal();
        window.addEventListener('resize', resizeCanvasToFitModal);
    }
}
function closeDrawingPad() { const modal = document.getElementById('drawingModal'); if (modal) modal.classList.remove('active'); removeEraserPreview(); window.removeEventListener('resize', resizeCanvasToFitModal); }
function clearCanvas() { if (ctx && canvas) { ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle='white'; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.globalCompositeOperation='source-over'; } }
function resizeCanvasToFitModal() {
    const container = document.querySelector('.drawing-container'); const header = document.querySelector('.drawing-header'); const controls = document.querySelector('.drawing-controls'); if (!container || !canvas) return;
    const cs = getComputedStyle(container); const py = parseInt(cs.paddingTop) + parseInt(cs.paddingBottom); const hh = header ? header.offsetHeight : 0; const ch = controls ? controls.offsetHeight : 0; const gap = 8; const cms = getComputedStyle(canvas); const cmb = parseInt(cms.marginBottom) || 0; const h = Math.max(200, container.clientHeight - py - hh - ch - gap - cmb); const w = container.clientWidth - (parseInt(cs.paddingLeft) + parseInt(cs.paddingRight)); canvas.style.width = w + 'px'; canvas.style.height = h + 'px'; const dpr = window.devicePixelRatio || 1; const nw = Math.floor(w * dpr); const nh = Math.floor(h * dpr); if (canvas.width !== nw || canvas.height !== nh) { canvas.width = nw; canvas.height = nh; ctx.scale(dpr, dpr); ctx.fillStyle='white'; ctx.fillRect(0,0,w,h); ctx.lineCap='round'; ctx.lineJoin='round'; ctx.strokeStyle = currentDrawMode === 'draw' ? 'black' : 'white'; ctx.lineWidth = brushSize; }
}
function setDrawMode(mode) { currentDrawMode = mode; updateDrawingControls(); if (currentDrawMode !== 'erase') removeEraserPreview(); }
function changeBrushSize(size) { brushSize = parseInt(size); const d = document.getElementById('brushSizeDisplay'); if (d) d.textContent = `${brushSize}px`; }
function updateDrawingControls() { const drawBtn = document.getElementById('drawMode'); const eraseBtn = document.getElementById('eraseMode'); if (drawBtn && eraseBtn) { drawBtn.classList.toggle('active', currentDrawMode === 'draw'); eraseBtn.classList.toggle('active', currentDrawMode === 'erase'); } const input = document.getElementById('brushSize'); const disp = document.getElementById('brushSizeDisplay'); if (input) input.value = brushSize; if (disp) disp.textContent = `${brushSize}px`; if (canvas) canvas.style.cursor = currentDrawMode === 'erase' ? 'none' : 'crosshair'; }
function drawEraserCursor(x,y) { if (!canvas || !ctx) return; ctx.save(); ctx.globalCompositeOperation = 'source-over'; let preview = document.getElementById('eraserPreview'); if (!preview) { preview = document.createElement('div'); preview.id='eraserPreview'; preview.style.position='absolute'; preview.style.pointerEvents='none'; preview.style.border='1px solid rgba(17,24,39,0.5)'; preview.style.borderRadius='50%'; preview.style.boxShadow='0 0 0 2px rgba(255,255,255,0.6) inset'; const modal = document.getElementById('drawingModal'); if (modal) modal.appendChild(preview); } const rect = canvas.getBoundingClientRect(); const modal = document.getElementById('drawingModal'); const mrect = modal ? modal.getBoundingClientRect() : { left:0, top:0 }; const scaleX = rect.width / canvas.width; const scaleY = rect.height / canvas.height; const radius = (brushSize) * scaleX; preview.style.width = `${radius*2}px`; preview.style.height = `${radius*2}px`; preview.style.left = `${(rect.left - mrect.left) + x*scaleX - radius}px`; preview.style.top = `${(rect.top - mrect.top) + y*scaleY - radius}px`; ctx.restore(); }
function removeEraserPreview() { const p = document.getElementById('eraserPreview'); if (p && p.parentNode) p.parentNode.removeChild(p); }
async function submitDrawing() {
    if (!canvas) { showNotification('Drawing canvas not available', 'error'); return; }
    const imageData = ctx.getImageData(0,0,canvas.width,canvas.height); const pixelData = imageData.data; let has = false; for (let i=0;i<pixelData.length;i+=4){ const r=pixelData[i],g=pixelData[i+1],b=pixelData[i+2]; if (r!==255||g!==255||b!==255){ has=true; break; }} if (!has) { showNotification('Please draw something before submitting', 'error'); return; }
    hideWelcomeSection();
    canvas.toBlob(async function(blob){ if(!blob){ showNotification('Failed to process drawing','error'); return; } const formData = new FormData(); formData.append('image', blob, 'drawing.png'); if (currentChatId) formData.append('chat_id', currentChatId); const drawingUrl = canvas.toDataURL(); addImageMessageToUI('user', drawingUrl); closeDrawingPad(); showTypingIndicator(); try { const res = await fetch('/api/recognize_handwriting', { method:'POST', credentials:'include', body: formData }); hideTypingIndicator(); if (!res.ok) { try { await res.json(); } catch (e) {} throw new Error(); } const data = await res.json(); currentChatId = data.chat_id || currentChatId; addMessageToUI('bot', data.response); await loadChatHistory(); } catch(e){ hideTypingIndicator(); addMessageToUI('bot','Sorry, I had trouble processing your drawing. Please check your connection and try again.'); } }, 'image/png', 0.8);
}

// Clock & Calculator & To-Do
function openClock() { const m = document.getElementById('clockModal'); if (!m) return; const now = new Date(); const el = document.getElementById('clockDisplay'); if (el) el.textContent = now.toLocaleTimeString(); m.style.display = 'flex'; }
function closeClock() { const m = document.getElementById('clockModal'); if (m) m.style.display = 'none'; }
setInterval(() => { const el = document.getElementById('clockDisplay'); if (el && document.getElementById('clockModal')?.style.display === 'flex') el.textContent = new Date().toLocaleTimeString(); }, 500);
async function evaluateCalculator() { const input = document.getElementById('calcInput'); const out = document.getElementById('calcResult'); if (!input || !out) return; const expr = (input.value||'').trim(); if (!expr) { out.textContent='Enter an expression'; return; } try { const res = await fetch('/api/calc', { method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include', body: JSON.stringify({ expression: expr }) }); const data = await res.json(); if (!res.ok) throw new Error(data.error || 'Calculation failed'); out.textContent = `Result: ${data.result}`; } catch(e){ out.textContent = `Error: ${e.message}`; } }
function openCalculator() { window.location.href = 'calculator.html'; }
function closeCalculator() { /* no-op when on separate page */ }
function openTodoPage() { window.location.href = 'todo.html'; }

console.log('📝 BYTE Chat Script loaded');

// Expose functions used by inline HTML onclick/handlers to global scope
window.toggleSidebar = toggleSidebar;
window.clearCurrentChat = clearCurrentChat;
window.createNewChat = createNewChat;
window.deleteChat = deleteChat;
window.closeDeleteModal = closeDeleteModal;
window.confirmDeleteChat = confirmDeleteChat;
window.startRenameChat = startRenameChat;
window.finishRenameChat = finishRenameChat;
window.cancelRenameChat = cancelRenameChat;
window.sendMessage = sendMessage;
window.handleKeyPress = handleKeyPress;
window.autoResize = autoResize;
window.toggleFileUpload = toggleFileUpload;
window.handleFileSelect = handleFileSelect;
window.removeFile = removeFile;
window.toggleVoiceInput = toggleVoiceInput;
window.stopVoiceInput = stopVoiceInput;
window.openDrawingPad = openDrawingPad;
window.closeDrawingPad = closeDrawingPad;
window.clearCanvas = clearCanvas;
window.setDrawMode = setDrawMode;
window.changeBrushSize = changeBrushSize;
window.submitDrawing = submitDrawing;
window.openImageModal = openImageModal;
window.closeImageModal = closeImageModal;
window.openClock = openClock;
window.closeClock = closeClock;
window.evaluateCalculator = evaluateCalculator;
window.openCalculator = openCalculator;
window.closeCalculator = closeCalculator;
window.openTodoPage = openTodoPage;
window.logout = logout;


