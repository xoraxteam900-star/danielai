/**
 * Daniel Voice Assistant - Frontend Application
 * Handles audio/video capture, wake word detection, and TTS
 */

// Configuration
const CONFIG = {
    BACKEND_URL: 'http://localhost:8000',
    WAKE_WORD: 'hey daniel',
    AUDIO_CHUNK_DURATION: 2000, // ms
    VISUALIZER_UPDATE_INTERVAL: 50
};

// Application State
const state = {
    isListening: false,
    isMuted: false,
    wakeWordEnabled: true,
    cameraEnabled: false,
    isProcessing: false,
    isPushToTalk: false,
    mediaRecorder: null,
    audioChunks: [],
    stream: null,
    audioContext: null,
    analyser: null,
    lastCommand: ''
};

// DOM Elements
const elements = {
    statusIndicator: document.getElementById('statusIndicator'),
    statusText: document.getElementById('statusText'),
    cameraFeed: document.getElementById('cameraFeed'),
    cameraPlaceholder: document.getElementById('cameraPlaceholder'),
    cameraToggleBtn: document.getElementById('cameraToggleBtn'),
    micToggleBtn: document.getElementById('micToggleBtn'),
    pushToTalkBtn: document.getElementById('pushToTalkBtn'),
    wakeWordToggle: document.getElementById('wakeWordToggle'),
    conversationLog: document.getElementById('conversationLog'),
    backendStatus: document.getElementById('backendStatus'),
    messinessBadge: document.getElementById('messinessBadge'),
    messinessLabel: document.getElementById('messinessLabel'),
    visualizerCanvas: document.getElementById('visualizerCanvas')
};

// Initialize Application
async function init() {
    console.log('Initializing Daniel...');
    
    // Check backend connection
    await checkBackend();
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize audio context for visualizer
    initAudioVisualizer();
    
    updateStatus('Ready', true);
    addMessage('system', '👋 Hello! I\'m Daniel. Say "Hey Daniel" to wake me up, or press the button to talk!');
}

// Check backend connection
async function checkBackend() {
    try {
        const response = await fetch(`${CONFIG.BACKEND_URL}/health`);
        const data = await response.json();
        elements.backendStatus.textContent = 'Connected to backend';
        elements.backendStatus.style.color = 'var(--success-color)';
        return true;
    } catch (error) {
        console.error('Backend not available:', error);
        elements.backendStatus.textContent = 'Backend not connected - some features may not work';
        elements.backendStatus.style.color = 'var(--warning-color)';
        return false;
    }
}

// Setup event listeners
function setupEventListeners() {
    // Camera toggle
    elements.cameraToggleBtn.addEventListener('click', toggleCamera);
    
    // Mic toggle
    elements.micToggleBtn.addEventListener('click', toggleMic);
    
    // Push to talk
    elements.pushToTalkBtn.addEventListener('mousedown', startPushToTalk);
    elements.pushToTalkBtn.addEventListener('mouseup', stopPushToTalk);
    elements.pushToTalkBtn.addEventListener('mouseleave', stopPushToTalk);
    elements.pushToTalkBtn.addEventListener('touchstart', startPushToTalk);
    elements.pushToTalkBtn.addEventListener('touchend', stopPushToTalk);
    
    // Wake word toggle
    elements.wakeWordToggle.addEventListener('change', (e) => {
        state.wakeWordEnabled = e.target.checked;
        if (state.wakeWordEnabled && !state.isMuted) {
            startListening();
        } else if (!state.wakeWordEnabled) {
            stopListening();
        }
    });
}

// Update status indicator
function updateStatus(text, isActive = false) {
    elements.statusText.textContent = text;
    const dot = elements.statusIndicator.querySelector('.status-dot');
    if (isActive) {
        dot.classList.add('active');
    } else {
        dot.classList.remove('active');
    }
}

// Initialize audio visualizer
function initAudioVisualizer() {
    const canvas = elements.visualizerCanvas;
    const ctx = canvas.getContext('2d');
    
    canvas.width = canvas.offsetWidth || 300;
    canvas.height = canvas.offsetHeight || 80;
    
    function draw() {
        if (!state.analyser) {
            requestAnimationFrame(draw);
            return;
        }
        
        const bufferLength = state.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        state.analyser.getByteFrequencyData(dataArray);
        
        ctx.fillStyle = 'rgba(15, 23, 42, 0.3)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        const barWidth = (canvas.width / bufferLength) * 2.5;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height;
            
            const gradient = ctx.createLinearGradient(0, canvas.height, 0, canvas.height - barHeight);
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#22c55e');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
            
            x += barWidth + 1;
        }
        
        requestAnimationFrame(draw);
    }
    
    draw();
}

// Toggle camera
async function toggleCamera() {
    if (state.cameraEnabled) {
        await stopCamera();
    } else {
        await startCamera();
    }
}

// Start camera
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        
        elements.cameraFeed.srcObject = stream;
        elements.cameraFeed.classList.add('active');
        elements.cameraPlaceholder.style.display = 'none';
        elements.cameraToggleBtn.textContent = 'Turn Off Camera';
        state.cameraEnabled = true;
        
        addMessage('system', '📷 Camera is now on!');
        
    } catch (error) {
        console.error('Camera error:', error);
        addMessage('system', '❌ Could not access camera. Please check permissions.');
    }
}

// Stop camera
async function stopCamera() {
    if (elements.cameraFeed.srcObject) {
        elements.cameraFeed.srcObject.getTracks().forEach(track => track.stop());
        elements.cameraFeed.srcObject = null;
    }
    
    elements.cameraFeed.classList.remove('active');
    elements.cameraPlaceholder.style.display = 'flex';
    elements.cameraToggleBtn.textContent = 'Turn On Camera';
    elements.messinessBadge.style.display = 'none';
    state.cameraEnabled = false;
    
    addMessage('system', '📷 Camera is now off.');
}

// Toggle microphone
async function toggleMic() {
    if (state.isMuted) {
        await unmuteMic();
    } else {
        await muteMic();
    }
}

// Mute microphone
async function muteMic() {
    state.isMuted = true;
    elements.micToggleBtn.textContent = 'Unmute';
    elements.micToggleBtn.classList.remove('btn-secondary');
    elements.micToggleBtn.classList.add('btn-warning');
    elements.pushToTalkBtn.disabled = true;
    
    stopListening();
    addMessage('system', '🔇 Microphone muted.');
}

// Unmute microphone
async function unmuteMic() {
    state.isMuted = false;
    elements.micToggleBtn.textContent = 'Mute';
    elements.micToggleBtn.classList.remove('btn-warning');
    elements.micToggleBtn.classList.add('btn-secondary');
    elements.pushToTalkBtn.disabled = false;
    
    if (state.wakeWordEnabled) {
        await startListening();
    }
    
    addMessage('system', '🔊 Microphone unmuted.');
}

// Start listening (for wake word)
async function startListening() {
    if (state.isListening || state.isMuted) return;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        state.stream = stream;
        
        // Set up audio context for visualizer
        state.audioContext = new AudioContext();
        const source = state.audioContext.createMediaStreamSource(stream);
        state.analyser = state.audioContext.createAnalyser();
        state.analyser.fftSize = 256;
        source.connect(state.analyser);
        
        // Set up MediaRecorder for chunked audio
        state.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });
        
        state.audioChunks = [];
        
        state.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                state.audioChunks.push(e.data);
            }
        };
        
        state.mediaRecorder.onstop = async () => {
            if (state.audioChunks.length > 0 && !state.isMuted) {
                await processAudio();
            }
            state.audioChunks = [];
        };
        
        state.mediaRecorder.start(CONFIG.AUDIO_CHUNK_DURATION);
        state.isListening = true;
        
        updateStatus('Listening for "Hey Daniel"...', true);
        
    } catch (error) {
        console.error('Microphone error:', error);
        addMessage('system', '❌ Could not access microphone. Please check permissions.');
    }
}

// Stop listening
function stopListening() {
    if (!state.isListening) return;
    
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
    }
    
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
    }
    
    state.isListening = false;
    updateStatus('Ready', true);
}

// Push to Talk - Start
async function startPushToTalk(e) {
    e.preventDefault();
    if (state.isMuted || state.isProcessing) return;
    
    state.isPushToTalk = true;
    elements.pushToTalkBtn.classList.add('active');
    elements.pushToTalkBtn.textContent = 'Listening...';
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        state.stream = stream;
        
        // Set up audio context
        state.audioContext = new AudioContext();
        const source = state.audioContext.createMediaStreamSource(stream);
        state.analyser = state.audioContext.createAnalyser();
        state.analyser.fftSize = 256;
        source.connect(state.analyser);
        
        // Set up MediaRecorder
        state.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });
        
        state.audioChunks = [];
        
        state.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                state.audioChunks.push(e.data);
            }
        };
        
        state.mediaRecorder.onstop = async () => {
            await processAudio(true);
            state.audioChunks = [];
        };
        
        state.mediaRecorder.start(100);
        
    } catch (error) {
        console.error('Push to talk error:', error);
    }
}

// Push to Talk - Stop
function stopPushToTalk() {
    if (!state.isPushToTalk) return;
    
    state.isPushToTalk = false;
    elements.pushToTalkBtn.classList.remove('active');
    elements.pushToTalkBtn.textContent = 'Hold to Speak';
    
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
    }
}

// Process audio through backend
async function processAudio(isPushToTalk = false) {
    if (state.isProcessing) return;
    state.isProcessing = true;
    
    try {
        // Create audio blob
        const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
        
        // Stop stream
        if (state.stream) {
            state.stream.getTracks().forEach(track => track.stop());
            state.stream = null;
        }
        
        // Create FormData
        const formData = new FormData();
        formData.append('audio', audioBlob, 'audio.webm');
        
        // Send to backend
        const response = await fetch(`${CONFIG.BACKEND_URL}/stt`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        console.log('STT Response:', data);
        
        if (data.text) {
            addMessage('user', `🎤 "${data.text}"`);
            
            // Check for wake word or process command directly
            if (data.wake_word_detected || isPushToTalk) {
                await handleCommand(data.command || data.text, isPushToTalk);
            } else if (state.wakeWordEnabled) {
                // Just listening for wake word
            }
        }
        
    } catch (error) {
        console.error('Audio processing error:', error);
        addMessage('system', '❌ Error processing audio. Please try again.');
    } finally {
        state.isProcessing = false;
        
        // Restart listening if wake word enabled and not push to talk
        if (state.wakeWordEnabled && !state.isMuted && !state.isPushToTalk) {
            startListening();
        }
    }
}

// Handle command
async function handleCommand(command, hasFrame = false) {
    try {
        let frameData = null;
        
        // Capture frame if camera is on
        if (state.cameraEnabled && hasFrame) {
            const canvas = document.createElement('canvas');
            canvas.width = elements.cameraFeed.videoWidth;
            canvas.height = elements.cameraFeed.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(elements.cameraFeed, 0, 0);
            frameData = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
        }
        
        const formData = new FormData();
        formData.append('command', command);
        
        if (frameData) {
            formData.append('frame_data', frameData);
        }
        
        const response = await fetch(`${CONFIG.BACKEND_URL}/command`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        console.log('Command Response:', data);
        
        // Add assistant response
        if (data.response) {
            addMessage('assistant', `🤖 ${data.response}`);
            speakText(data.response);
        }
        
        // Update messiness badge
        if (data.messiness) {
            updateMessinessBadge(data.messiness);
        }
        
    } catch (error) {
        console.error('Command handling error:', error);
        addMessage('system', '❌ Error processing command.');
    }
}

// Text to Speech using Web Speech API
function speakText(text) {
    if (!('speechSynthesis' in window)) {
        console.error('TTS not supported');
        return;
    }
    
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    
    // Configure voice
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    
    // Try to find a good English voice
    const voices = window.speechSynthesis.getVoices();
    const englishVoice = voices.find(v => v.lang.startsWith('en-') && v.name.includes('Google')) ||
                        voices.find(v => v.lang.startsWith('en-')) ||
                        voices[0];
    
    if (englishVoice) {
        utterance.voice = englishVoice;
    }
    
    utterance.onstart = () => {
        console.log('TTS started');
    };
    
    utterance.onend = () => {
        console.log('TTS ended');
    };
    
    utterance.onerror = (e) => {
        console.error('TTS error:', e);
    };
    
    window.speechSynthesis.speak(utterance);
}

// Update messiness badge
function updateMessinessBadge(messiness) {
    elements.messinessBadge.style.display = 'block';
    
    if (messiness.level === 'clean') {
        elements.messinessLabel.textContent = '✨ Room: Clean';
        elements.messinessBadge.style.borderLeft = '4px solid var(--success-color)';
    } else if (messiness.level === 'messy') {
        elements.messinessLabel.textContent = '🧹 Room: Messy';
        elements.messinessBadge.style.borderLeft = '4px solid var(--danger-color)';
    } else {
        elements.messinessLabel.textContent = '😐 Room: Moderate';
        elements.messinessBadge.style.borderLeft = '4px solid var(--warning-color)';
    }
}

// Add message to conversation log
function addMessage(type, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${type}`;
    messageDiv.innerHTML = `<span class="message-text">${text}</span>`;
    
    elements.conversationLog.appendChild(messageDiv);
    elements.conversationLog.scrollTop = elements.conversationLog.scrollHeight;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);
