// Configuration
const API_URL = 'http://localhost:8000';

// DOM Elements
const startBtn = document.getElementById('startBtn');
const videoStream = document.getElementById('videoStream');
const placeholder = document.getElementById('placeholder');
const wordDisplay = document.getElementById('wordDisplay');
const spaceBtn = document.getElementById('spaceBtn');
const deleteBtn = document.getElementById('deleteBtn');
const clearBtn = document.getElementById('clearBtn');
const status = document.getElementById('status');

// State
let isRunning = false;
let pollingInterval = null;

// Show status message
function showStatus(message, type = 'normal') {
    if (status) {
        status.textContent = message;
        status.className = 'status';
        if (type === 'active') {
            status.classList.add('active');
        } else if (type === 'error') {
            status.classList.add('error');
        }
    }
    console.log('[Status]', message);
}

// Start/Stop webcam
if (startBtn) {
    startBtn.addEventListener('click', async () => {
        if (!isRunning) {
            // Start webcam
            try {
                showStatus('Starting backend webcam...', 'normal');
                console.log('Calling /start_webcam...');
                
                const response = await fetch(`${API_URL}/start_webcam`, {
                    method: 'POST'
                });
                
                const data = await response.json();
                console.log('Start webcam response:', data);
                
                if (data.success) {
                    // Show video stream
                    videoStream.src = `${API_URL}/video_feed`;
                    videoStream.style.display = 'block';
                    placeholder.style.display = 'none';
                    
                    // Update button
                    startBtn.textContent = 'Stop Backend Webcam';
                    startBtn.classList.remove('btn-primary');
                    startBtn.classList.add('btn-danger');
                    
                    isRunning = true;
                    showStatus('Webcam active - Show hand gestures', 'active');
                    
                    // Start polling for word updates
                    startWordPolling();
                } else {
                    showStatus('Failed to start webcam: ' + data.message, 'error');
                }
            } catch (error) {
                console.error('Error starting webcam:', error);
                showStatus('Error: Could not connect to backend', 'error');
            }
        } else {
            // Stop webcam
            try {
                console.log('Calling /stop_webcam...');
                const response = await fetch(`${API_URL}/stop_webcam`, {
                    method: 'POST'
                });
                
                const data = await response.json();
                console.log('Stop webcam response:', data);
                
                if (data.success) {
                    // Hide video stream
                    videoStream.style.display = 'none';
                    placeholder.style.display = 'block';
                    videoStream.src = '';
                    
                    // Update button
                    startBtn.textContent = 'Start Backend Webcam';
                    startBtn.classList.remove('btn-danger');
                    startBtn.classList.add('btn-primary');
                    
                    isRunning = false;
                    showStatus('Webcam stopped', 'normal');
                    
                    // Stop polling
                    stopWordPolling();
                }
            } catch (error) {
                console.error('Error stopping webcam:', error);
                showStatus('Error stopping webcam', 'error');
            }
        }
    });
}

// Poll for word updates
function startWordPolling() {
    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/get_word`);
            const data = await response.json();
            if (wordDisplay) {
                wordDisplay.textContent = data.word || '[empty]';
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 500); // Update every 500ms
}

function stopWordPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

// Add space button
if (spaceBtn) {
    spaceBtn.addEventListener('click', async () => {
        try {
            const response = await fetch(`${API_URL}/add_space`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.success && wordDisplay) {
                wordDisplay.textContent = data.word || '[empty]';
                showStatus('Space added', 'active');
            }
        } catch (error) {
            console.error('Error:', error);
            showStatus('Error adding space', 'error');
        }
    });
}

// Delete button
if (deleteBtn) {
    deleteBtn.addEventListener('click', async () => {
        try {
            const response = await fetch(`${API_URL}/delete_char`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.success && wordDisplay) {
                wordDisplay.textContent = data.word || '[empty]';
                showStatus('Character deleted', 'active');
            }
        } catch (error) {
            console.error('Error:', error);
            showStatus('Error deleting character', 'error');
        }
    });
}

// Clear button
if (clearBtn) {
    clearBtn.addEventListener('click', async () => {
        if (!confirm('Clear the entire word?')) return;
        
        try {
            const response = await fetch(`${API_URL}/clear_word`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.success && wordDisplay) {
                wordDisplay.textContent = '[empty]';
                showStatus('Word cleared', 'active');
            }
        } catch (error) {
            console.error('Error:', error);
            showStatus('Error clearing word', 'error');
        }
    });
}

// Check backend connection on load
window.addEventListener('load', async () => {
    console.log('Page loaded, checking backend connection...');
    try {
        const response = await fetch(`${API_URL}/`);
        const data = await response.json();
        console.log('Backend response:', data);
        showStatus('Connected to backend - Ready to start', 'active');
    } catch (error) {
        console.error('Backend connection error:', error);
        showStatus('Cannot connect to backend - Make sure app.py is running', 'error');
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (isRunning) {
        fetch(`${API_URL}/stop_webcam`, { method: 'POST' });
    }
});