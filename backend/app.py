"""
FastAPI Backend with Webcam Streaming

Backend opens its own webcam and streams video to frontend.
All gesture processing happens here on the backend.

Run with: python app.py
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import threading
import time

# Import our utility modules
from utils.config import *
from utils.model_manager import GestureModelManager
from utils.hand_detector import HandDetector
from utils.feature_extractor import extract_hand_features
from utils.word_builder import WordBuilder
from utils.visualization import draw_hand_landmarks, draw_ui_overlay


# ===================================================================
# FASTAPI APP SETUP
# ===================================================================

app = FastAPI(title="Hand Gesture Recognition - Backend Webcam")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ===================================================================
# GLOBAL VARIABLES
# ===================================================================

model_manager = None
hand_detector = None
word_builder = None

# Webcam state
webcam = None
webcam_active = False
current_frame = None
frame_lock = threading.Lock()


# ===================================================================
# INITIALIZE MODELS ON STARTUP
# ===================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models when server starts"""
    global model_manager, hand_detector, word_builder
    
    print("\n" + "=" * 60)
    print("🚀 Starting Hand Gesture Recognition Backend")
    print("=" * 60)
    
    try:
        print("Loading ML models...")
        model_manager = GestureModelManager(MODEL_PATH, SCALER_PATH)
        hand_detector = HandDetector(MAX_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE)
        word_builder = WordBuilder(PREDICTION_INTERVAL, LETTER_ADD_INTERVAL, CONFIDENCE_THRESHOLD)
        
        print("✅ Models loaded successfully!")
        print(f"\n🌐 Open browser: http://localhost:8000/static/index.html")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up when server shuts down"""
    global hand_detector, webcam, webcam_active
    
    # Stop webcam
    webcam_active = False
    if webcam:
        webcam.release()
    
    # Close hand detector
    if hand_detector:
        hand_detector.close()
    
    print("👋 Server shutdown complete")


# ===================================================================
# WEBCAM PROCESSING THREAD
# ===================================================================

def process_webcam():
    """
    Runs in background thread:
    - Captures frames from backend webcam
    - Detects hands and makes predictions
    - Draws overlays
    - Stores processed frame for streaming
    """
    global webcam, current_frame, webcam_active
    
    print("📷 Opening backend webcam...")
    webcam = cv2.VideoCapture(0)
    
    if not webcam.isOpened():
        print("❌ Could not open webcam!")
        return
    
    print("✅ Webcam opened successfully!")
    prediction_text = "Show hand gesture..."
    
    while webcam_active:
        ret, frame = webcam.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        detection_result = hand_detector.detect(frame)
        features, hand_landmarks = extract_hand_features(detection_result)
        
        # Draw hand landmarks
        draw_hand_landmarks(frame, hand_landmarks)
        
        # Make predictions
        if word_builder.should_predict():
            if features is not None:
                letter, confidence = model_manager.predict(features, CLASS_TO_CHAR)
                word_builder.update_prediction(letter, confidence)
                prediction_text = f"Detected: {letter} ({confidence:.2f})"
            else:
                word_builder.update_prediction("", 0.0)
                prediction_text = "No hand detected"
        
        # Try to add letter
        added_letter = word_builder.try_add_letter()
        if added_letter:
            print(f"[Added] '{added_letter}' → Word: '{word_builder.get_word()}'")
        
        # Draw UI overlay
        hand_count = len(hand_landmarks) if hand_landmarks else 0
        draw_ui_overlay(frame, word_builder.get_state(), prediction_text, hand_count)
        
        # Store the processed frame
        with frame_lock:
            current_frame = frame.copy()
        
        # Control frame rate
        time.sleep(0.03)  # ~30 FPS
    
    # Cleanup
    webcam.release()
    print("📷 Webcam closed")


# ===================================================================
# VIDEO STREAMING GENERATOR
# ===================================================================

def generate_frames():
    """
    Generator that yields video frames as JPEG for streaming
    """
    global current_frame
    
    while webcam_active:
        with frame_lock:
            if current_frame is None:
                continue
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)


# ===================================================================
# API ENDPOINTS
# ===================================================================

@app.get("/")
async def root():
    """API info"""
    return {
        "message": "Hand Gesture Recognition API",
        "webcam_active": webcam_active,
        "frontend": "Open /static/index.html to use the web interface"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": all([model_manager, hand_detector, word_builder]),
        "webcam_active": webcam_active
    }


@app.post("/start_webcam")
async def start_webcam():
    """Start the backend webcam"""
    global webcam_active
    
    if webcam_active:
        return {"success": False, "message": "Webcam already running"}
    
    print("🎬 Starting webcam thread...")
    webcam_active = True
    
    # Start webcam in background thread
    thread = threading.Thread(target=process_webcam, daemon=True)
    thread.start()
    
    # Give it a moment to start
    time.sleep(1)
    
    print(f"✅ Webcam thread started. Active: {webcam_active}")
    return {"success": True, "message": "Webcam started"}


@app.post("/stop_webcam")
async def stop_webcam():
    """Stop the backend webcam"""
    global webcam_active
    
    webcam_active = False
    time.sleep(0.5)  # Give thread time to stop
    
    return {"success": True, "message": "Webcam stopped"}


@app.get("/video_feed")
async def video_feed():
    """Stream video frames to frontend"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/get_word")
async def get_word():
    """Get the current composed word"""
    return {"word": word_builder.get_word()}


@app.post("/add_space")
async def add_space():
    """Add a space to the word"""
    word_builder.add_space()
    return {"success": True, "word": word_builder.get_word()}


@app.post("/delete_char")
async def delete_char():
    """Delete last character"""
    word_builder.delete_last_character()
    return {"success": True, "word": word_builder.get_word()}


@app.post("/clear_word")
async def clear_word():
    """Clear the entire word"""
    old_word = word_builder.clear_word()
    return {"success": True, "old_word": old_word, "word": word_builder.get_word()}


# ===================================================================
# RUN THE SERVER
# ===================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Don't reload with webcam
    )