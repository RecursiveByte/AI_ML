"""
FastAPI Backend for Hand Gesture Recognition
Works with React frontend that sends webcam frames

Run with: python app.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from typing import Optional
import base64
from io import BytesIO
from PIL import Image

# Import our utility modules
from utils.config import *
from utils.model_manager import GestureModelManager
from utils.hand_detector import HandDetector
from utils.feature_extractor import extract_hand_features
from utils.word_builder import WordBuilder


# ===================================================================
# FASTAPI APP SETUP
# ===================================================================

app = FastAPI(
    title="Hand Gesture Recognition API",
    description="Real-time hand gesture recognition - Works with React frontend",
    version="2.0.0"
)

# CORS - Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
# GLOBAL INSTANCES
# ===================================================================

model_manager = None
hand_detector = None
word_builder = None


@app.on_event("startup")
async def startup_event():
    """Initialize models when the server starts"""
    global model_manager, hand_detector, word_builder
    
    print("\n" + "=" * 60)
    print("🚀 Hand Gesture Recognition API Starting...")
    print("=" * 60)
    
    try:
        print("Loading ML models...")
        model_manager = GestureModelManager(MODEL_PATH, SCALER_PATH)
        hand_detector = HandDetector(MAX_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE)
        word_builder = WordBuilder(PREDICTION_INTERVAL, LETTER_ADD_INTERVAL, CONFIDENCE_THRESHOLD)
        
        print("✅ All models loaded successfully!")
        print("\n📡 API Server Info:")
        print(f"   Backend API: http://localhost:8000")
        print(f"   API Docs: http://localhost:8000/docs")
        print(f"   React Frontend: http://localhost:3000 (run 'npm start' in frontend folder)")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up when server shuts down"""
    global hand_detector
    
    if hand_detector:
        hand_detector.close()
    
    print("👋 Server shutdown complete")


# ===================================================================
# REQUEST/RESPONSE MODELS
# ===================================================================

class ImageRequest(BaseModel):
    """Request format for base64 encoded images from React frontend"""
    image: str  # base64 encoded image


class PredictionResponse(BaseModel):
    """Response for gesture prediction"""
    success: bool
    current_letter: str
    confidence: float
    hand_count: int
    current_word: str
    time_until_next_add: float
    letter_added: Optional[str] = None
    message: Optional[str] = None


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Convert base64 string from React to OpenCV image
    
    Args:
        base64_string: base64 encoded image (may have data URL prefix)
        
    Returns:
        OpenCV image (numpy array in BGR format)
    """
    try:
        # Remove the data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format (BGR)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        print(f"Error decoding image: {e}")
        raise


# ===================================================================
# API ENDPOINTS
# ===================================================================

@app.get("/")
async def root():
    """API info"""
    return {
        "status": "online",
        "message": "Hand Gesture Recognition API",
        "version": "2.0.0",
        "mode": "React Frontend Mode",
        "endpoints": {
            "predict": "/api/predict",
            "word_operations": ["/api/word/space", "/api/word/delete", "/api/word/clear", "/api/word/reset"]
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_ready = all([model_manager, hand_detector, word_builder])
    
    return {
        "status": "healthy" if models_ready else "unhealthy",
        "models_loaded": models_ready,
        "word_builder_state": {
            "current_word": word_builder.get_word() if word_builder else "",
            "current_letter": word_builder.current_letter if word_builder else "",
        }
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_gesture(request: ImageRequest):
    """
    Main prediction endpoint
    React sends webcam frame, backend processes and returns prediction
    
    This endpoint:
    1. Receives base64 image from React
    2. Detects hands and extracts features
    3. Makes prediction
    4. Updates word builder
    5. Automatically adds letter if 5 seconds passed
    6. Returns current state
    """
    try:
        # Decode the image from React
        frame = decode_base64_image(request.image)
        
        # Detect hands
        detection_result = hand_detector.detect(frame)
        features, hand_landmarks = extract_hand_features(detection_result)
        
        # Update prediction if it's time
        if word_builder.should_predict():
            if features is not None:
                letter, confidence = model_manager.predict(features, CLASS_TO_CHAR)
                word_builder.update_prediction(letter, confidence)
            else:
                word_builder.update_prediction("", 0.0)
        
        # Try to add letter if enough time has passed
        added_letter = word_builder.try_add_letter()
        
        if added_letter:
            print(f"[Added] '{added_letter}' → Word: '{word_builder.get_word()}'")
        
        # Get current state
        state = word_builder.get_state()
        hand_count = len(hand_landmarks) if hand_landmarks else 0
        
        return PredictionResponse(
            success=True,
            current_letter=state.current_letter or "",
            confidence=round(state.current_confidence, 3),
            hand_count=hand_count,
            current_word=state.composed_word or "",
            time_until_next_add=round(state.time_until_next_add, 1),
            letter_added=added_letter,
            message="Prediction successful"
        )
    
    except Exception as e:
        print(f"Error in predict_gesture: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/word/space")
async def add_space():
    """Add a space to the current word"""
    try:
        word_builder.add_space()
        return {
            "success": True,
            "current_word": word_builder.get_word(),
            "message": "Space added"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add space: {str(e)}")


@app.post("/api/word/delete")
async def delete_last_character():
    """Delete the last character from the word"""
    try:
        word_builder.delete_last_character()
        return {
            "success": True,
            "current_word": word_builder.get_word(),
            "message": "Character deleted"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")


@app.post("/api/word/clear")
async def clear_word():
    """Clear the current word"""
    try:
        old_word = word_builder.clear_word()
        return {
            "success": True,
            "previous_word": old_word,
            "current_word": "",
            "message": "Word cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear: {str(e)}")


@app.post("/api/word/reset")
async def reset_word_builder():
    """Reset the word builder to initial state"""
    try:
        word_builder.reset()
        return {
            "success": True,
            "message": "Word builder reset successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset: {str(e)}")


@app.get("/api/word/get")
async def get_current_word():
    """Get the current word"""
    try:
        return {
            "success": True,
            "current_word": word_builder.get_word()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get word: {str(e)}")


# ===================================================================
# RUN THE SERVER
# ===================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes
    )
