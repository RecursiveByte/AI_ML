# Model file paths
MODEL_PATH = 'models/hand_landmark_dnn.keras'
SCALER_PATH = 'models/scaler.pkl'

# MediaPipe settings
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Timing settings
PREDICTION_INTERVAL = 1.0      # how often to make predictions (seconds)
LETTER_ADD_INTERVAL = 5.0      # how long to hold gesture before adding letter
CONFIDENCE_THRESHOLD = 0.5     # minimum confidence to accept a prediction

# Feature dimensions
FEATURE_SIZE = 127  # total feature vector size

# Map numbers to letters (0=A, 1=B, etc.)
CLASS_TO_CHAR = {i: chr(ord('A') + i) for i in range(26)}