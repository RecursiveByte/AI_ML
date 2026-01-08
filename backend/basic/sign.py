# ===============================
# IMPORTS
# ===============================
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib
import time


# ===============================
# LOAD MODEL & SCALER
# ===============================
MODEL_PATH = "hand_landmark_dnn.keras"
SCALER_PATH = "scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ===============================
# FEATURE EXTRACTION (CHIRALITY-BASED)
# ===============================
def extract_hand_features(frame):
    """
    Extract hand features with consistent left/right hand ordering.
    This prevents gesture confusion when hands overlap or change depth.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    features = np.full(127, -1.0, dtype=np.float32)

    if not result.multi_hand_landmarks:
        return None, None

    # -------------------------------
    # SORT BY CHIRALITY (LEFT/RIGHT)
    # -------------------------------
    hand_landmarks_list = result.multi_hand_landmarks
    hand_chirality_list = result.multi_handedness

    # Create pairs of (chirality_label, landmarks)
    hands_with_label = []
    for i, hand in enumerate(hand_landmarks_list):
        # Get hand label: "Left" or "Right"
        label = hand_chirality_list[i].classification[0].label
        hands_with_label.append((label, hand))

    # Sort alphabetically: "Left" comes before "Right"
    # This ensures consistent ordering regardless of detection order
    hands_with_label.sort(key=lambda x: x[0])
    
    sorted_hands = [h[1] for h in hands_with_label]

    # Feature 0: uses_two_hands flag
    features[0] = 1.0 if len(sorted_hands) == 2 else 0.0

    # Fill in landmark coordinates
    idx = 1
    for h in range(2):
        if h < len(sorted_hands):
            # Extract x, y, z for all 21 landmarks
            for lm in sorted_hands[h].landmark:
                features[idx]     = lm.x
                features[idx + 1] = lm.y
                features[idx + 2] = lm.z
                idx += 3
        else:
            # Skip 63 positions (21 landmarks × 3 coords) for missing hand
            idx += 63

    return features.reshape(1, -1), sorted_hands


# ===============================
# CLASS TO LETTER MAP
# ===============================
CLASS_TO_CHAR = {i: chr(ord('A') + i) for i in range(26)}


# ===============================
# WEBCAM LOOP (PREDICT EVERY 1 SEC)
# ===============================
cap = cv2.VideoCapture(0)

last_prediction_time = time.time()
prediction_text = "Show hand... (predicts every 1 sec)"

print("Starting Hand Gesture Recognition...")
print("Press ESC to quit")
print("Using chirality-based hand sorting (Left/Right)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror effect (optional)
    frame = cv2.flip(frame, 1)

    features, hand_landmarks = extract_hand_features(frame)

    # Draw landmarks on detected hands
    if hand_landmarks:
        for hand in hand_landmarks:
            mp_draw.draw_landmarks(
                frame, 
                hand, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

    current_time = time.time()

    # 🔁 Predict every 1 second
    if current_time - last_prediction_time >= 1:
        if features is not None:
            features_scaled = scaler.transform(features)
            probs = model.predict(features_scaled, verbose=0)
            pred_class = np.argmax(probs)
            confidence = np.max(probs)

            prediction_text = f"Prediction: {CLASS_TO_CHAR[pred_class]} (Confidence: {confidence:.2f})"
            
            # Optional: Print to console for debugging
            print(f"[{time.strftime('%H:%M:%S')}] {prediction_text}")
        else:
            prediction_text = "No hand detected"
            print(f"[{time.strftime('%H:%M:%S')}] No hand detected")

        last_prediction_time = current_time  # Reset timer

    # Display prediction text with background for better visibility
    text_size = cv2.getTextSize(prediction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(frame, (10, 10), (text_size[0] + 20, 50), (0, 0, 0), -1)
    cv2.putText(
        frame,
        prediction_text,
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Display hand count
    hand_count = len(hand_landmarks) if hand_landmarks else 0
    cv2.putText(
        frame,
        f"Hands: {hand_count}",
        (15, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )

    cv2.imshow("Hand Gesture Recognition (Two-Hand Support)", frame)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
print("Application closed.")