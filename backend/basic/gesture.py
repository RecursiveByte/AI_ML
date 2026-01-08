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
# WORD BUILDER - ADDS LETTER EVERY 5 SECONDS
# ===============================
cap = cv2.VideoCapture(0)

last_prediction_time = time.time()
last_letter_add_time = time.time()
prediction_text = "Show hand gesture..."
current_letter = ""
current_confidence = 0.0
composed_word = ""

# Timer settings
PREDICTION_INTERVAL = 1.0  # Predict every 1 second
LETTER_ADD_INTERVAL = 5.0  # Add letter to word every 5 seconds

print("=" * 60)
print("Hand Gesture Word Builder")
print("=" * 60)
print("Instructions:")
print("  - Show a hand gesture")
print("  - Hold the gesture steady")
print("  - After 5 seconds, the detected letter will be added to your word")
print("  - Press SPACE to add a space")
print("  - Press BACKSPACE to delete last letter")
print("  - Press ENTER to clear the word")
print("  - Press ESC to quit")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror effect
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
    if current_time - last_prediction_time >= PREDICTION_INTERVAL:
        if features is not None:
            features_scaled = scaler.transform(features)
            probs = model.predict(features_scaled, verbose=0)
            pred_class = np.argmax(probs)
            confidence = np.max(probs)

            current_letter = CLASS_TO_CHAR[pred_class]
            current_confidence = confidence
            prediction_text = f"Detected: {current_letter} ({confidence:.2f})"
        else:
            current_letter = ""
            current_confidence = 0.0
            prediction_text = "No hand detected"

        last_prediction_time = current_time

    # ⏱️ Add letter to word every 5 seconds
    if current_time - last_letter_add_time >= LETTER_ADD_INTERVAL:
        if current_letter and current_confidence > 0.5:  # Only add if confident
            composed_word += current_letter
            print(f"[Added] '{current_letter}' → Word: '{composed_word}'")
        last_letter_add_time = current_time

    # Calculate countdown for next letter addition
    time_until_add = LETTER_ADD_INTERVAL - (current_time - last_letter_add_time)
    countdown_text = f"Next add in: {time_until_add:.1f}s"

    # ===============================
    # DISPLAY UI
    # ===============================
    frame_height, frame_width = frame.shape[:2]

    # Background for top section
    cv2.rectangle(frame, (0, 0), (frame_width, 150), (0, 0, 0), -1)

    # Current prediction
    cv2.putText(frame, prediction_text, (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Countdown timer
    cv2.putText(frame, countdown_text, (15, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Hand count
    hand_count = len(hand_landmarks) if hand_landmarks else 0
    cv2.putText(frame, f"Hands detected: {hand_count}", (15, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Composed word (large and prominent)
    word_display = composed_word if composed_word else "[empty]"
    cv2.putText(frame, f"Word: {word_display}", (15, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Hand Gesture Word Builder", frame)

    # ===============================
    # KEYBOARD CONTROLS
    # ===============================
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC to quit
        break
    elif key == 32:  # SPACE to add space
        composed_word += " "
        print(f"[Space] Word: '{composed_word}'")
    elif key == 8:  # BACKSPACE to delete last character
        if composed_word:
            composed_word = composed_word[:-1]
            print(f"[Backspace] Word: '{composed_word}'")
    elif key == 13:  # ENTER to clear word
        print(f"[Clear] Final word was: '{composed_word}'")
        composed_word = ""


cap.release()
cv2.destroyAllWindows()

print("=" * 60)
print(f"Final composed word: '{composed_word}'")
print("Application closed.")
print("=" * 60)