import cv2
from utils.config import *
from utils.model_manager import GestureModelManager
from utils.hand_detector import HandDetector
from utils.feature_extractor import extract_hand_features
from utils.word_builder import WordBuilder
from utils.visualization import draw_hand_landmarks, draw_ui_overlay


def main():
    """Run the gesture recognition with a webcam"""
    
    # Initialize all our components
    model_manager = GestureModelManager(MODEL_PATH, SCALER_PATH)
    hand_detector = HandDetector(MAX_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE)
    word_builder = WordBuilder(PREDICTION_INTERVAL, LETTER_ADD_INTERVAL, CONFIDENCE_THRESHOLD)
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    print("=" * 60)
    print("Hand Gesture Word Builder")
    print("=" * 60)
    print("Instructions:")
    print("  Show a gesture and hold it steady")
    print("  Letter adds after 5 seconds")
    print("  SPACE = add space")
    print("  BACKSPACE = delete last letter")
    print("  ENTER = clear word")
    print("  ESC = quit")
    print("=" * 60)
    
    prediction_text = "Show hand gesture..."
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        detection_result = hand_detector.detect(frame)
        features, hand_landmarks = extract_hand_features(detection_result)
        
        # Draw the hand landmarks
        draw_hand_landmarks(frame, hand_landmarks)
        
        # Make predictions at regular intervals
        if word_builder.should_predict():
            if features is not None:
                letter, confidence = model_manager.predict(features, CLASS_TO_CHAR)
                word_builder.update_prediction(letter, confidence)
                prediction_text = f"Detected: {letter} ({confidence:.2f})"
            else:
                word_builder.update_prediction("", 0.0)
                prediction_text = "No hand detected"
        
        # Try to add letter if enough time has passed
        added_letter = word_builder.try_add_letter()
        if added_letter:
            print(f"[Added] '{added_letter}' → Word: '{word_builder.get_word()}'")
        
        # Draw the UI
        hand_count = len(hand_landmarks) if hand_landmarks else 0
        draw_ui_overlay(frame, word_builder.get_state(), prediction_text, hand_count)
        
        cv2.imshow("Hand Gesture Word Builder", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            word_builder.add_space()
            print(f"[Space] Word: '{word_builder.get_word()}'")
        elif key == 8:  # BACKSPACE
            word_builder.delete_last_character()
            print(f"[Backspace] Word: '{word_builder.get_word()}'")
        elif key == 13:  # ENTER
            final_word = word_builder.clear_word()
            print(f"[Clear] Final word was: '{final_word}'")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()
    
    print("=" * 60)
    print(f"Final word: '{word_builder.get_word()}'")
    print("Application closed.")
    print("=" * 60)


if __name__ == "__main__":
    main()