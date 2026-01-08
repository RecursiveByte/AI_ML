"""
Functions for drawing stuff on frames - useful for debugging and demos
"""

import cv2
import mediapipe as mp


def draw_hand_landmarks(frame, hand_landmarks_list):
    """
    Draw the hand skeleton on the frame
    
    Args:
        frame: image to draw on
        hand_landmarks_list: list of hand landmarks from MediaPipe
    """
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    
    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )


def draw_ui_overlay(frame, word_builder_state, prediction_text: str, hand_count: int):
    """
    Draw a compact UI overlay showing current status
    """
    height, width = frame.shape[:2]

    # Smaller black background at the top
    cv2.rectangle(frame, (0, 0), (width, 95), (0, 0, 0), -1)

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Current prediction
    cv2.putText(frame, prediction_text, (10, 22),
                font, 0.55, (0, 255, 0), 2)

    # Countdown timer
    countdown_text = f"Next add: {word_builder_state.time_until_next_add:.1f}s"
    cv2.putText(frame, countdown_text, (10, 45),
                font, 0.45, (255, 255, 0), 1)

    # Hand count
    cv2.putText(frame, f"Hands: {hand_count}", (10, 65),
                font, 0.45, (255, 255, 255), 1)

    # Current word
    word_display = word_builder_state.composed_word if word_builder_state.composed_word else "[empty]"
    cv2.putText(frame, f"Word: {word_display}", (10, 88),
                font, 0.55, (0, 255, 255), 2)



