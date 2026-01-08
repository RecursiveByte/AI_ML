"""
Wraps MediaPipe hand detection in a simple class
"""

import cv2
import mediapipe as mp


class HandDetector:
    """Uses MediaPipe to find hands in images"""
    
    def __init__(self, 
                 max_hands: int = 2, 
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        """
        Set up the hand detector
        
        Args:
            max_hands: how many hands to look for at once
            min_detection_confidence: how sure it needs to be to detect a hand
            min_tracking_confidence: how sure it needs to be while tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect(self, frame):
        """
        Find hands in a video frame
        
        Args:
            frame: image from your webcam (BGR format from OpenCV)
            
        Returns:
            MediaPipe results with hand landmarks
        """
        # MediaPipe needs RGB, but OpenCV gives us BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(frame_rgb)
    
    def close(self):
        """Clean up when you're done"""
        self.hands.close()
