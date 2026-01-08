"""
Turns hand landmarks into a feature vector that the model can understand
"""

import numpy as np
from typing import Tuple, Optional, List


def extract_hand_features(detection_result) -> Tuple[Optional[np.ndarray], Optional[List]]:
    """
    Convert MediaPipe hand landmarks into a standardized feature vector
    
    The key trick here is sorting hands by left/right so the model always
    gets the same hand in the same position. Otherwise it gets confused
    when your hands cross or one moves in front of the other.
    
    Args:
        detection_result: output from MediaPipe hand detection
        
    Returns:
        features: numpy array (1, 127) with all the landmark data
        sorted_hands: list of hand landmarks in consistent order
        Returns (None, None) if no hands are found
    """
    # Start with a blank feature vector
    features = np.full(127, -1.0, dtype=np.float32)
    
    # If no hands were detected, return None
    if not detection_result.multi_hand_landmarks:
        return None, None
    
    hand_landmarks_list = detection_result.multi_hand_landmarks
    hand_chirality_list = detection_result.multi_handedness
    
    # Pair each hand with its label (Left or Right)
    hands_with_label = []
    for i, hand in enumerate(hand_landmarks_list):
        label = hand_chirality_list[i].classification[0].label
        hands_with_label.append((label, hand))
    
    # Sort them alphabetically - Left before Right
    # This keeps everything consistent
    hands_with_label.sort(key=lambda x: x[0])
    sorted_hands = [h[1] for h in hands_with_label]
    
    # First feature: are we using one hand or two?
    features[0] = 1.0 if len(sorted_hands) == 2 else 0.0
    
    # Now fill in all the landmark positions
    idx = 1
    for hand_idx in range(2):  # we always reserve space for 2 hands
        if hand_idx < len(sorted_hands):
            # We found this hand, so save all 21 landmarks
            for landmark in sorted_hands[hand_idx].landmark:
                features[idx] = landmark.x      # x position
                features[idx + 1] = landmark.y  # y position
                features[idx + 2] = landmark.z  # z depth
                idx += 3
        else:
            # This hand wasn't detected, skip its space
            idx += 63  # 21 landmarks × 3 coordinates
    
    return features.reshape(1, -1), sorted_hands