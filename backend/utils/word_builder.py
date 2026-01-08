"""
Manages the logic of building words from detected hand gestures
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class WordBuilderState:
    """Just a container to hold the current state"""
    composed_word: str
    current_letter: str
    current_confidence: float
    last_prediction_time: float
    last_letter_add_time: float
    time_until_next_add: float


class WordBuilder:
    """Handles the word building process with timing logic"""
    
    def __init__(self, 
                 prediction_interval: float = 1.0,
                 letter_add_interval: float = 5.0,
                 confidence_threshold: float = 0.5):
        """
        Set up the word builder
        
        Args:
            prediction_interval: how often to check the gesture (seconds)
            letter_add_interval: how long to hold gesture before adding (seconds)
            confidence_threshold: minimum confidence needed to add a letter
        """
        self.prediction_interval = prediction_interval
        self.letter_add_interval = letter_add_interval
        self.confidence_threshold = confidence_threshold
        
        self.reset()
    
    def reset(self):
        """Start fresh with an empty word"""
        self.composed_word = ""
        self.current_letter = ""
        self.current_confidence = 0.0
        self.last_prediction_time = time.time()
        self.last_letter_add_time = time.time()
    
    def get_state(self) -> WordBuilderState:
        """Get all the current info as a nice package"""
        current_time = time.time()
        time_until_add = max(0, self.letter_add_interval - (current_time - self.last_letter_add_time))
        
        return WordBuilderState(
            composed_word=self.composed_word,
            current_letter=self.current_letter,
            current_confidence=self.current_confidence,
            last_prediction_time=self.last_prediction_time,
            last_letter_add_time=self.last_letter_add_time,
            time_until_next_add=time_until_add
        )
    
    def should_predict(self) -> bool:
        """Is it time to make a new prediction?"""
        elapsed = time.time() - self.last_prediction_time
        return elapsed >= self.prediction_interval
    
    def should_add_letter(self) -> bool:
        """Has the gesture been held long enough to add the letter?"""
        elapsed = time.time() - self.last_letter_add_time
        return elapsed >= self.letter_add_interval
    
    def update_prediction(self, letter: str, confidence: float):
        """
        Update what gesture we're currently seeing
        
        Args:
            letter: the detected letter
            confidence: how confident the model is (0 to 1)
        """
        self.current_letter = letter
        self.current_confidence = confidence
        self.last_prediction_time = time.time()
    
    def try_add_letter(self) -> Optional[str]:
        """
        Try to add the current letter to the word
        
        Returns:
            The letter that was added, or None if not added
        """
        if not self.should_add_letter():
            return None
        
        # Only add if we have a letter and confidence is high enough
        if self.current_letter and self.current_confidence >= self.confidence_threshold:
            self.composed_word += self.current_letter
            self.last_letter_add_time = time.time()
            return self.current_letter
        else:
            # Reset timer even if we didn't add anything
            self.last_letter_add_time = time.time()
            return None
    
    def add_space(self):
        """Add a space to the word"""
        self.composed_word += " "
    
    def delete_last_character(self):
        """Remove the last character from the word"""
        if self.composed_word:
            self.composed_word = self.composed_word[:-1]
    
    def clear_word(self) -> str:
        """
        Clear the word and return what it was
        
        Returns:
            The word before clearing
        """
        old_word = self.composed_word
        self.composed_word = ""
        return old_word
    
    def get_word(self) -> str:
        """Get the current word"""
        return self.composed_word