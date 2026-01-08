"""
Handles all the machine learning stuff - loading models and making predictions
"""

import numpy as np
from tensorflow.keras.models import load_model
import joblib
from typing import Tuple


class GestureModelManager:
    """Loads and manages the trained gesture recognition model"""
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Load the model and scaler from files
        
        Args:
            model_path: where your Keras model is saved
            scaler_path: where your scaler pickle file is saved
        """
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        
        print(f"Loading scaler from: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        print("Models loaded successfully!")
    
    def predict(self, features: np.ndarray, class_to_char: dict) -> Tuple[str, float]:
        """
        Take raw features and predict which letter it is
        
        Args:
            features: numpy array of hand landmark features
            class_to_char: dictionary mapping class indices to letters
            
        Returns:
            (predicted_letter, confidence_score)
        """
        # Normalize the features using our saved scaler
        features_scaled = self.scaler.transform(features)
        
        # Get prediction probabilities from the model
        probabilities = self.model.predict(features_scaled, verbose=0)
        
        # Find which class has the highest probability
        predicted_class = np.argmax(probabilities)
        confidence = np.max(probabilities)
        
        # Convert the class number to a letter
        letter = class_to_char[predicted_class]
        
        return letter, float(confidence)