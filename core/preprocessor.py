"""
Data preprocessing utilities
"""
import pickle
import os
import numpy as np
from config.settings import Config

class Preprocessor:
    """Handles data preprocessing - scaling and label encoding"""
    
    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.load_preprocessors()
    
    def load_preprocessors(self):
        """Load scaler and label encoder"""
        try:
            self._load_scaler()
            self._load_label_encoder()
        except Exception as e:
            raise RuntimeError(f"Failed to load preprocessors: {e}")
    
    def _load_scaler(self):
        """Load the feature scaler"""
        if not os.path.exists(Config.SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found: {Config.SCALER_PATH}")
        
        with open(Config.SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        print("✓ Scaler loaded")
    
    def _load_label_encoder(self):
        """Load the label encoder"""
        if not os.path.exists(Config.LABEL_ENCODER_PATH):
            raise FileNotFoundError(f"Label encoder not found: {Config.LABEL_ENCODER_PATH}")
        
        with open(Config.LABEL_ENCODER_PATH, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print("✓ Label encoder loaded")
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmark coordinates"""
        landmarks = landmarks.reshape(1, -1)
        normalized = self.scaler.transform(landmarks)
        return normalized.astype(np.float32)
    
    def decode_prediction(self, prediction_class):
        """Convert prediction class to gesture name"""
        return self.label_encoder.inverse_transform([prediction_class])[0]
    
    @property
    def gesture_classes(self):
        """Get available gesture classes"""
        return self.label_encoder.classes_ if self.label_encoder else []
    
    @property
    def num_classes(self):
        """Get number of gesture classes"""
        return len(self.gesture_classes)