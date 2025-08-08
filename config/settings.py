"""
Configuration settings for Hand Gesture Recognition
"""
import os

class Config:
    """Main configuration class"""
    
    # Model paths
    BASE_MODEL_DIR = "model/neural_network"
    KERAS_MODEL_PATH = os.path.join(BASE_MODEL_DIR, "checkpoints/hand_gesture_model.h5")
    TFLITE_MODEL_PATH = os.path.join(BASE_MODEL_DIR, "hand_gesture_model.tflite")
    SCALER_PATH = os.path.join(BASE_MODEL_DIR, "scaler.pkl")
    LABEL_ENCODER_PATH = os.path.join(BASE_MODEL_DIR, "label_encoder.pkl")
    
    # MediaPipe settings
    MAX_NUM_HANDS = 1
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Recognition settings
    CONFIDENCE_THRESHOLD = 0.5
    PREDICTION_HISTORY_SIZE = 5
    SMOOTHING_WINDOW = 3
    
    # Camera settings
    CAMERA_INDEX = 0
    FLIP_HORIZONTAL = True
    
    # UI settings
    PANEL_WIDTH = 300
    PANEL_HEIGHT = 200
    PANEL_MARGIN = 10
    
    # Logging
    LOG_DIR = "logs"
    STATS_ENABLED = True
    
    # Input size (21 landmarks × 2 coordinates)
    INPUT_SIZE = 42
    NUM_LANDMARKS = 21

class DebugConfig:
    """Debug configuration"""
    
    ENABLE_LANDMARK_RANGE_DEBUG = True
    ENABLE_PREDICTION_DEBUG = True
    ENABLE_FPS_DEBUG = True
    PRINT_PREDICTIONS = False