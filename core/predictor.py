"""
Gesture prediction with smoothing
"""
import numpy as np
from collections import Counter, deque
from config.settings import Config

class GesturePredictor:
    """Handles gesture prediction with smoothing and confidence filtering"""
    
    def __init__(self, model_manager, preprocessor, confidence_threshold=None):
        self.model_manager = model_manager
        self.preprocessor = preprocessor
        self.confidence_threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD
        
        # Prediction history for smoothing
        self.prediction_history = deque(maxlen=Config.PREDICTION_HISTORY_SIZE)
        self.confidence_history = deque(maxlen=Config.PREDICTION_HISTORY_SIZE)
        
        # Statistics
        self.total_predictions = 0
        self.high_confidence_predictions = 0
    
    def predict(self, landmarks, debug_mode=False):
        """Make prediction with preprocessing and smoothing"""
        try:
            # Debug landmark range
            if debug_mode:
                stats = self._get_landmark_stats(landmarks)
                print(f"Raw landmarks: min={stats['min']:.1f}, max={stats['max']:.1f}")
            
            # Preprocess landmarks
            normalized_landmarks = self.preprocessor.normalize_landmarks(landmarks)
            
            if debug_mode:
                norm_stats = self._get_landmark_stats(normalized_landmarks.flatten())
                print(f"Normalized: min={norm_stats['min']:.3f}, max={norm_stats['max']:.3f}")
            
            # Get model prediction
            predictions = self.model_manager.predict(normalized_landmarks)
            
            # Extract results
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            gesture_name = self.preprocessor.decode_prediction(predicted_class)
            
            # Update statistics
            self.total_predictions += 1
            if confidence > self.confidence_threshold:
                self.high_confidence_predictions += 1
            
            # Apply smoothing
            smoothed_gesture = self._smooth_prediction(gesture_name, confidence)
            
            if debug_mode:
                print(f"Raw prediction: {gesture_name} ({confidence:.3f})")
                print(f"Smoothed: {smoothed_gesture}")
                print(f"All probabilities: {predictions[0]}")
            
            return {
                'gesture': smoothed_gesture,
                'raw_gesture': gesture_name,
                'confidence': confidence,
                'probabilities': predictions[0],
                'is_high_confidence': confidence > self.confidence_threshold
            }
            
        except Exception as e:
            return {
                'gesture': 'Error',
                'raw_gesture': 'Error',
                'confidence': 0.0,
                'probabilities': None,
                'error': str(e),
                'is_high_confidence': False
            }
    
    def _smooth_prediction(self, gesture, confidence):
        """Apply smoothing to predictions"""
        # Add to history
        self.prediction_history.append(gesture)
        self.confidence_history.append(confidence)
        
        # If not enough history, return raw prediction
        if len(self.prediction_history) < Config.SMOOTHING_WINDOW:
            return gesture if confidence > self.confidence_threshold else "Low Confidence"
        
        # Get recent high-confidence predictions
        recent_predictions = []
        for i in range(-Config.SMOOTHING_WINDOW, 0):
            if self.confidence_history[i] > self.confidence_threshold:
                recent_predictions.append(self.prediction_history[i])
        
        # If we have enough high-confidence predictions, use voting
        if len(recent_predictions) >= 2:
            most_common = Counter(recent_predictions).most_common(1)
            return most_common[0][0] if most_common else gesture
        
        # Return current prediction if confident, otherwise low confidence
        return gesture if confidence > self.confidence_threshold else "Low Confidence"
    
    def _get_landmark_stats(self, landmarks):
        """Get landmark statistics for debugging"""
        return {
            'min': float(landmarks.min()),
            'max': float(landmarks.max()),
            'mean': float(landmarks.mean()),
            'std': float(landmarks.std())
        }
    
    def clear_history(self):
        """Clear prediction history"""
        self.prediction_history.clear()
        self.confidence_history.clear()
        print("Prediction history cleared")
    
    def get_prediction_stats(self):
        """Get prediction statistics"""
        confidence_rate = (
            self.high_confidence_predictions / self.total_predictions * 100
            if self.total_predictions > 0 else 0
        )
        
        return {
            'total_predictions': self.total_predictions,
            'high_confidence_predictions': self.high_confidence_predictions,
            'confidence_rate': confidence_rate,
            'confidence_threshold': self.confidence_threshold
        }
    
    def update_confidence_threshold(self, new_threshold):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
        print(f"Confidence threshold updated to: {self.confidence_threshold:.3f}")