"""
Hand landmark extraction utilities
"""
import numpy as np
import mediapipe as mp
from config.settings import Config

class LandmarkExtractor:
    """Extracts and processes hand landmarks"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=Config.MAX_NUM_HANDS,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
        )
    
    def extract_from_image(self, rgb_image):
        """Extract landmarks from RGB image"""
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None, None
        
        # Get first hand (we only process one hand)
        hand_landmarks = results.multi_hand_landmarks[0]
        return results, hand_landmarks
    
    def landmarks_to_coordinates(self, hand_landmarks, image_width, image_height):
        """Convert landmarks to pixel coordinates"""
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            # Convert normalized coordinates (0-1) to pixel coordinates
            pixel_x = landmark.x * image_width
            pixel_y = landmark.y * image_height
            landmarks.extend([pixel_x, pixel_y])
        
        return np.array(landmarks)
    
    def validate_landmarks(self, landmarks):
        """Validate landmark array"""
        return (
            landmarks is not None and 
            len(landmarks) == Config.INPUT_SIZE and
            not np.any(np.isnan(landmarks)) and
            not np.any(np.isinf(landmarks))
        )
    
    def get_landmark_statistics(self, landmarks):
        """Get basic statistics of landmarks"""
        if landmarks is None or len(landmarks) == 0:
            return None
        
        return {
            'min': float(landmarks.min()),
            'max': float(landmarks.max()),
            'mean': float(landmarks.mean()),
            'std': float(landmarks.std())
        }
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'hands'):
            self.hands.close()