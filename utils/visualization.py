"""
Visualization utilities for the recognition app
"""
import cv2
import numpy as np
import mediapipe as mp
from config.settings import Config

class VisualizationManager:
    """Handles all visualization aspects of the app"""
    
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        
        # UI settings
        self.panel_width = Config.PANEL_WIDTH
        self.panel_height = Config.PANEL_HEIGHT
        self.panel_margin = Config.PANEL_MARGIN
        
        # Colors
        self.colors = {
            'background': (0, 0, 0),
            'border': (255, 255, 255),
            'text': (255, 255, 255),
            'high_confidence': (0, 255, 0),
            'low_confidence': (0, 255, 255),
            'error': (0, 0, 255),
            'debug': (255, 255, 0)
        }
    
    def draw_hand_landmarks(self, image, hand_landmarks):
        """Draw hand landmarks on image"""
        self.mp_drawing.draw_landmarks(
            image, 
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
    
    def draw_info_panel(self, image, prediction_result, fps, model_type, debug_info=None):
        """Draw comprehensive information panel"""
        height, width = image.shape[:2]
        
        # Calculate panel position
        panel_x = width - self.panel_width - self.panel_margin
        panel_y = self.panel_margin
        
        # Draw panel background
        self._draw_panel_background(image, panel_x, panel_y)
        
        # Draw content
        y_offset = panel_y + 25
        y_offset = self._draw_title(image, panel_x, y_offset)
        y_offset = self._draw_prediction_info(image, panel_x, y_offset, prediction_result)
        y_offset = self._draw_system_info(image, panel_x, y_offset, fps, model_type)
        y_offset = self._draw_probabilities(image, panel_x, y_offset, prediction_result)
        
        if debug_info:
            self._draw_debug_info(image, panel_x, y_offset, debug_info)
    
    def _draw_panel_background(self, image, panel_x, panel_y):
        """Draw semi-transparent panel background"""
        overlay = image.copy()
        cv2.rectangle(
            overlay, 
            (panel_x, panel_y), 
            (panel_x + self.panel_width, panel_y + self.panel_height), 
            self.colors['background'], 
            -1
        )
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw border
        cv2.rectangle(
            image, 
            (panel_x, panel_y), 
            (panel_x + self.panel_width, panel_y + self.panel_height), 
            self.colors['border'], 
            2
        )
    
    def _draw_title(self, image, panel_x, y_offset):
        """Draw panel title"""
        cv2.putText(
            image, 
            "Hand Gesture Recognition", 
            (panel_x + 10, y_offset), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            self.colors['text'], 
            2
        )
        return y_offset + 30
    
    def _draw_prediction_info(self, image, panel_x, y_offset, prediction_result):
        """Draw prediction information"""
        gesture = prediction_result.get('gesture', 'Unknown')
        confidence = prediction_result.get('confidence', 0.0)
        is_high_confidence = prediction_result.get('is_high_confidence', False)
        
        # Gesture name
        cv2.putText(
            image, 
            f"Gesture: {gesture}", 
            (panel_x + 10, y_offset), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            self.colors['high_confidence'], 
            2
        )
        y_offset += 25
        
        # Confidence
        conf_color = self.colors['high_confidence'] if is_high_confidence else self.colors['low_confidence']
        cv2.putText(
            image, 
            f"Confidence: {confidence:.3f}", 
            (panel_x + 10, y_offset), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            conf_color, 
            2
        )
        return y_offset + 25
    
    def _draw_system_info(self, image, panel_x, y_offset, fps, model_type):
        """Draw system information"""
        # Model type
        cv2.putText(
            image, 
            f"Model: {model_type}", 
            (panel_x + 10, y_offset), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            self.colors['text'], 
            1
        )
        y_offset += 20
        
        # FPS
        cv2.putText(
            image, 
            f"FPS: {fps}", 
            (panel_x + 10, y_offset), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            self.colors['text'], 
            1
        )
        return y_offset + 25
    
    def _draw_probabilities(self, image, panel_x, y_offset, prediction_result):
        """Draw class probabilities"""
        probabilities = prediction_result.get('probabilities')
        gesture_classes = prediction_result.get('gesture_classes', [])
        
        if probabilities is None or len(gesture_classes) == 0:
            return y_offset
        
        cv2.putText(
            image, 
            "Probabilities:", 
            (panel_x + 10, y_offset), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            self.colors['text'], 
            1
        )
        y_offset += 15
        
        max_idx = np.argmax(probabilities) if len(probabilities) > 0 else -1
        
        for i, class_name in enumerate(gesture_classes):
            if i < len(probabilities):
                prob_text = f"{class_name}: {probabilities[i]:.3f}"
                prob_color = self.colors['high_confidence'] if i == max_idx else (200, 200, 200)
                
                cv2.putText(
                    image, 
                    prob_text, 
                    (panel_x + 15, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.35, 
                    prob_color, 
                    1
                )
                y_offset += 15
        
        return y_offset
    
    def _draw_debug_info(self, image, panel_x, y_offset, debug_info):
        """Draw debug information"""
        if 'landmark_range' in debug_info:
            cv2.putText(
                image, 
                f"Raw range: {debug_info['landmark_range']}", 
                (panel_x + 10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.3, 
                self.colors['debug'], 
                1
            )
    
    def draw_help_text(self, image):
        """Draw help text overlay"""
        help_texts = [
            "Controls:",
            "q - Quit",
            "s - Save stats", 
            "c - Clear history",
            "t - Toggle model",
            "d - Debug mode",
            "h - Toggle help"
        ]
        
        y_start = 30
        for i, text in enumerate(help_texts):
            cv2.putText(
                image, 
                text, 
                (10, y_start + i * 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                self.colors['text'], 
                1
            )