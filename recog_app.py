"""
Clean Hand Gesture Recognition App
Refactored version with modular design
"""
import cv2
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import Config, DebugConfig
from core.model_manager import ModelManager
from core.preprocessor import Preprocessor
from core.predictor import GesturePredictor
from utils.landmark_extractor import LandmarkExtractor
from utils.visualization import VisualizationManager
from utils.statistics import StatisticsManager, FPSCounter

class HandGestureRecognitionApp:
    """Main application class - clean and modular"""
    
    def __init__(self, use_tflite=False, debug_mode=False):
        """Initialize the application"""
        print("🚀 Initializing Hand Gesture Recognition App...")
        
        self.debug_mode = debug_mode
        self.show_help = False
        
        try:
            # Initialize core components
            self.model_manager = ModelManager(use_tflite=use_tflite)
            self.preprocessor = Preprocessor()
            self.predictor = GesturePredictor(self.model_manager, self.preprocessor)
            
            # Initialize utilities
            self.landmark_extractor = LandmarkExtractor()
            self.visualizer = VisualizationManager()
            self.stats_manager = StatisticsManager()
            self.fps_counter = FPSCounter()
            
            print("✅ Application initialized successfully!")
            self._print_startup_info()
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    def _print_startup_info(self):
        """Print startup information"""
        print("\n" + "="*50)
        print("SYSTEM INFO")
        print("="*50)
        print(f"Model type: {self.model_manager.model_type}")
        print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        print(f"Gesture classes: {list(self.preprocessor.gesture_classes)}")
        print(f"Confidence threshold: {self.predictor.confidence_threshold:.3f}")
        print(f"Input size: {Config.INPUT_SIZE} features")
        print("="*50)
    
    def run(self):
        """Run the main application loop"""
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        print("\n🎥 Starting camera capture...")
        self._print_controls()
        
        self.stats_manager.start_session()
        self.fps_counter.start()
        
        try:
            while True:
                if not self._process_frame(cap):
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        
        finally:
            self._cleanup(cap)
    
    def _process_frame(self, cap):
        """Process a single frame"""
        ret, frame = cap.read()
        if not ret:
            return False
        
        # Flip frame for mirror effect
        if Config.FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)
        
        height, width, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract landmarks
        results, hand_landmarks = self.landmark_extractor.extract_from_image(rgb_frame)
        
        # Initialize prediction result
        prediction_result = {
            'gesture': 'No Hand',
            'confidence': 0.0,
            'probabilities': None,
            'gesture_classes': list(self.preprocessor.gesture_classes),
            'is_high_confidence': False
        }
        
        # Process if hand detected
        has_detection = hand_landmarks is not None
        
        if has_detection:
            # Draw landmarks
            self.visualizer.draw_hand_landmarks(frame, hand_landmarks)
            
            # Extract coordinates
            landmarks = self.landmark_extractor.landmarks_to_coordinates(
                hand_landmarks, width, height
            )
            
            # Validate and predict
            if self.landmark_extractor.validate_landmarks(landmarks):
                prediction_result = self.predictor.predict(landmarks, self.debug_mode)
                prediction_result['gesture_classes'] = list(self.preprocessor.gesture_classes)
        
        # Update statistics
        self.stats_manager.update_frame_stats(has_detection)
        self.fps_counter.update()
        
        # Draw UI
        self._draw_ui(frame, prediction_result)
        
        # Show frame
        cv2.imshow('Hand Gesture Recognition - Clean Version', frame)
        
        # Handle keyboard input
        return self._handle_keyboard_input()
    
    def _draw_ui(self, frame, prediction_result):
        """Draw user interface elements"""
        # Main info panel
        debug_info = None
        if self.debug_mode and 'gesture' in prediction_result:
            debug_info = {
                'landmark_range': 'Debug info here'  # You can expand this
            }
        
        self.visualizer.draw_info_panel(
            frame,
            prediction_result,
            self.fps_counter.get_fps(),
            self.model_manager.model_type,
            debug_info
        )
        
        # Help overlay
        if self.show_help:
            self.visualizer.draw_help_text(frame)
    
    def _handle_keyboard_input(self):
        """Handle keyboard input"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('s'):
            self._save_statistics()
        elif key == ord('c'):
            self.predictor.clear_history()
        elif key == ord('t'):
            self._toggle_model()
        elif key == ord('d'):
            self._toggle_debug()
        elif key == ord('h'):
            self._toggle_help()
        elif key == ord('+') or key == ord('='):
            self._adjust_confidence_threshold(0.05)
        elif key == ord('-'):
            self._adjust_confidence_threshold(-0.05)
        
        return True
    
    def _save_statistics(self):
        """Save session statistics"""
        additional_data = {
            'model_info': {
                'type': self.model_manager.model_type,
                'gesture_classes': list(self.preprocessor.gesture_classes)
            },
            'prediction_stats': self.predictor.get_prediction_stats()
        }
        
        self.stats_manager.save_session_stats(additional_data)
    
    def _toggle_model(self):
        """Toggle between model types"""
        try:
            new_type = self.model_manager.toggle_model_type()
            print(f"🔄 Switched to {new_type} model")
        except Exception as e:
            print(f"❌ Cannot toggle model: {e}")
    
    def _toggle_debug(self):
        """Toggle debug mode"""
        self.debug_mode = not self.debug_mode
        print(f"🐛 Debug mode: {'ON' if self.debug_mode else 'OFF'}")
    
    def _toggle_help(self):
        """Toggle help display"""
        self.show_help = not self.show_help
        print(f"❓ Help display: {'ON' if self.show_help else 'OFF'}")
    
    def _adjust_confidence_threshold(self, delta):
        """Adjust confidence threshold"""
        old_threshold = self.predictor.confidence_threshold
        new_threshold = max(0.0, min(1.0, old_threshold + delta))
        self.predictor.update_confidence_threshold(new_threshold)
    
    def _print_controls(self):
        """Print control instructions"""
        print("\n🎮 CONTROLS:")
        print("q - Quit application")
        print("s - Save session statistics") 
        print("c - Clear prediction history")
        print("t - Toggle model type (Keras ↔ TensorFlow Lite)")
        print("d - Toggle debug mode")
        print("h - Toggle help overlay")
        print("+ - Increase confidence threshold")
        print("- - Decrease confidence threshold")
        print("\n👁️ Watch the camera window...")
    
    def _cleanup(self, cap):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        
        # Print session summary
        self.stats_manager.print_session_summary()
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        print("✅ Cleanup complete")

def check_requirements():
    """Check if all required models and files exist"""
    required_files = [
        Config.SCALER_PATH,
        Config.LABEL_ENCODER_PATH
    ]
    
    # Check if at least one model exists
    keras_exists = os.path.exists(Config.KERAS_MODEL_PATH)
    tflite_exists = os.path.exists(Config.TFLITE_MODEL_PATH)
    
    if not (keras_exists or tflite_exists):
        required_files.extend([Config.KERAS_MODEL_PATH, Config.TFLITE_MODEL_PATH])
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Please run neural_network_training.ipynb first!")
        return False
    
    return True

def main():
    """Main entry point"""
    print("🤖 Hand Gesture Recognition - Clean Version")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Choose model type
    keras_exists = os.path.exists(Config.KERAS_MODEL_PATH)
    tflite_exists = os.path.exists(Config.TFLITE_MODEL_PATH)
    
    print(f"\n📁 Available models:")
    if keras_exists:
        print("   ✅ Keras model (.h5)")
    if tflite_exists:
        print("   ✅ TensorFlow Lite model (.tflite)")
    
    use_tflite = False
    if keras_exists and tflite_exists:
        choice = input("\n🤔 Choose model (k=Keras, t=TensorFlow Lite, default=Keras): ").lower()
        use_tflite = (choice == 't')
    elif tflite_exists:
        use_tflite = True
        print("   ℹ️ Using TensorFlow Lite (only available)")
    else:
        print("   ℹ️ Using Keras (only available)")
    
    # Debug mode option
    debug_mode = input("\n🐛 Enable debug mode? (y/n, default=n): ").lower() == 'y'
    
    try:
        # Create and run app
        app = HandGestureRecognitionApp(use_tflite=use_tflite, debug_mode=debug_mode)
        app.run()
        
    except Exception as e:
        print(f"\n💥 Application error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure camera is connected and accessible")
        print("2. Check that all model files exist") 
        print("3. Verify Python environment and dependencies")

if __name__ == "__main__":
    main()