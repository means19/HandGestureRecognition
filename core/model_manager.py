"""
Model loading and management
"""
import os
import tensorflow as tf
from config.settings import Config

class ModelManager:
    """Manages model loading and inference"""
    
    def __init__(self, use_tflite=False, model_path=None):
        self.use_tflite = use_tflite
        self.model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        self.load_model(model_path)
    
    def load_model(self, model_path=None):
        """Load the appropriate model"""
        try:
            if model_path is None:
                model_path = Config.TFLITE_MODEL_PATH if self.use_tflite else Config.KERAS_MODEL_PATH
            
            if self.use_tflite:
                self._load_tflite_model(model_path)
            else:
                self._load_keras_model(model_path)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _load_tflite_model(self, model_path):
        """Load TensorFlow Lite model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TensorFlow Lite model not found: {model_path}")
        
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"✓ TensorFlow Lite model loaded: {model_path}")
    
    def _load_keras_model(self, model_path):
        """Load Keras model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Keras model not found: {model_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Keras model loaded: {model_path}")
    
    def predict(self, landmarks):
        """Make prediction using the loaded model"""
        try:
            if self.use_tflite:
                return self._predict_tflite(landmarks)
            else:
                return self._predict_keras(landmarks)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def _predict_tflite(self, landmarks):
        """Predict using TensorFlow Lite model"""
        self.interpreter.set_tensor(self.input_details[0]['index'], landmarks)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])
    
    def _predict_keras(self, landmarks):
        """Predict using Keras model"""
        return self.model.predict(landmarks, verbose=0)
    
    def toggle_model_type(self):
        """Toggle between Keras and TensorFlow Lite"""
        keras_exists = os.path.exists(Config.KERAS_MODEL_PATH)
        tflite_exists = os.path.exists(Config.TFLITE_MODEL_PATH)
        
        if not (keras_exists and tflite_exists):
            raise RuntimeError("Cannot toggle: Both models are not available")
        
        self.use_tflite = not self.use_tflite
        self.load_model()
        return "TensorFlow Lite" if self.use_tflite else "Keras"
    
    @property
    def model_type(self):
        """Get current model type"""
        return "TensorFlow Lite" if self.use_tflite else "Keras"