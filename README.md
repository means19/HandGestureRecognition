# Hand Gesture Recognition with Neural Networks

A real-time hand gesture recognition system using MediaPipe hand landmarks and TensorFlow neural networks. This project implements a complete pipeline from data collection to model deployment with a clean, modular architecture.

## Project Overview

This project can recognize hand gestures in real-time using computer vision and machine learning. It uses MediaPipe to extract 21 hand landmarks and a neural network to classify gestures.

### Features

- **Real-time Recognition**: Live gesture recognition from webcam
- **Dual Model Support**: Both Keras (.h5) and TensorFlow Lite (.tflite) models
- **Data Collection**: Interactive tool for gathering training data
- **Model Training**: Complete Jupyter notebook for neural network training
- **Debug Mode**: Detailed logging and visualization

## Project Structure

```
HandGestureRecognition/
├── 📁 config/
│   └── settings.py              # Configuration settings
├── 📁 core/
│   ├── model_manager.py         # Model loading and inference
│   ├── preprocessor.py          # Data preprocessing
│   └── predictor.py            # Gesture prediction with smoothing
├── 📁 utils/
│   ├── landmark_extractor.py   # Hand landmark extraction
│   ├── statistics.py           # Performance tracking
│   └── visualization.py        # UI visualization
├── 📁 dataset/
│   └── hand_gestures_dataset.csv # Training dataset
├── 📁 model/neural_network/
│   ├── checkpoints/            # Best model checkpoints
│   ├── hand_gesture_model.h5   # Keras model
│   ├── hand_gesture_model.tflite # TensorFlow Lite model
│   ├── scaler.pkl              # Feature scaler
│   └── label_encoder.pkl       # Label encoder
├── 📁 logs/                    # Session statistics
├── 📊 neural_network_training.ipynb # Training notebook
├── 📱 app.py                   # Data collection app
├── 🎯 recog_app.py            # Recognition app
├── 📋 requirements.txt         # Dependencies
└── 📚 README.md               # This file
```

## How this project works

### 1. Prerequisites

- Python 3.11 or lower (Since Mediapipe currently supports Python 3.11-)
- Webcam or camera device
- 4GB+ RAM recommended

### 2. Installation

```bash
# Clone the repository
git clone <your-repository-url>
cd HandGestureRecognition

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Collection

```bash
# Run data collection app
python app.py
```

**Instructions:**
- Press number keys (0-9) to select gesture class
- Press `SPACE` to capture samples
- Collect 50-100 samples per gesture
- Press `s` to save dataset

### 4. Model Training

```bash
# Open and run the training notebook
jupyter notebook neural_network_training.ipynb
```

**Training Process:**
1. Load and explore dataset
2. Preprocess hand landmarks
3. Train neural network
4. Evaluate model performance
5. Convert to TensorFlow Lite
6. Export trained models

### 5. Recognition

```bash
# Run the clean recognition app
python recog_app.py
```

## Usage Instructions

### Data Collection App (`app.py`)

| Key | Action |
|-----|--------|
| `0-9` | Select gesture class |
| `SPACE` | Capture current hand pose |
| `s` | Save dataset to CSV |
| `c` | Clear current session |
| `q` | Quit application |

### Recognition App (`recog_app.py`)

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save session statistics |
| `c` | Clear prediction history |
| `t` | Toggle model type (Keras ↔ TensorFlow Lite) |
| `d` | Toggle debug mode |
| `h` | Toggle help overlay |
| `+` | Increase confidence threshold |
| `-` | Decrease confidence threshold |

## Model Architecture

The neural network uses a simple but effective architecture:

```python
Input Layer (42 features: 21 landmarks × 2 coordinates)
    ↓
Dropout (0.2) → Dense (20, ReLU) → Dropout (0.4)
    ↓
Dense (10, ReLU)
    ↓
Output Layer (num_classes, Softmax)
```

### Model Features:
- **Input**: 42 hand landmark coordinates (x,y for 21 points)
- **Hidden Layers**: 20 → 10 neurons with ReLU activation
- **Regularization**: Dropout layers to prevent overfitting
- **Output**: Softmax probabilities for gesture classes

## Performance

### Typical Results:
- **Training Accuracy**: 95-99%
- **Validation Accuracy**: 85-95%
- **Inference Speed**: 
  - Keras Model: ~5-10ms per prediction
  - TensorFlow Lite: ~2-5ms per prediction
- **Model Size**:
  - Keras (.h5): ~50-100KB
  - TensorFlow Lite (.tflite): ~20-50KB

## Configuration

Modify `config/settings.py` to customize:

```python
# Model paths
KERAS_MODEL_PATH = "model/neural_network/checkpoints/hand_gesture_model.h5"
TFLITE_MODEL_PATH = "model/neural_network/hand_gesture_model.tflite"

# Recognition settings
CONFIDENCE_THRESHOLD = 0.5
PREDICTION_HISTORY_SIZE = 5
SMOOTHING_WINDOW = 3

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
```

## Development

### Adding New Gestures

1. **Collect Data**: Use `app.py` to gather samples
2. **Retrain Model**: Run training notebook
3. **Update Classes**: Model automatically detects new classes
4. **Test**: Use recognition app to validate

### Code Structure

- **Modular Design**: Each component has single responsibility
- **Error Handling**: Comprehensive exception handling
- **Type Safety**: Clear function signatures and documentation
- **Performance**: Optimized for real-time inference

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Troubleshooting

### Common Issues

**1. Model Not Found**
```bash
❌ Missing required files: model/neural_network/...
```
**Solution**: Run `neural_network_training.ipynb` first

**2. Camera Not Working**
```bash
❌ Cannot open camera
```
**Solution**: 
- Check camera permissions
- Try different camera index in `Config.CAMERA_INDEX`
- Ensure no other apps are using camera

**3. Low Accuracy**
```bash
Predictions are inconsistent
```
**Solution**:
- Collect more training data (100+ samples per gesture)
- Ensure consistent lighting conditions
- Check hand is clearly visible
- Adjust confidence threshold

**4. Slow Performance**
```bash
Low FPS or laggy recognition
```
**Solution**:
- Use TensorFlow Lite model
- Reduce camera resolution
- Close other applications
- Consider GPU acceleration

### Debug Mode

Enable debug mode for detailed information:

```bash
python recog_app_clean.py
# Then press 'd' to toggle debug mode
```

Debug output includes:
- Raw landmark coordinate ranges
- Normalized feature ranges
- Prediction probabilities
- Processing timing

## Advanced Usage

### Custom Training

```python
# Custom model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input((42,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

### Batch Processing

```python
from core.model_manager import ModelManager
from core.preprocessor import Preprocessor

# Load models
model_manager = ModelManager(use_tflite=True)
preprocessor = Preprocessor()

# Process multiple samples
for landmarks in landmark_batch:
    normalized = preprocessor.normalize_landmarks(landmarks)
    prediction = model_manager.predict(normalized)
```

## References

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


