# Hand Gesture Recognition with Neural Networks

A real-time hand gesture recognition system using MediaPipe hand landmarks and TensorFlow neural networks. 

## Project Overview

This project can recognize hand gestures in real-time using computer vision and machine learning. It uses MediaPipe to extract 21 hand landmarks and a neural network to classify gestures.

### Features

- **Real-time Recognition**: Live gesture recognition from webcam
- **Dual Model Support**: Both Keras (.h5) and TensorFlow Lite (.tflite) models
- **Data Collection**: Interactive tool for gathering training data
- **Model Training**: Complete Jupyter notebook for neural network training
- **Debug Mode**: Detailed logging and visualization

<img width="1200" height="1021" alt="demo_final" src="https://github.com/user-attachments/assets/558c5dce-1f7f-44ed-927d-ce67a7963a9d" />


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

## Troubleshooting

### Debug Mode

Enable debug mode for detailed information:

```bash
python recog_app.py
# Then press 'd' to toggle debug mode
```

Debug output includes:
- Raw landmark coordinate ranges
- Normalized feature ranges
- Prediction probabilities
- Processing timing

## References

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)





