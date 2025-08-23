# Intercropping Classification System

A state-of-the-art deep learning system for classifying intercropping plants using EfficientNetV2B1. The system can identify 15 different plant types with high accuracy using advanced augmentation and Test Time Augmentation (TTA).

## Features

- 🔍 Identifies 15 types of intercropping plants:
  - Cacao, Caimito, Guava, Guyabano, JackFruit
  - Kabugaw, Kalamansi, Mahugani, and 7 more classes
- 🎯 High accuracy model (75-85%+ with improvements)
- 🖥️ Multiple interfaces:
  - Command-line tool
  - Web interface (Streamlit)
  - TTA prediction script
- 🚀 Fast inference with Test Time Augmentation
- 📊 Confidence scores and top-3 predictions
- 🎨 Advanced data augmentation for better generalization

## Dataset Organization

### Intercropping Dataset: 1,184+ images across 15 classes
- **Total Images**: 1,184+ high-quality plant images
- **Classes**: 15 different intercropping plant types
- **Split**: 80% training / 20% validation
- **Resolution**: 256x256 pixels (upgraded from 224x224)
- **Format**: JPG images with advanced augmentation

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, for faster inference)

### Basic Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd coconut-pest-classifier
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### GPU Support (Optional)

For GPU acceleration, you need:
1. NVIDIA GPU with CUDA support
2. CUDA Toolkit 11.8
3. cuDNN 8.6 or later

Installation steps:
1. Download and install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Download and install [cuDNN v8.6](https://developer.nvidia.com/cudnn) (requires NVIDIA account)
3. Add CUDA paths to system environment variables

## Usage

### Command Line Interface

1. **Train the improved model**:
```bash
python intercropping_classifier.py
```

2. **Test with Test Time Augmentation** (recommended for best accuracy):
```bash
python predict_with_tta.py path/to/image.jpg
```

3. **Evaluate model with TTA on validation set**:
```bash
python intercropping_classifier.py --tta-eval
```

4. **Single prediction** (basic):
```bash
python test_prediction.py
```

### Web Interface

1. **Streamlit Interface**:
```bash
python -m streamlit run app.py
```
2. **FastAPI Service**:
```bash
python fastapi_app.py
```
3. Open your browser at `http://localhost:8501` (Streamlit) or `http://localhost:8000` (FastAPI)
4. Upload an image and view predictions with confidence scores

### Example TTA Prediction Output
```
🎯 TTA Prediction Results:
Predicted Class: Cacao
Confidence: 0.8945 (89.45%)

📊 Top 3 Predictions:
1. Cacao: 0.8945 (89.45%)
2. Guava: 0.0723 (7.23%)
3. JackFruit: 0.0198 (1.98%)
```

## Docker Deployment

### Local Development
```bash
docker-compose up --build
```

### Production Deployment
1. Configure SSL certificates
2. Update domain in nginx configuration
3. Run with production settings:
```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

## Project Structure

```
intercropping-classifier/
├── intercropping_classifier.py    # Main training script (EfficientNetV2B1)
├── predict_with_tta.py           # Test Time Augmentation prediction
├── test_prediction.py            # Basic prediction tool
├── app.py                        # Streamlit web interface
├── fastapi_app.py               # FastAPI web service
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── docker-compose.yml           # Docker services setup
├── datasets-source/             # Source dataset
│   └── intercropping/          # 15 plant classes
├── intercropping_classification/ # Processed dataset
│   ├── train/                  # Training images (80%)
│   └── validation/             # Validation images (20%)
└── model/                       # Trained model files
    ├── intercropping_classifier_final.h5
    └── best_intercropping_model_phase_*.h5
```

## Model Architecture & Improvements

### Enhanced Architecture
- **Base Model**: EfficientNetV2B1 (upgraded from B0 for better feature extraction)
- **Input Resolution**: 256x256 pixels (upgraded from 224x224)
- **Custom Head**: Residual dense layers with dropout for regularization
- **Classes**: 15 intercropping plant types

### Training Improvements
- **Progressive Training**: 3-phase transfer learning approach
- **Advanced Augmentation**: Brightness, rotation, zoom, channel shifts
- **Cosine Annealing**: Better learning rate scheduling
- **Class Weight Balancing**: Handles imbalanced datasets
- **Early Stopping**: Prevents overfitting

### Test Time Augmentation (TTA)
- **8x Predictions**: Averages multiple augmented versions
- **2-5% Accuracy Boost**: Improved inference reliability
- **Confidence Scores**: Top-3 predictions with probabilities

### Performance Expectations
- **Baseline**: 66.7% accuracy (original setup)
- **With Improvements**: 75-85%+ accuracy expected
- **TTA Boost**: Additional 2-5% improvement

For detailed training logs and model performance, check the generated plot files:
- `intercropping_training_history_phase_*.png`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset contributors
- EfficientNet authors
- TensorFlow team
