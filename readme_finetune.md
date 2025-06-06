# Coconut Pest Classification Model Fine-Tuning Process

This document details the process of fine-tuning our coconut pest classification model, which achieved high accuracy in identifying four different types of coconut pests.

## Dataset Information

### Classes
- Beetles
- Beal Miner
- Leaf Spot
- White Flies

### Dataset Distribution
- **Training Set**: 477 images
  - Beetles: 116 images
  - Beal Miner: 144 images
  - Leaf Spot: 171 images
  - White Flies: 163 images
- **Validation Set**: 117 images (20% split)

## Model Architecture

### Base Model
- **Architecture**: EfficientNetV2B0
- **Input Size**: 224x224x3 (RGB images)
- **Pre-trained Weights**: ImageNet

### Custom Classification Head
```
Global Average Pooling
↓
Batch Normalization
↓
Dense (256 units)
↓
Batch Normalization + ReLU
↓
Dropout (0.5)
↓
Dense (256 units)
↓
Batch Normalization + ReLU
↓
Dropout (0.5)
↓
Residual Connection (Skip Connection)
↓
Dense (4 units, Softmax)
```

## Fine-Tuning Strategy

### Phase 1: Head Training
- Freeze EfficientNetV2B0 base
- Train only the custom classification head
- Learning rate: 0.001
- Epochs: 15
- Best validation accuracy: 94.68%

### Phase 2: Partial Fine-Tuning
- Unfreeze top 30 layers of base model
- Keep earlier layers frozen
- Learning rate: 0.0001
- Epochs: 15
- Best validation accuracy: 96.81%

### Phase 3: Full Fine-Tuning
- Unfreeze all layers
- Very small learning rate: 0.00001
- Epochs: 15
- Best validation accuracy: 100%

## Training Optimizations

### Learning Rate Schedule
- Initial learning rates per phase:
  - Phase 1: 0.001
  - Phase 2: 0.0001
  - Phase 3: 0.00001
- ReduceLROnPlateau:
  - Monitor: val_accuracy
  - Factor: 0.9
  - Patience: 3

### Class Weight Balancing
Applied class weights to handle class imbalance:
```python
class_weights = {
    'beetles': 1.0296,
    'beal_miner': 1.2767,
    'leaf_spot': 0.8705,
    'white_flies': 0.9119
}
```

### Data Augmentation
- Random rotation (±20°)
- Random zoom (±20%)
- Random horizontal flip
- Random vertical flip
- Random brightness adjustment (±20%)

## Model Performance

### Final Metrics
- Training Accuracy: 99.48%
- Validation Accuracy: 100%
- Loss: 0.0114

### Per-Class Performance
All classes achieved near-perfect classification on the validation set, with the model showing robust performance across all pest types.

## Key Improvements Made

1. **Architecture Enhancement**
   - Added residual connections
   - Implemented double dense layers
   - Added batch normalization after each dense layer

2. **Training Strategy**
   - Progressive layer unfreezing
   - Class weight balancing
   - Learning rate scheduling

3. **Regularization**
   - Dropout layers (0.5)
   - Data augmentation
   - Early stopping

## Model Deployment

The final model is saved as 'best_model_phase_3.h5' and can be used for:
- Command-line inference (test_prediction.py)
- Web interface (Streamlit app)

## Future Improvements

1. **Data Collection**
   - Gather more images for underrepresented classes
   - Include more variety in image conditions

2. **Model Enhancement**
   - Experiment with other EfficientNet variants
   - Implement cross-validation
   - Test ensemble methods

3. **Deployment**
   - Add model quantization
   - Implement TensorFlow Lite conversion
   - Add model explanation features 