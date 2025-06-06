# Coconut Pest Classification Model Fine-Tuning Process

This document details the process of fine-tuning our coconut pest classification model, which achieved high accuracy in identifying four different types of coconut pests.

## Dataset Information

### Classes
- Beetles
- Leaf Miner
- Leaf Spot
- White Flies

### Dataset Organization
The dataset is organized into training and validation sets:

#### Training Set (80%): 473 images
- Beetles: 92 images
- Leaf Miner: 115 images
- Leaf Spot: 136 images
- White Flies: 130 images

#### Validation Set (20%): 121 images
- Beetles: 24 images
- Leaf Miner: 29 images
- Leaf Spot: 35 images
- White Flies: 33 images

### Data Augmentation
- Training data:
  * Random rotation (±20°)
  * Width/height shifts (±20%)
  * Shear transformation
  * Zoom range (±20%)
  * Horizontal flips
  * Fill mode: nearest
- Validation data:
  * Only rescaling (1/255)

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
Dropout (0.3)
↓
Dense (256 units)
↓
Batch Normalization + ReLU
↓
Dropout (0.3)
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
- Early stopping with patience of 8
- Learning rate reduction with patience of 4

### Phase 2: Partial Fine-Tuning
- Unfreeze top 30 layers of base model
- Keep earlier layers frozen
- Learning rate: 0.0001
- Epochs: 15
- Early stopping with patience of 8
- Learning rate reduction with patience of 4

### Phase 3: Full Fine-Tuning
- Unfreeze all layers
- Very small learning rate: 0.00001
- Epochs: 15
- Early stopping with patience of 8
- Learning rate reduction with patience of 4

## Training Optimizations

### Learning Rate Schedule
- Initial learning rates per phase:
  - Phase 1: 0.001
  - Phase 2: 0.0001
  - Phase 3: 0.00001
- Exponential decay:
  - Decay rate: 0.9
  - Steps: 5 epochs worth of batches
  - Staircase: True

### Class Weight Balancing
Applied class weights to handle class imbalance using sklearn's compute_class_weight with 'balanced' mode.

### Early Stopping Strategy
- Monitor: validation accuracy
- Patience: 8 epochs
- Restore best weights: True

### Learning Rate Reduction
- Monitor: validation accuracy
- Factor: 0.5
- Patience: 4 epochs
- Minimum learning rate: 1e-7

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
   - Dropout layers (0.3)
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