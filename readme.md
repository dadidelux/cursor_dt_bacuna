# Coconut Pest Classification Model

A deep learning model for classifying different types of coconut pests using TensorFlow and EfficientNet architecture.

## Dataset Structure

The dataset consists of four pest categories:
- Beetles (116 images)
- Beal Miner (144 images)
- Leaf Spot (171 images)
- White Flies (163 images)

Images are split 80-20 between training and validation sets.

## Model Architecture

The model uses EfficientNet-B0 architecture with transfer learning:

### 1. Base Model
- EfficientNet-B0 pre-trained on ImageNet
- Input: RGB images of size 224x224x3
- Images are normalized (pixel values divided by 255)

### Why EfficientNet?
EfficientNet is a state-of-the-art architecture that:
1. Uses compound scaling to balance network depth, width, and resolution
2. Achieves better accuracy and efficiency than traditional CNNs
3. Requires fewer parameters while maintaining high performance
4. Utilizes mobile inverted bottleneck convolution (MBConv) blocks

### Model Structure
```
1. EfficientNet-B0 (pre-trained, frozen)
   ↓
2. Global Average Pooling 2D
   ↓
3. Batch Normalization
   ↓
4. Dropout (0.2)
   ↓
5. Dense Layer (128 units, ReLU)
   ↓
6. Dropout (0.3)
   ↓
7. Output Layer (4 units, Softmax)
```

## Transfer Learning Benefits
- Leverages knowledge from ImageNet (1.4M images)
- Reduces training time significantly
- Improves generalization
- Better performance with limited data

## Training Parameters

The model uses the following training configuration:

- **Batch Size**: 32 images
- **Epochs**: 20
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-entropy
- **Metrics**: Accuracy

### Data Augmentation
To improve model robustness, the following augmentations are applied to training images:
```python
rotation_range=20
width_shift_range=0.2
height_shift_range=0.2
shear_range=0.2
zoom_range=0.2
horizontal_flip=True
```

## Using the Model for Predictions

Once trained, you can use the model for predictions using the following code:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def predict_pest(image_path, model_path='coconut_pest_classifier.h5'):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get class names
    class_names = ['beetles', 'beal_miner', 'leaf_spot', 'white_flies']
    
    # Get the predicted class and confidence
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_class, confidence

# Example usage
image_path = 'path/to/your/image.jpg'
pest_type, confidence = predict_pest(image_path)
print(f"Predicted pest: {pest_type}")
print(f"Confidence: {confidence:.2%}")
```

## Model Performance

The training process generates a `training_history.png` file showing:
- Training vs. Validation Accuracy
- Training vs. Validation Loss

This helps visualize the model's learning progress and identify potential overfitting.

## Adjusting Parameters

To modify the model's behavior, you can adjust these parameters in `coconut_disease_classifier.py`:

```python
# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20

# Model architecture
# Increase/decrease number of filters in Conv2D layers
# Adjust Dense layer sizes
# Modify Dropout rate

# Data augmentation
# Adjust augmentation parameters in train_datagen
```

## Requirements

```bash
tensorflow==2.15.0
numpy==1.24.3
pandas==2.1.4
scikit-learn==1.3.2
matplotlib==3.8.2
Pillow==10.1.0
```

## Setup and Training

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Organize dataset:
```bash
python organize_dataset.py
```

3. Train the model:
```bash
python coconut_disease_classifier.py
```

## Model Output

The training process will generate:
1. `coconut_pest_classifier.h5` - The trained model
2. `training_history.png` - Training progress visualization