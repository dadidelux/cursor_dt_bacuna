#!/usr/bin/env python3
"""
Test Time Augmentation (TTA) Prediction Script
Usage: python predict_with_tta.py image_path
"""

import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# Configuration
IMG_HEIGHT = 256
IMG_WIDTH = 256

def predict_with_tta(model, image_path, class_names, num_augmentations=8):
    """Predict with Test Time Augmentation for better accuracy."""
    
    # Load and preprocess image
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    
    predictions = []
    
    # Original image
    predictions.append(model.predict(np.expand_dims(img_array, axis=0), verbose=0))
    
    # Augmented versions
    tta_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )
    
    for _ in range(num_augmentations - 1):
        augmented = tta_datagen.random_transform(img_array)
        predictions.append(model.predict(np.expand_dims(augmented, axis=0), verbose=0))
    
    # Average predictions
    avg_prediction = np.mean(predictions, axis=0)
    predicted_class_idx = np.argmax(avg_prediction, axis=1)[0]
    confidence = np.max(avg_prediction)
    
    # Get top 3 predictions
    top3_indices = np.argsort(avg_prediction[0])[-3:][::-1]
    top3_predictions = [(class_names[i], avg_prediction[0][i]) for i in top3_indices]
    
    return predicted_class_idx, confidence, top3_predictions

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_with_tta.py <image_path>")
        print("Example: python predict_with_tta.py test_image.jpg")
        return
    
    image_path = sys.argv[1]
    model_path = 'intercropping_classifier_final.h5'
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Train the model first.")
        return
    
    # Load model
    print("Loading model...")
    model = load_model(model_path)
    
    # Get class names from training directory
    train_dir = 'intercropping_classification/train'
    if os.path.exists(train_dir):
        class_names = sorted(os.listdir(train_dir))
    else:
        # Fallback class names for intercropping
        class_names = [
            'Cacao', 'Caimito', 'Guava', 'Guyabano', 'JackFruit', 
            'Kabugaw', 'Kalamansi', 'Mahugani', 'Class_9', 'Class_10',
            'Class_11', 'Class_12', 'Class_13', 'Class_14', 'Class_15'
        ]
    
    print(f"Predicting with TTA for: {image_path}")
    print(f"Number of classes: {len(class_names)}")
    
    # Make prediction with TTA
    predicted_class_idx, confidence, top3_predictions = predict_with_tta(
        model, image_path, class_names, num_augmentations=8
    )
    
    print(f"\n🎯 TTA Prediction Results:")
    print(f"Predicted Class: {class_names[predicted_class_idx]}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    print(f"\n📊 Top 3 Predictions:")
    for i, (class_name, prob) in enumerate(top3_predictions, 1):
        print(f"{i}. {class_name}: {prob:.4f} ({prob*100:.2f}%)")

if __name__ == '__main__':
    main()