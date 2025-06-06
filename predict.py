import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import sys

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    # Load image
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    
    # Convert to array and add batch dimension
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize
    img_array = img_array / 255.0
    
    return img_array

def predict_disease(model_path, image_path):
    """Predict disease class for a given image"""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    processed_image = load_and_preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Get class names from the model
    class_names = list(model.output_names)
    
    return class_names[predicted_class], confidence

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    model_path = 'coconut_disease_classifier.h5'
    
    try:
        disease_class, confidence = predict_disease(model_path, image_path)
        print(f"Predicted Disease: {disease_class}")
        print(f"Confidence: {confidence:.2%}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 