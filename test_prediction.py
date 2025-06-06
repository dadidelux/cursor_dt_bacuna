import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os
import random

# Class labels
class_names = ['beetles', 'leaf_miner', 'leaf_spot', 'white_flies']

def preprocess_image(image_path):
    """Preprocess a single image for prediction."""
    # Load and resize image
    img = load_img(image_path, target_size=(224, 224))
    # Convert to array and add batch dimension
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess input (same as training)
    img_array = img_array / 255.0
    return img_array

def predict_image(model, image_path):
    """Make prediction for a single image."""
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    
    # Get the predicted class and confidence
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return class_names[predicted_class], confidence

def get_validation_images():
    """Get list of validation images for each class."""
    validation_dir = 'dataset/validation'
    validation_images = {}
    
    for class_name in class_names:
        class_dir = os.path.join(validation_dir, class_name)
        if os.path.exists(class_dir):
            images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            validation_images[class_name] = sorted(images)  # Sort images for consistent numbering
    
    return validation_images

def main():
    # Load the best model (from Phase 3)
    model = load_model('best_model_phase_3.h5')
    print("Model loaded successfully!")
    
    # Get validation images
    validation_images = get_validation_images()
    
    if not validation_images:
        print("Error: No validation images found in dataset/validation directory")
        return
    
    print("\nAvailable validation images per class:")
    for class_name, images in validation_images.items():
        print(f"{class_name.replace('_', ' ').title()}: {len(images)} images")
    
    while True:
        print("\nOptions:")
        print("1. Test random image")
        print("2. Test specific class")
        print("3. Quit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '3':
            break
        
        if choice == '1':
            # Select random class and image
            class_name = random.choice(list(validation_images.keys()))
            image_path = random.choice(validation_images[class_name])
            image_number = validation_images[class_name].index(image_path) + 1
            print(f"\nSelected random image from class: {class_name.replace('_', ' ').title()}")
            print(f"Image {image_number} of {len(validation_images[class_name])}: {os.path.basename(image_path)}")
            
        elif choice == '2':
            # Let user select class
            print("\nAvailable classes:")
            for i, class_name in enumerate(class_names):
                print(f"{i+1}. {class_name.replace('_', ' ').title()}")
            
            try:
                class_idx = int(input("\nEnter class number (1-4): ")) - 1
                if class_idx < 0 or class_idx >= len(class_names):
                    print("Invalid class number")
                    continue
                
                class_name = class_names[class_idx]
                if not validation_images[class_name]:
                    print(f"No images available for {class_name}")
                    continue
                
                # Show available images in the class
                print(f"\nImages in {class_name.replace('_', ' ').title()} class:")
                for i, img_path in enumerate(validation_images[class_name], 1):
                    print(f"{i}. {os.path.basename(img_path)}")
                
                # Let user select image number
                img_num = int(input(f"\nEnter image number (1-{len(validation_images[class_name])}): "))
                if img_num < 1 or img_num > len(validation_images[class_name]):
                    print("Invalid image number")
                    continue
                
                image_path = validation_images[class_name][img_num - 1]
                print(f"\nSelected image {img_num} of {len(validation_images[class_name])}: {os.path.basename(image_path)}")
                
            except ValueError:
                print("Invalid input")
                continue
        else:
            print("Invalid choice")
            continue
            
        try:
            # Make prediction
            predicted_class, confidence = predict_image(model, image_path)
            
            # Print results
            print("\nResults:")
            print(f"Image Path: {image_path}")
            print(f"True Class: {os.path.basename(os.path.dirname(image_path)).replace('_', ' ').title()}")
            print(f"Predicted Class: {predicted_class.replace('_', ' ').title()}")
            print(f"Confidence: {confidence*100:.2f}%")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 