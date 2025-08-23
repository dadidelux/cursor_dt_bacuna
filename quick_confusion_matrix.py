import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import random

# Intercropping class labels
class_names = ['Cacao', 'Caimito', 'Guava', 'Guyabano', 'JackFruit', 'Kabugaw', 'Kalamansi']

# Custom DepthwiseConv2D layer for compatibility
class CompatibleDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

def load_model_fast(model_path):
    """Load model with compatibility fixes."""
    try:
        custom_objects = {'DepthwiseConv2D': CompatibleDepthwiseConv2D}
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
    except:
        try:
            model = load_model(model_path, compile=False)
            return model
        except:
            return None

def load_sample_validation_data(max_images_per_class=5):
    """Load a sample of validation images for faster processing."""
    validation_dir = 'intercropping_classification/validation'
    images = []
    labels = []
    true_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(validation_dir, class_name)
        if os.path.exists(class_dir):
            image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Take only a sample of images for faster processing
            sample_size = min(max_images_per_class, len(image_files))
            if sample_size > 0:
                selected_files = random.sample(image_files, sample_size)
                
                for image_file in selected_files:
                    image_path = os.path.join(class_dir, image_file)
                    try:
                        img = load_img(image_path, target_size=(256, 256))
                        img_array = img_to_array(img) / 255.0
                        
                        images.append(img_array)
                        labels.append(class_idx)
                        true_labels.append(class_name)
                        
                    except Exception as e:
                        print(f"Error loading {image_path}: {str(e)}")
    
    return np.array(images), np.array(labels), true_labels

def predict_batch_fast(model, images):
    """Make predictions with minimal verbosity."""
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes, predictions

def generate_confusion_matrix_fast(y_true, y_pred, class_names, model_name):
    """Generate confusion matrix quickly."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    accuracy = accuracy_score(y_true, y_pred)
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.3f}', 
             ha='center', va='center', transform=plt.gca().transAxes, 
             fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = f'confusion_matrix_{model_name.replace(".h5", "")}_quick.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.show()
    
    return cm, accuracy

def print_quick_metrics(y_true, y_pred, class_names, model_name, accuracy):
    """Print quick metrics summary."""
    print("\n" + "="*50)
    print("QUICK CONFUSION MATRIX RESULTS")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        if cm[i].sum() > 0:
            class_accuracy = cm[i, i] / cm[i].sum()
            print(f"{class_name:12}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%)")
    
    # Save quick results
    output_file = f'confusion_matrix_results_{model_name.replace(".h5", "")}_quick.txt'
    with open(output_file, 'w') as f:
        f.write(f"QUICK CONFUSION MATRIX RESULTS\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)\n\n")
        f.write("Per-Class Accuracy:\n")
        for i, class_name in enumerate(class_names):
            if cm[i].sum() > 0:
                class_accuracy = cm[i, i] / cm[i].sum()
                f.write(f"{class_name}: {class_accuracy:.3f}\n")

def main():
    print("QUICK CONFUSION MATRIX GENERATOR")
    print("="*40)
    
    # Find model
    possible_models = [
        'intercropping_classifier_final.h5',
        'best_intercropping_model_phase_3.h5',
        'intercropping_classifier_v2_phase_3.h5'
    ]
    
    model_path = None
    for model_file in possible_models:
        if os.path.exists(model_file):
            model_path = model_file
            break
    
    if not model_path:
        print("No intercropping model found!")
        return
    
    print(f"Loading {model_path}...")
    model = load_model_fast(model_path)
    if not model:
        print("Failed to load model!")
        return
    
    print("Loading sample validation data (5 images per class)...")
    images, labels, true_labels = load_sample_validation_data(max_images_per_class=5)
    print(f"Loaded {len(images)} sample images")
    
    print("Making predictions...")
    predicted_classes, _ = predict_batch_fast(model, images)
    
    print("Generating confusion matrix...")
    cm, accuracy = generate_confusion_matrix_fast(labels, predicted_classes, class_names, model_path)
    
    print_quick_metrics(labels, predicted_classes, class_names, model_path, accuracy)
    
    print(f"\nQuick analysis complete!")
    print(f"Files saved:")
    print(f"- confusion_matrix_{model_path.replace('.h5', '')}_quick.png")
    print(f"- confusion_matrix_results_{model_path.replace('.h5', '')}_quick.txt")

if __name__ == "__main__":
    main() 