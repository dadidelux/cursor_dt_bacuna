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

# Intercropping class labels (based on the dataset structure)
class_names = ['Cacao', 'Caimito', 'Guava', 'Guyabano', 'JackFruit', 'Kabugaw', 'Kalamansi']

# Custom DepthwiseConv2D layer to handle compatibility issues
class CompatibleDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove the 'groups' parameter if it exists (not supported in older versions)
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

def preprocess_image(image_path):
    """Preprocess a single image for prediction."""
    # Load and resize image (using the same size as training)
    img = load_img(image_path, target_size=(256, 256))
    # Convert to array and add batch dimension
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess input (same as training)
    img_array = img_array / 255.0
    return img_array

def load_validation_data():
    """Load all validation images and their labels."""
    validation_dir = 'intercropping_classification/validation'
    images = []
    labels = []
    true_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(validation_dir, class_name)
        if os.path.exists(class_dir):
            image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            for image_file in image_files:
                image_path = os.path.join(class_dir, image_file)
                try:
                    # Preprocess image
                    img = load_img(image_path, target_size=(256, 256))
                    img_array = img_to_array(img)
                    img_array = img_array / 255.0
                    
                    images.append(img_array)
                    labels.append(class_idx)
                    true_labels.append(class_name)
                    
                except Exception as e:
                    print(f"Error loading {image_path}: {str(e)}")
    
    return np.array(images), np.array(labels), true_labels

def predict_batch(model, images):
    """Make predictions for a batch of images."""
    predictions = model.predict(images, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes, predictions

def generate_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Generate and plot confusion matrix."""
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure with larger size
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Confusion Matrix - Intercropping Classification Model ({model_name})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add accuracy text
    accuracy = accuracy_score(y_true, y_pred)
    plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.3f}', 
             ha='center', va='center', transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(".h5", "")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def print_detailed_metrics(y_true, y_pred, true_labels, class_names, model_name):
    """Print detailed classification metrics."""
    print("\n" + "="*70)
    print("DETAILED INTERCROPPING CLASSIFICATION METRICS")
    print("="*70)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nModel: {model_name}")
    print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 50)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 50)
    cm = confusion_matrix(y_true, y_pred)
    for i, class_name in enumerate(class_names):
        if cm[i].sum() > 0:
            class_accuracy = cm[i, i] / cm[i].sum()
            print(f"{class_name:12}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%)")
    
    # Confusion matrix as table
    print("\nConfusion Matrix:")
    print("-" * 50)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    # Save detailed results to file
    output_filename = f'confusion_matrix_results_{model_name.replace(".h5", "")}.txt'
    with open(output_filename, 'w') as f:
        f.write("INTERCROPPING CLASSIFICATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)\n\n")
        f.write("Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write("-" * 50 + "\n")
        f.write(cm_df.to_string() + "\n")

def load_model_with_compatibility(model_path):
    """Load model with compatibility fixes for older TensorFlow versions."""
    try:
        # First try loading with custom objects
        custom_objects = {
            'DepthwiseConv2D': CompatibleDepthwiseConv2D
        }
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded with custom objects successfully!")
        return model
    except Exception as e1:
        print(f"Failed to load with custom objects: {str(e1)}")
        try:
            # Try loading without custom objects
            model = load_model(model_path, compile=False)
            print("Model loaded without custom objects successfully!")
            return model
        except Exception as e2:
            print(f"Failed to load without custom objects: {str(e2)}")
            try:
                # Try loading with safe_mode
                model = load_model(model_path, compile=False, safe_mode=True)
                print("Model loaded with safe_mode successfully!")
                return model
            except Exception as e3:
                print(f"All loading methods failed. Last error: {str(e3)}")
                return None

def main():
    # List of possible model files to try
    possible_models = [
        'intercropping_classifier_final_old.h5',
        'intercropping_classifier_final.h5',
        'intercropping_classifier_improved_final.h5',
        'best_intercropping_model_phase_3.h5',
        'intercropping_classifier_v2_phase_3.h5'
    ]
    
    # Find the model file
    model_path = None
    for model_file in possible_models:
        if os.path.exists(model_file):
            model_path = model_file
            break
    
    if model_path is None:
        print("Error: No intercropping model found!")
        print("Available models to check:")
        for model_file in possible_models:
            print(f"  - {model_file}")
        return
    
    print(f"Loading {model_path}...")
    model = load_model_with_compatibility(model_path)
    
    if model is None:
        print("Failed to load any intercropping model!")
        return
    
    print("\nLoading validation data...")
    try:
        images, labels, true_labels = load_validation_data()
        print(f"Loaded {len(images)} validation images")
        
        if len(images) == 0:
            print("No validation images found!")
            return
            
    except Exception as e:
        print(f"Error loading validation data: {str(e)}")
        return
    
    print(f"\nValidation data distribution:")
    for i, class_name in enumerate(class_names):
        count = np.sum(labels == i)
        print(f"{class_name:12}: {count} images")
    
    print("\nMaking predictions...")
    try:
        predicted_classes, predictions = predict_batch(model, images)
        print("Predictions completed!")
        
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return
    
    print("\nGenerating confusion matrix...")
    try:
        cm = generate_confusion_matrix(labels, predicted_classes, class_names, model_path)
        print(f"Confusion matrix saved as 'confusion_matrix_{model_path.replace('.h5', '')}.png'")
        
    except Exception as e:
        print(f"Error generating confusion matrix: {str(e)}")
        return
    
    print("\nGenerating detailed metrics...")
    try:
        print_detailed_metrics(labels, predicted_classes, true_labels, class_names, model_path)
        print(f"Detailed results saved as 'confusion_matrix_results_{model_path.replace('.h5', '')}.txt'")
        
    except Exception as e:
        print(f"Error generating detailed metrics: {str(e)}")
        return
    
    print("\n" + "="*70)
    print("INTERCROPPING ANALYSIS COMPLETE!")
    print("="*70)
    print("Files generated:")
    print(f"- confusion_matrix_{model_path.replace('.h5', '')}.png (visual confusion matrix)")
    print(f"- confusion_matrix_results_{model_path.replace('.h5', '')}.txt (detailed metrics)")

if __name__ == "__main__":
    main() 