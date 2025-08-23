import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd

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

def load_validation_data():
    """Load all validation images and their labels."""
    validation_dir = 'dataset/validation'
    images = []
    labels = []
    true_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(validation_dir, class_name)
        if os.path.exists(class_dir):
            image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in image_files:
                image_path = os.path.join(class_dir, image_file)
                try:
                    # Preprocess image
                    img = load_img(image_path, target_size=(224, 224))
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

def generate_confusion_matrix(y_true, y_pred, class_names):
    """Generate and plot confusion matrix."""
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure with larger size
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix - Coconut Disease Classification Model', fontsize=16, fontweight='bold')
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
    plt.savefig('confusion_matrix_best_model_phase_3.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def print_detailed_metrics(y_true, y_pred, true_labels, class_names):
    """Print detailed classification metrics."""
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION METRICS")
    print("="*60)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 40)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 40)
    cm = confusion_matrix(y_true, y_pred)
    for i, class_name in enumerate(class_names):
        if cm[i].sum() > 0:
            class_accuracy = cm[i, i] / cm[i].sum()
            print(f"{class_name:12}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%)")
    
    # Confusion matrix as table
    print("\nConfusion Matrix:")
    print("-" * 40)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    # Save detailed results to file
    with open('confusion_matrix_results.txt', 'w') as f:
        f.write("COCONUT DISEASE CLASSIFICATION RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: best_model_phase_3.h5\n")
        f.write(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)\n\n")
        f.write("Classification Report:\n")
        f.write("-" * 40 + "\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write("-" * 40 + "\n")
        f.write(cm_df.to_string() + "\n")

def main():
    print("Loading best_model_phase_3.h5...")
    try:
        model = load_model('best_model_phase_3.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
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
        cm = generate_confusion_matrix(labels, predicted_classes, class_names)
        print("Confusion matrix saved as 'confusion_matrix_best_model_phase_3.png'")
        
    except Exception as e:
        print(f"Error generating confusion matrix: {str(e)}")
        return
    
    print("\nGenerating detailed metrics...")
    try:
        print_detailed_metrics(labels, predicted_classes, true_labels, class_names)
        print("Detailed results saved as 'confusion_matrix_results.txt'")
        
    except Exception as e:
        print(f"Error generating detailed metrics: {str(e)}")
        return
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Files generated:")
    print("- confusion_matrix_best_model_phase_3.png (visual confusion matrix)")
    print("- confusion_matrix_results.txt (detailed metrics)")

if __name__ == "__main__":
    main() 