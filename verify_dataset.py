import os
import shutil
from PIL import Image
import imagehash
from pathlib import Path

def compute_image_hash(image_path):
    """Compute perceptual hash of an image."""
    try:
        return str(imagehash.average_hash(Image.open(image_path)))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def verify_datasets():
    """Verify consistency between datasets and dataset/train folders."""
    reference_dir = "datasets-source"
    train_dir = "dataset/train"
    val_dir = "dataset/validation"
    
    # Map folder names (case-insensitive)
    reference_folders = {f.lower(): f for f in os.listdir(reference_dir) if os.path.isdir(os.path.join(reference_dir, f))}
    train_folders = {f.lower(): f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))}
    val_folders = {f.lower(): f for f in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, f))}
    
    print("=== Dataset Structure Analysis ===")
    print(f"\nReference folders ({reference_dir}):")
    for f in reference_folders.values():
        print(f"- {f}")
    
    print(f"\nTraining folders ({train_dir}):")
    for f in train_folders.values():
        print(f"- {f}")
        
    print(f"\nValidation folders ({val_dir}):")
    for f in val_folders.values():
        print(f"- {f}")
    
    # Verify image consistency
    print("\n=== Image Distribution ===")
    
    for folder_lower, ref_folder in reference_folders.items():
        train_folder = train_folders.get(folder_lower)
        val_folder = val_folders.get(folder_lower)
        
        if not train_folder or not val_folder:
            print(f"\nWarning: Missing folder for class {ref_folder}")
            continue
            
        ref_path = os.path.join(reference_dir, ref_folder)
        train_path = os.path.join(train_dir, train_folder)
        val_path = os.path.join(val_dir, val_folder)
        
        ref_images = {f.lower(): f for f in os.listdir(ref_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        train_images = {f.lower(): f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        val_images = {f.lower(): f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        
        print(f"\nAnalyzing {ref_folder}:")
        print(f"- Reference images: {len(ref_images)}")
        print(f"- Training images: {len(train_images)}")
        print(f"- Validation images: {len(val_images)}")
        
        # Verify all images are properly distributed
        all_train_and_val = set(train_images.keys()) | set(val_images.keys())
        missing_images = set(ref_images.keys()) - all_train_and_val
        extra_images = all_train_and_val - set(ref_images.keys())
        
        if missing_images:
            print(f"Warning: {len(missing_images)} images from reference not in train or validation")
            
        if extra_images:
            print(f"Warning: {len(extra_images)} images in train/validation not from reference")
            
        # Check for duplicates between train and validation
        duplicates = set(train_images.keys()) & set(val_images.keys())
        if duplicates:
            print(f"Warning: {len(duplicates)} images appear in both train and validation sets")

if __name__ == "__main__":
    verify_datasets() 