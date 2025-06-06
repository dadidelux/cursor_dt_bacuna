import os
import shutil
import random
from pathlib import Path

def create_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def clear_directory(path):
    """Remove all contents of a directory if it exists."""
    if os.path.exists(path):
        shutil.rmtree(path)
    create_directory(path)

def split_dataset(source_dir="datasets-source", train_dir="dataset/train", val_dir="dataset/validation", train_ratio=0.8):
    """Split dataset into training and validation sets."""
    # Clear existing directories
    clear_directory(train_dir)
    clear_directory(val_dir)
    
    # Process each class
    class_counts = {}
    
    for class_name in os.listdir(source_dir):
        source_class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(source_class_dir):
            continue
            
        # Create class directories in train and validation
        train_class_dir = os.path.join(train_dir, class_name.lower())
        val_class_dir = os.path.join(val_dir, class_name.lower())
        create_directory(train_class_dir)
        create_directory(val_class_dir)
        
        # Get all images
        images = [f for f in os.listdir(source_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        # Calculate split
        n_train = int(len(images) * train_ratio)
        train_images = images[:n_train]
        val_images = images[n_train:]
        
        # Copy images
        for img in train_images:
            shutil.copy2(
                os.path.join(source_class_dir, img),
                os.path.join(train_class_dir, img)
            )
        
        for img in val_images:
            shutil.copy2(
                os.path.join(source_class_dir, img),
                os.path.join(val_class_dir, img)
            )
        
        # Store counts
        class_counts[class_name] = {
            'total': len(images),
            'train': len(train_images),
            'validation': len(val_images)
        }
    
    # Print summary
    print("\n=== Dataset Split Summary ===")
    print(f"Split ratio: {train_ratio:.0%} training, {1-train_ratio:.0%} validation\n")
    
    print("Class distribution:")
    print("-" * 60)
    print(f"{'Class':<15} {'Total':>10} {'Training':>10} {'Validation':>10}")
    print("-" * 60)
    
    total_train = 0
    total_val = 0
    total_all = 0
    
    for class_name, counts in class_counts.items():
        print(f"{class_name:<15} {counts['total']:>10} {counts['train']:>10} {counts['validation']:>10}")
        total_train += counts['train']
        total_val += counts['validation']
        total_all += counts['total']
    
    print("-" * 60)
    print(f"{'Total':<15} {total_all:>10} {total_train:>10} {total_val:>10}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    split_dataset() 