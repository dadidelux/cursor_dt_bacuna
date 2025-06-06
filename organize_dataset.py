import os
import shutil
import random
from pathlib import Path

# Source and destination directories
source_dirs = {
    'beetles': 'datasets/Beetles',
    'beal_miner': 'datasets/Leaf_Miner',
    'leaf_spot': 'datasets/Leaf_Spot',
    'white_flies': 'datasets/White_Files'
}

dest_base = 'dataset'
validation_split = 0.2  # 20% for validation

def create_dirs():
    """Create destination directories if they don't exist"""
    for category in source_dirs.keys():
        Path(f'{dest_base}/train/{category}').mkdir(parents=True, exist_ok=True)
        Path(f'{dest_base}/validation/{category}').mkdir(parents=True, exist_ok=True)

def organize_files():
    """Organize files into training and validation sets"""
    for category, source_dir in source_dirs.items():
        # Get all image files
        files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle files
        random.shuffle(files)
        
        # Calculate split point
        split_idx = int(len(files) * validation_split)
        
        # Split into validation and training sets
        validation_files = files[:split_idx]
        train_files = files[split_idx:]
        
        print(f"\nProcessing {category}:")
        print(f"Total files: {len(files)}")
        print(f"Training files: {len(train_files)}")
        print(f"Validation files: {len(validation_files)}")
        
        # Copy validation files
        for f in validation_files:
            src = os.path.join(source_dir, f)
            dst = os.path.join(dest_base, 'validation', category, f)
            shutil.copy2(src, dst)
            
        # Copy training files
        for f in train_files:
            src = os.path.join(source_dir, f)
            dst = os.path.join(dest_base, 'train', category, f)
            shutil.copy2(src, dst)

if __name__ == "__main__":
    print("Creating directories...")
    create_dirs()
    
    print("Organizing files...")
    organize_files()
    
    print("\nDataset organization complete!") 