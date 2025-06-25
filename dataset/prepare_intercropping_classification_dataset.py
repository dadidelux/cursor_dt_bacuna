import os
import shutil
import random

SRC_DIR = 'datasets-source/intercropping'
BASE_DST_DIR = 'intercropping_classification'
TRAIN_DIR = os.path.join(BASE_DST_DIR, 'train')
VAL_DIR = os.path.join(BASE_DST_DIR, 'validation')
SPLIT_RATIO = 0.8  # 80% train, 20% validation

# Create target directories for each class
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

for class_name in os.listdir(SRC_DIR):
    class_path = os.path.join(SRC_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    dst_train_class = os.path.join(TRAIN_DIR, class_name)
    dst_val_class = os.path.join(VAL_DIR, class_name)
    os.makedirs(dst_train_class, exist_ok=True)
    os.makedirs(dst_val_class, exist_ok=True)
    images = [f for f in os.listdir(class_path) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    for fname in train_imgs:
        src_img = os.path.join(class_path, fname)
        dst_img = os.path.join(dst_train_class, fname)
        shutil.copy2(src_img, dst_img)
    for fname in val_imgs:
        src_img = os.path.join(class_path, fname)
        dst_img = os.path.join(dst_val_class, fname)
        shutil.copy2(src_img, dst_img)

print(f"All intercropping images split into train and validation folders for classification training.") 