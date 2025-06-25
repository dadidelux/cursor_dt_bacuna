import os
import shutil

# Source and target directories
SRC_DIR = 'datasets-source/intercropping'
DST_DIR = 'intercropping_yolo'
IMG_DST = os.path.join(DST_DIR, 'images', 'train')
LBL_DST = os.path.join(DST_DIR, 'labels', 'train')

# Create target directories
os.makedirs(IMG_DST, exist_ok=True)
os.makedirs(LBL_DST, exist_ok=True)

# Supported image extensions
IMG_EXTS = {'.jpg', '.jpeg', '.png'}

# Copy images and create empty label files
for class_name in os.listdir(SRC_DIR):
    class_path = os.path.join(SRC_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    for fname in os.listdir(class_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMG_EXTS:
            src_img = os.path.join(class_path, fname)
            dst_img = os.path.join(IMG_DST, f'{class_name}_{fname}')
            shutil.copy2(src_img, dst_img)
            # Create empty label file for annotation
            label_name = os.path.splitext(f'{class_name}_{fname}')[0] + '.txt'
            open(os.path.join(LBL_DST, label_name), 'a').close()

print(f"Images copied to {IMG_DST} and empty label files created in {LBL_DST}.")
print("You can now annotate the images using a tool like LabelImg in YOLO format.") 