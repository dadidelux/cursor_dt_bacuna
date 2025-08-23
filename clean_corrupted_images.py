import os
import shutil
from PIL import Image

SOURCE_DIR = 'datasets-source/intercropping'
CORRUPT_DIR = 'datasets-source/intercropping_corrupted'
IMG_SIZE = (224, 224)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def move_corrupted_image(src_path, class_name, img_name):
    dest_class_dir = os.path.join(CORRUPT_DIR, class_name)
    ensure_dir(dest_class_dir)
    dest_path = os.path.join(dest_class_dir, img_name)
    shutil.move(src_path, dest_path)
    print(f"Moved corrupted image: {src_path} -> {dest_path}")

def scan_and_move_corrupted():
    num_checked = 0
    num_corrupt = 0
    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if not os.path.isfile(img_path):
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify image integrity
                # Try to fully load and resize the image
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(IMG_SIZE)
            except Exception as e:
                move_corrupted_image(img_path, class_name, img_name)
                num_corrupt += 1
            num_checked += 1
    print(f"Checked {num_checked} images. Found and moved {num_corrupt} corrupted images.")

if __name__ == '__main__':
    scan_and_move_corrupted() 