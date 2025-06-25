import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import sys
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to models
INTERCROPPING_MODEL_PATH = 'intercropping_classifier.h5'
COCONUT_DISEASE_MODEL_PATH = 'best_model_phase_3.h5'

# Class names (update if needed)
INTERCROPPING_CLASSES = [
    "Kabugaw", "Jack Fruit", "Narra Tree", "Kalamansi", "Guava", "Rambutan", "Mahugani", "Guyabano", "Cacao", "Paper Tree"
]
COCONUT_DISEASE_CLASSES = [
    "Healthy", "Coconut_Leaf_Blight", "Coconut_Leaf_Spot", "Coconut_Leaf_Yellowing"
]

IMG_SIZE = (224, 224)
THRESHOLD = 0.3

def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def infer_intercropping(img_path):
    model = tf.keras.models.load_model(INTERCROPPING_MODEL_PATH)
    x = load_and_preprocess(img_path)
    preds = model.predict(x)[0]
    present = [INTERCROPPING_CLASSES[i] for i, p in enumerate(preds) if p > THRESHOLD]
    return present, preds

def infer_coconut_disease(img_path):
    model = tf.keras.models.load_model(COCONUT_DISEASE_MODEL_PATH)
    x = load_and_preprocess(img_path)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return COCONUT_DISEASE_CLASSES[idx], preds

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python infer_intercropping_and_coconut_disease.py <image_path>")
        sys.exit(1)
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        sys.exit(1)
    print(f"\nAnalyzing: {os.path.basename(img_path)}\n")
    plants, plant_probs = infer_intercropping(img_path)
    disease, disease_probs = infer_coconut_disease(img_path)
    print("Intercropping plants detected:")
    if plants:
        for p in plants:
            print(f"  - {p}")
    else:
        print("  None detected (threshold={THRESHOLD})")
    print(f"\nCoconut disease detected: {disease}")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # 20% for validation
        ...
    )
    train_generator = train_datagen.flow_from_directory(
        ...,
        subset='training',
    )
    val_generator = train_datagen.flow_from_directory(
        ...,
        subset='validation',
    ) 