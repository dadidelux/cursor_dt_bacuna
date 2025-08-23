import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Configuration
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
EPOCHS_PER_PHASE = 20
INITIAL_LR = 1e-4
TRAIN_DIR = 'intercropping_classification/train'
VAL_DIR = 'intercropping_classification/validation'
SOURCE_DIR = 'datasets-source/intercropping'
SPLIT_RATIO = 0.2  # 20% validation


def prepare_data():
    """Split images from SOURCE_DIR into train/validation folders."""
    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"Source directory '{SOURCE_DIR}' does not exist.")
    
    if os.path.exists(TRAIN_DIR):
        shutil.rmtree(TRAIN_DIR)
    if os.path.exists(VAL_DIR):
        shutil.rmtree(VAL_DIR)
    os.makedirs(TRAIN_DIR)
    os.makedirs(VAL_DIR)

    classes_found = 0
    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        
        images = [f for f in os.listdir(class_path) 
                 if os.path.isfile(os.path.join(class_path, f)) and 
                 f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        if not images:
            print(f"[WARNING] No valid images found in class '{class_name}'")
            continue
            
        # Sort first to ensure reproducible splits
        images.sort()
        random.seed(42)  # Fixed seed for reproducible splits
        random.shuffle(images)
        split_idx = int(len(images) * (1 - SPLIT_RATIO))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        val_class_dir = os.path.join(VAL_DIR, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_class_dir, img))
        
        classes_found += 1
        print(f"Processed class '{class_name}': {len(train_images)} train, {len(val_images)} val images")
    
    if classes_found == 0:
        raise RuntimeError("No valid image classes found in source directory.")
    
    print(f"Data prepared: {TRAIN_DIR} and {VAL_DIR} with {classes_found} classes")


def create_model(num_classes):
    base_model = EfficientNetV2B1(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    block_1 = layers.Dense(128, activation=None)(x)
    block_1 = layers.BatchNormalization()(block_1)
    block_1 = layers.Activation('relu')(block_1)
    block_1 = layers.Dropout(0.5)(block_1)
    block_2 = layers.Dense(128, activation=None)(block_1)
    block_2 = layers.BatchNormalization()(block_2)
    block_2 = layers.Activation('relu')(block_2)
    block_2 = layers.Dropout(0.5)(block_2)
    residual = layers.Add()([block_1, block_2])
    x = layers.Dropout(0.4)(residual)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def compute_class_weights(train_generator):
    labels = []
    total_samples = train_generator.samples
    steps = total_samples // BATCH_SIZE + 1
    
    # Reset generator to ensure we start from the beginning
    train_generator.reset()
    
    for i in range(steps):
        try:
            _, y = next(train_generator)
            labels.extend(np.argmax(y, axis=1))
        except (OSError, StopIteration) as e:
            print(f"[WARNING] Skipping batch due to error: {e}")
            continue
        if len(labels) >= total_samples:
            break
    
    # Reset generator again for training
    train_generator.reset()
    
    if not labels:
        raise RuntimeError("No valid images found for class weight computation.")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(class_weights))


def create_callbacks(phase):
    return [
        ModelCheckpoint(
            f'best_intercropping_model_phase_{phase}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]


def plot_training_history(history, phase_idx):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Phase {phase_idx + 1} Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Phase {phase_idx + 1} Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(f'intercropping_training_history_phase_{phase_idx + 1}.png')
    plt.close()


def main():
    try:
        prepare_data()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] Data preparation failed: {e}")
        return
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        channel_shift_range=20.0,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    try:
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )
        validation_generator = val_datagen.flow_from_directory(
            VAL_DIR,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )
    except Exception as e:
        print(f"[ERROR] Failed to create data generators: {e}")
        return
    class_weights = compute_class_weights(train_generator)
    print("Class weights:", class_weights)
    num_classes = train_generator.num_classes
    model, base_model = create_model(num_classes)
    model.summary()
    training_phases = [
        {
            'name': 'Head Training',
            'layers_to_unfreeze': 0,
            'learning_rate': INITIAL_LR
        },
        {
            'name': 'Fine-tuning Top Layers',
            'layers_to_unfreeze': 30,
            'learning_rate': INITIAL_LR * 0.5
        },
        {
            'name': 'Fine-tuning All Layers',
            'layers_to_unfreeze': None,
            'learning_rate': INITIAL_LR * 0.1
        }
    ]
    for phase_idx, phase in enumerate(training_phases, 1):
        print(f"\nPhase {phase_idx}: {phase['name']}")
        if phase['layers_to_unfreeze'] is None:
            base_model.trainable = True
        else:
            base_model.trainable = True
            for layer in base_model.layers[:-phase['layers_to_unfreeze']]:
                layer.trainable = False
        lr_schedule = CosineDecay(
            initial_learning_rate=phase['learning_rate'],
            decay_steps=train_generator.samples // BATCH_SIZE * EPOCHS_PER_PHASE,
            alpha=0.1
        )
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        try:
            history = model.fit(
                train_generator,
                epochs=EPOCHS_PER_PHASE,
                validation_data=validation_generator,
                class_weight=class_weights,
                callbacks=create_callbacks(phase_idx),
                verbose=1
            )
        except Exception as e:
            print(f"[ERROR] Training failed for phase {phase_idx}: {e}")
            print("Attempting to continue with next phase...")
            continue
        plot_training_history(history, phase_idx - 1)
        model.save(f'intercropping_model_phase_{phase_idx}.h5')
    model.save('intercropping_classifier_final.h5')
    print("\nTraining completed. Final model saved as 'intercropping_classifier_final.h5'")

def predict_with_tta(model, image_path, num_augmentations=8):
    """Predict with Test Time Augmentation for better accuracy."""
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import numpy as np
    
    # Load and preprocess image
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    
    predictions = []
    
    # Original image
    predictions.append(model.predict(np.expand_dims(img_array, axis=0), verbose=0))
    
    # Augmented versions
    tta_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )
    
    for _ in range(num_augmentations - 1):
        augmented = tta_datagen.random_transform(img_array)
        predictions.append(model.predict(np.expand_dims(augmented, axis=0), verbose=0))
    
    # Average predictions
    avg_prediction = np.mean(predictions, axis=0)
    predicted_class = np.argmax(avg_prediction, axis=1)[0]
    confidence = np.max(avg_prediction)
    
    return predicted_class, confidence, avg_prediction


def evaluate_model_with_tta(model_path, test_dir, num_augmentations=8):
    """Evaluate model using Test Time Augmentation."""
    import os
    from tensorflow.keras.models import load_model
    
    model = load_model(model_path)
    
    correct_predictions = 0
    total_predictions = 0
    
    # Get class names
    class_names = sorted(os.listdir(test_dir))
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for image_file in os.listdir(class_dir):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(class_dir, image_file)
                predicted_class, confidence, _ = predict_with_tta(
                    model, image_path, num_augmentations
                )
                
                if predicted_class == class_idx:
                    correct_predictions += 1
                total_predictions += 1
                
                if total_predictions % 50 == 0:
                    print(f"Evaluated {total_predictions} images...")
    
    accuracy = correct_predictions / total_predictions
    print(f"\nTTA Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    return accuracy


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--tta-eval':
        # Evaluate with TTA
        model_path = 'intercropping_classifier_final.h5'
        test_dir = 'intercropping_classification/validation'
        if os.path.exists(model_path) and os.path.exists(test_dir):
            evaluate_model_with_tta(model_path, test_dir)
        else:
            print("Model or test directory not found. Train model first.")
    else:
        # Regular training
        main() 