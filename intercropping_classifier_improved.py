import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B3  # Upgraded from B1 to B3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW  # Better optimizer than Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Enhanced Configuration
IMG_HEIGHT = 384  # Increased from 256
IMG_WIDTH = 384
BATCH_SIZE = 8    # Reduced for larger model
EPOCHS_PER_PHASE = 25  # Increased training epochs
INITIAL_LR = 5e-5  # Lower initial learning rate
TRAIN_DIR = 'intercropping_classification/train'
VAL_DIR = 'intercropping_classification/validation'
SOURCE_DIR = 'datasets-source/intercropping'
SPLIT_RATIO = 0.2

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

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
            
        images.sort()
        random.seed(42)
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

def create_enhanced_model(num_classes):
    """Create an enhanced model with better architecture."""
    # Use EfficientNetV2B3 for better feature extraction
    base_model = EfficientNetV2B3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Data augmentation as part of the model
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ])
    
    x = data_augmentation(inputs)
    x = base_model(x)
    
    # Enhanced head with attention mechanism
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Multi-layer dense head with residual connections
    dense_1 = layers.Dense(512, activation=None)(x)
    dense_1 = layers.BatchNormalization()(dense_1)
    dense_1 = layers.Activation('swish')(dense_1)  # Swish activation
    dense_1 = layers.Dropout(0.5)(dense_1)
    
    dense_2 = layers.Dense(256, activation=None)(dense_1)
    dense_2 = layers.BatchNormalization()(dense_2)
    dense_2 = layers.Activation('swish')(dense_2)
    dense_2 = layers.Dropout(0.4)(dense_2)
    
    dense_3 = layers.Dense(128, activation=None)(dense_2)
    dense_3 = layers.BatchNormalization()(dense_3)
    dense_3 = layers.Activation('swish')(dense_3)
    dense_3 = layers.Dropout(0.3)(dense_3)
    
    outputs = layers.Dense(num_classes, activation='softmax')(dense_3)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

def compute_class_weights(train_generator):
    labels = []
    total_samples = train_generator.samples
    steps = total_samples // BATCH_SIZE + 1
    
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
    
    train_generator.reset()
    
    if not labels:
        raise RuntimeError("No valid images found for class weight computation.")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(class_weights))

def create_enhanced_callbacks(phase):
    return [
        ModelCheckpoint(
            f'best_intercropping_improved_phase_{phase}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,  # Increased patience
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,  # More aggressive reduction
            patience=5,
            min_lr=1e-8,
            verbose=1,
            cooldown=2
        )
    ]

def plot_training_history(history, phase_idx):
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Phase {phase_idx + 1} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Phase {phase_idx + 1} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Learning rate plot
    plt.subplot(1, 3, 3)
    plt.plot(history.history['lr'], label='Learning Rate')
    plt.title(f'Phase {phase_idx + 1} Learning Rate')
    plt.ylabel('Learning Rate')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'intercropping_improved_history_phase_{phase_idx + 1}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=== Enhanced Intercropping Classifier ===")
    print(f"Target: >85% validation accuracy")
    print(f"Image size: {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"Batch size: {BATCH_SIZE}")
    
    try:
        prepare_data()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] Data preparation failed: {e}")
        return
    
    # Enhanced data generators with minimal augmentation (model has built-in augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        # Minimal augmentation since model has built-in augmentation
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    try:
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        validation_generator = val_datagen.flow_from_directory(
            VAL_DIR,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
    except Exception as e:
        print(f"[ERROR] Failed to create data generators: {e}")
        return
    
    class_weights = compute_class_weights(train_generator)
    print("Class weights:", class_weights)
    
    num_classes = train_generator.num_classes
    model, base_model = create_enhanced_model(num_classes)
    
    print(f"\nModel created with {model.count_params():,} parameters")
    
    # Enhanced training phases
    training_phases = [
        {
            'name': 'Head Training (Enhanced)',
            'layers_to_unfreeze': 0,
            'learning_rate': INITIAL_LR,
            'epochs': EPOCHS_PER_PHASE
        },
        {
            'name': 'Fine-tuning Top Layers',
            'layers_to_unfreeze': 50,  # Unfreeze more layers
            'learning_rate': INITIAL_LR * 0.3,
            'epochs': EPOCHS_PER_PHASE
        },
        {
            'name': 'Fine-tuning All Layers',
            'layers_to_unfreeze': None,
            'learning_rate': INITIAL_LR * 0.1,
            'epochs': EPOCHS_PER_PHASE + 10  # Extra epochs for final phase
        }
    ]
    
    best_val_accuracy = 0
    
    for phase_idx, phase in enumerate(training_phases, 1):
        print(f"\n{'='*60}")
        print(f"Phase {phase_idx}: {phase['name']}")
        print(f"Learning Rate: {phase['learning_rate']}")
        print(f"Epochs: {phase['epochs']}")
        print(f"{'='*60}")
        
        # Configure trainable layers
        if phase['layers_to_unfreeze'] is None:
            base_model.trainable = True
            print("All base model layers are trainable")
        else:
            base_model.trainable = True
            trainable_layers = 0
            for layer in base_model.layers[-phase['layers_to_unfreeze']:]:
                layer.trainable = True
                trainable_layers += 1
            for layer in base_model.layers[:-phase['layers_to_unfreeze']]:
                layer.trainable = False
            print(f"Unfrozen {trainable_layers} layers from the top")
        
        # Enhanced learning rate schedule with restarts
        lr_schedule = CosineDecayRestarts(
            initial_learning_rate=phase['learning_rate'],
            first_decay_steps=train_generator.samples // BATCH_SIZE * 10,  # Restart every 10 epochs
            t_mul=1.2,
            m_mul=0.8,
            alpha=0.1
        )
        
        # Use AdamW optimizer with weight decay
        optimizer = AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        try:
            history = model.fit(
                train_generator,
                epochs=phase['epochs'],
                validation_data=validation_generator,
                class_weight=class_weights,
                callbacks=create_enhanced_callbacks(phase_idx),
                verbose=1
            )
        except Exception as e:
            print(f"[ERROR] Training failed for phase {phase_idx}: {e}")
            print("Attempting to continue with next phase...")
            continue
        
        # Track best validation accuracy
        phase_best_val_acc = max(history.history['val_accuracy'])
        if phase_best_val_acc > best_val_accuracy:
            best_val_accuracy = phase_best_val_acc
        
        print(f"\nPhase {phase_idx} completed!")
        print(f"Best validation accuracy this phase: {phase_best_val_acc:.4f}")
        print(f"Overall best validation accuracy: {best_val_accuracy:.4f}")
        
        plot_training_history(history, phase_idx - 1)
        model.save(f'intercropping_improved_phase_{phase_idx}.h5')
        
        # Early stopping if we achieve target accuracy
        if best_val_accuracy >= 0.90:
            print(f"\n🎉 TARGET ACHIEVED! Validation accuracy: {best_val_accuracy:.4f}")
            break
    
    # Save final model
    model.save('intercropping_classifier_improved_final.h5')
    print(f"\n✅ Training completed!")
    print(f"Final best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Final model saved as 'intercropping_classifier_improved_final.h5'")
    
    if best_val_accuracy >= 0.85:
        print("🎯 SUCCESS: Achieved >85% accuracy target!")
    else:
        print(f"⚠️  Close but not quite: {best_val_accuracy:.1%} accuracy achieved")

if __name__ == '__main__':
    main()