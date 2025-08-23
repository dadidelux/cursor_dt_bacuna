import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Configuration - IMPROVED HYPERPARAMETERS
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8  # Reduced from 16 for more frequent updates
EPOCHS_PER_PHASE = 25  # Increased from 15
INITIAL_LR = 1e-3  # Increased from 1e-4 for faster learning
TRAIN_DIR = 'intercropping_classification/train'
VAL_DIR = 'intercropping_classification/validation'
SOURCE_DIR = 'datasets-source/intercropping'
SPLIT_RATIO = 0.2
EXISTING_MODEL_PATH = 'intercropping_classifier_final.h5'
NEW_MODEL_PATH = 'intercropping_classifier_transfer_learning_v2.h5'


def clean_corrupted_images():
    """Clean corrupted images from the source directory before training."""
    from PIL import Image
    import os
    
    corrupted_count = 0
    total_count = 0
    
    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if not os.path.isfile(img_path):
                continue
                
            total_count += 1
            try:
                # Try to open and process the image
                with Image.open(img_path) as img:
                    img.verify()  # Verify the image
                    img.close()
                    
                # Try to load and resize (more aggressive check)
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img.resize((IMG_HEIGHT, IMG_WIDTH))
                    
            except Exception as e:
                print(f"Corrupted image found: {img_path} - {e}")
                try:
                    os.remove(img_path)
                    corrupted_count += 1
                except:
                    print(f"Could not remove corrupted image: {img_path}")
    
    print(f"Cleaned {corrupted_count} corrupted images out of {total_count} total images")


def prepare_data():
    """Split images from SOURCE_DIR into train/validation folders."""
    # First clean corrupted images
    print("Cleaning corrupted images...")
    clean_corrupted_images()
    
    if os.path.exists(TRAIN_DIR):
        shutil.rmtree(TRAIN_DIR)
    if os.path.exists(VAL_DIR):
        shutil.rmtree(VAL_DIR)
    os.makedirs(TRAIN_DIR)
    os.makedirs(VAL_DIR)

    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)
        split_idx = int(len(images) * (1 - SPLIT_RATIO))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        val_class_dir = os.path.join(VAL_DIR, class_name)
        os.makedirs(train_class_dir)
        os.makedirs(val_class_dir)

        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_class_dir, img))
    print(f"Data prepared: {TRAIN_DIR} and {VAL_DIR}")


def load_existing_model():
    """Load the existing trained model for transfer learning."""
    if not os.path.exists(EXISTING_MODEL_PATH):
        print(f"Warning: {EXISTING_MODEL_PATH} not found. Creating new model from scratch.")
        return create_new_model()
    
    print(f"Loading existing model from {EXISTING_MODEL_PATH}")
    
    # Try to load with custom_objects to handle compatibility issues
    try:
        model = tf.keras.models.load_model(EXISTING_MODEL_PATH)
        print(f"Loaded model with {len(model.layers)} layers")
        print(f"Model output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"Error loading existing model: {e}")
        print("This might be due to TensorFlow version compatibility issues.")
        print("Creating new model from scratch instead...")
        return create_new_model()


def create_new_model():
    """Create a new model from scratch (fallback if existing model not found)."""
    print("Creating new model from scratch...")
    base_model = EfficientNetV2B0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    block_1 = layers.Dense(256, activation=None)(x)
    block_1 = layers.BatchNormalization()(block_1)
    block_1 = layers.Activation('relu')(block_1)
    block_1 = layers.Dropout(0.3)(block_1)
    block_2 = layers.Dense(256, activation=None)(block_1)
    block_2 = layers.BatchNormalization()(block_2)
    block_2 = layers.Activation('relu')(block_2)
    block_2 = layers.Dropout(0.3)(block_2)
    residual = layers.Add()([block_1, block_2])
    outputs = layers.Dense(15, activation='softmax')(residual)  # Updated to 15 classes
    model = tf.keras.Model(inputs, outputs)
    return model


def adapt_model_for_new_data(model, num_classes):
    """Adapt the model if the number of classes has changed."""
    if model.output_shape[-1] == num_classes:
        print(f"Model already has {num_classes} classes. No adaptation needed.")
        return model
    
    print(f"Adapting model from {model.output_shape[-1]} to {num_classes} classes...")
    
    # Get the base model (EfficientNetV2B0)
    base_model = None
    for layer in model.layers:
        if 'efficientnet' in layer.name.lower():
            base_model = layer
            break
    
    if base_model is None:
        print("Could not find EfficientNet base model. Creating new model.")
        return create_new_model()
    
    # Create new classification head
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    block_1 = layers.Dense(256, activation=None)(x)
    block_1 = layers.BatchNormalization()(block_1)
    block_1 = layers.Activation('relu')(block_1)
    block_1 = layers.Dropout(0.3)(block_1)
    block_2 = layers.Dense(256, activation=None)(block_1)
    block_2 = layers.BatchNormalization()(block_2)
    block_2 = layers.Activation('relu')(block_2)
    block_2 = layers.Dropout(0.3)(block_2)
    residual = layers.Add()([block_1, block_2])
    outputs = layers.Dense(num_classes, activation='softmax')(residual)
    
    new_model = tf.keras.Model(inputs, outputs)
    return new_model


def compute_class_weights(train_generator):
    """Compute class weights for imbalanced datasets."""
    labels = []
    total_samples = train_generator.samples
    steps = total_samples // BATCH_SIZE + 1
    
    for i in range(steps):
        try:
            _, y = next(train_generator)
            labels.extend(np.argmax(y, axis=1))
        except OSError as e:
            print(f"[WARNING] Skipping batch due to error: {e}")
            continue
        if len(labels) >= total_samples:
            break
    
    if not labels:
        raise RuntimeError("No valid images found for class weight computation.")
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    class_weight_dict = dict(zip(np.unique(labels), class_weights))
    print(f"Class weights: {class_weight_dict}")
    return class_weight_dict


def create_callbacks(phase):
    """Create callbacks for training with improved settings."""
    return [
        ModelCheckpoint(
            f'best_intercropping_transfer_v2_phase_{phase}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # Increased from 12
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.7,  # Less aggressive reduction (was 0.5)
            patience=8,  # Increased from 5
            min_lr=1e-7,
            verbose=1
        )
    ]


def plot_training_history(history, phase_idx):
    """Plot training history for each phase."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Transfer Learning V2 Phase {phase_idx + 1} Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Transfer Learning V2 Phase {phase_idx + 1} Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'intercropping_transfer_learning_v2_history_phase_{phase_idx + 1}.png')
    plt.close()


def main():
    """Main training function with improved transfer learning strategy."""
    print("Starting IMPROVED Transfer Learning Training...")
    
    # Prepare data
    prepare_data()
    
    # Setup data generators with ENHANCED augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # Increased from 20
        width_shift_range=0.3,  # Increased from 0.2
        height_shift_range=0.3,  # Increased from 0.2
        shear_range=0.3,  # Increased from 0.2
        zoom_range=0.3,  # Increased from 0.2
        horizontal_flip=True,
        vertical_flip=True,  # Added vertical flip
        brightness_range=[0.8, 1.2],  # Added brightness variation
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    num_classes = len(train_generator.class_indices)
    print(f'Classes: {train_generator.class_indices}')
    print(f'Number of classes: {num_classes}')
    
    # Load existing model
    model = load_existing_model()
    
    # Adapt model if needed
    model = adapt_model_for_new_data(model, num_classes)
    
    # Compute class weights
    class_weights = compute_class_weights(train_generator)
    
    # IMPROVED training phases for transfer learning
    training_phases = [
        {
            'name': 'Fine-tune Classification Head Only (Higher LR)',
            'unfreeze_layers': 0,
            'learning_rate': INITIAL_LR,
            'epochs': EPOCHS_PER_PHASE
        },
        {
            'name': 'Fine-tune Top EfficientNet Layers (Medium LR)',
            'unfreeze_layers': 50,  # Increased from 30
            'learning_rate': INITIAL_LR * 0.1,
            'epochs': EPOCHS_PER_PHASE
        },
        {
            'name': 'Fine-tune All Layers (Low LR)',
            'unfreeze_layers': None,  # Unfreeze all
            'learning_rate': INITIAL_LR * 0.01,
            'epochs': EPOCHS_PER_PHASE
        }
    ]
    
    # Train through each phase
    for phase_idx, phase in enumerate(training_phases):
        print(f"\n{'='*60}")
        print(f"Phase {phase_idx + 1}: {phase['name']}")
        print(f"{'='*60}")
        
        # Configure model for this phase
        if phase['unfreeze_layers'] is None:
            # Unfreeze all layers
            for layer in model.layers:
                layer.trainable = True
        elif phase['unfreeze_layers'] == 0:
            # Freeze base model, train only classification head
            for layer in model.layers:
                if 'efficientnet' in layer.name.lower():
                    layer.trainable = False
                else:
                    layer.trainable = True
        else:
            # Unfreeze specific number of layers
            efficientnet_layers = []
            for layer in model.layers:
                if 'efficientnet' in layer.name.lower():
                    efficientnet_layers.append(layer)
            
            # Freeze all EfficientNet layers initially
            for layer in efficientnet_layers:
                layer.trainable = False
            
            # Unfreeze top layers
            for layer in efficientnet_layers[-phase['unfreeze_layers']:]:
                layer.trainable = True
        
        # Print trainable layers
        trainable_count = sum(1 for layer in model.layers if layer.trainable)
        print(f"Trainable layers: {trainable_count}/{len(model.layers)}")
        print(f"Learning rate: {phase['learning_rate']}")
        
        # Compile model with fixed learning rate
        model.compile(
            optimizer=Adam(learning_rate=phase['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for this phase
        try:
            history = model.fit(
                train_generator,
                epochs=phase['epochs'],
                validation_data=val_generator,
                callbacks=create_callbacks(phase_idx + 1),
                class_weight=class_weights,
                verbose=1
            )
            
            # Plot history for this phase
            plot_training_history(history, phase_idx)
            
            # Save phase model
            model.save(f'intercropping_transfer_learning_v2_phase_{phase_idx + 1}.h5')
            
        except (OSError, tf.errors.UnknownError) as e:
            print(f"[WARNING] Error during training phase {phase_idx + 1}: {e}")
            print("This might be due to corrupted images. Trying to continue...")
            
            # Try to continue with next phase if possible
            if phase_idx < len(training_phases) - 1:
                print("Continuing to next phase...")
                continue
            else:
                print("This was the final phase. Saving current model state...")
                model.save(f'intercropping_transfer_learning_v2_phase_{phase_idx + 1}_partial.h5')
                break
    
    # Save final model
    model.save(NEW_MODEL_PATH)
    print(f"\nTransfer learning completed. Final model saved as '{NEW_MODEL_PATH}'")
    
    # Save training plots
    print("Training plots saved as:")
    for i in range(len(training_phases)):
        print(f"  - intercropping_transfer_learning_v2_history_phase_{i + 1}.png")


if __name__ == "__main__":
    main() 