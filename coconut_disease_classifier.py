import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS_PER_PHASE = 15
INITIAL_LR = 1e-3

def create_model():
    """Create and return the model using EfficientNetV2B0 architecture with progressive unfreezing"""
    # Load EfficientNetV2B0 with pre-trained weights
    base_model = EfficientNetV2B0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Create the model with improved head
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # First dense block with residual connection
    block_1 = layers.Dense(256, activation=None)(x)
    block_1 = layers.BatchNormalization()(block_1)
    block_1 = layers.Activation('relu')(block_1)
    block_1 = layers.Dropout(0.3)(block_1)
    
    # Second dense block
    block_2 = layers.Dense(256, activation=None)(block_1)
    block_2 = layers.BatchNormalization()(block_2)
    block_2 = layers.Activation('relu')(block_2)
    block_2 = layers.Dropout(0.3)(block_2)
    
    # Residual connection (now shapes match: both 256)
    residual = layers.Add()([block_1, block_2])
    
    # Final classification layer
    outputs = layers.Dense(4, activation='softmax')(residual)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

def compute_class_weights(train_generator):
    """Compute class weights to handle class imbalance"""
    labels = []
    total_samples = train_generator.samples
    steps = total_samples // BATCH_SIZE + 1
    
    for i in range(steps):
        _, y = next(train_generator)
        labels.extend(np.argmax(y, axis=1))
        if len(labels) >= total_samples:
            break
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return dict(enumerate(class_weights))

def create_callbacks(phase):
    """Create callbacks for training with phase-specific settings"""
    return [
        ModelCheckpoint(
            f'best_model_phase_{phase}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
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
    # Create figure with two subplots
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Phase {phase_idx + 1} Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Phase {phase_idx + 1} Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'training_history_phase_{phase_idx + 1}.png')
    plt.close()

def main():
    # Data augmentation with mixup
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    # Compute class weights
    class_weights = compute_class_weights(train_generator)
    print("Class weights:", class_weights)
    
    # Create model
    model, base_model = create_model()
    
    # Print model summary
    model.summary()
    
    # Training phases with progressive unfreezing
    training_phases = [
        {
            'name': 'Head Training',
            'layers_to_unfreeze': 0,
            'learning_rate': INITIAL_LR
        },
        {
            'name': 'Fine-tuning Top Layers',
            'layers_to_unfreeze': 30,
            'learning_rate': INITIAL_LR * 0.1
        },
        {
            'name': 'Fine-tuning All Layers',
            'layers_to_unfreeze': None,  # Unfreeze all
            'learning_rate': INITIAL_LR * 0.01
        }
    ]
    
    # Train through each phase
    for phase_idx, phase in enumerate(training_phases, 1):
        print(f"\nPhase {phase_idx}: {phase['name']}")
        
        # Unfreeze layers for this phase
        if phase['layers_to_unfreeze'] is None:
            base_model.trainable = True
        else:
            base_model.trainable = True
            for layer in base_model.layers[:-phase['layers_to_unfreeze']]:
                layer.trainable = False
        
        # Learning rate schedule
        lr_schedule = ExponentialDecay(
            initial_learning_rate=phase['learning_rate'],
            decay_steps=train_generator.samples // BATCH_SIZE * 5,
            decay_rate=0.9,
            staircase=True
        )
        
        # Compile with phase-specific learning rate
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for this phase
        history = model.fit(
            train_generator,
            epochs=EPOCHS_PER_PHASE,
            validation_data=validation_generator,
            class_weight=class_weights,
            callbacks=create_callbacks(phase_idx),
            verbose=1
        )
        
        # Plot history for this phase
        plot_training_history(history, phase_idx - 1)
        
        # Save phase model
        model.save(f'model_phase_{phase_idx}.h5')
    
    # Save final model
    model.save('coconut_pest_classifier_final.h5')
    print("\nTraining completed. Final model saved as 'coconut_pest_classifier_final.h5'")

if __name__ == '__main__':
    main() 