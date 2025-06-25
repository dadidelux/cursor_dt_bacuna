import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

# Paths
TRAIN_DIR = 'intercropping_classification/train'
VAL_DIR = 'intercropping_classification/validation'
MODEL_PATH = 'intercropping_classifier.h5'
HISTORY_PATH = 'intercropping_training_history.png'

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
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
    shuffle=True
)

num_classes = len(train_generator.class_indices)
print(f'Classes: {train_generator.class_indices}')

# Model creation function
def create_model(base_model):
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

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

# Load EfficientNetV2B0 base model
base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

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

# Callbacks
callbacks = lambda phase: [
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

# Train through each phase
for phase_idx, phase in enumerate(training_phases, 1):
    print(f"\nPhase {phase_idx}: {phase['name']}")
    # Unfreeze layers for this phase
    if phase['layers_to_unfreeze'] is None:
        base_model.trainable = True
    elif phase['layers_to_unfreeze'] == 0:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-phase['layers_to_unfreeze']]:
            layer.trainable = False
    # Create model for this phase
    model = create_model(base_model)
    model.summary()
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
        validation_data=val_generator,
        callbacks=callbacks(phase_idx),
        verbose=1
    )
    # Plot history for this phase
    plot_training_history(history, phase_idx - 1)
    # Save phase model
    model.save(f'intercropping_classifier_phase_{phase_idx}.h5')

# Save final model
model.save(MODEL_PATH)
print(f"\nTraining completed. Final model saved as '{MODEL_PATH}'")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print(tf.sysconfig.get_build_info())

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights)) 