"""
PneumoNet - Real Kaggle Dataset Training
Dataset: Chest X-Ray Images (Pneumonia)
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ─── CONFIG ──────────────────────────────────────────
IMG_SIZE    = 128        # image resize dimension
BATCH_SIZE  = 32         # images per batch
EPOCHS      = 20         # max training epochs
LR          = 1e-3       # learning rate

TRAIN_DIR   = "chest_xray/train"
VAL_DIR     = "chest_xray/val"
TEST_DIR    = "chest_xray/test"
MODEL_DIR   = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── DATA AUGMENTATION ───────────────────────────────
# Training data - augmentation for better generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,          # slight rotation
    width_shift_range=0.1,      # horizontal shift
    height_shift_range=0.1,     # vertical shift
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,       # flip X-rays horizontally
    fill_mode='nearest'
)

# Validation/Test - only rescale, no augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

# ─── DATA LOADERS ────────────────────────────────────
print("\nLoading dataset...")

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',        # 0=NORMAL, 1=PNEUMONIA
    color_mode='rgb',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

test_gen = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

print(f"\nClass mapping: {train_gen.class_indices}")
print(f"Train samples:      {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Test samples:       {test_gen.samples}")

# ─── CLASS WEIGHTS ───────────────────────────────────
# Dataset is imbalanced (more Pneumonia than Normal)
# Class weights fix this automatically
total = train_gen.samples
n_normal    = len(os.listdir(os.path.join(TRAIN_DIR, 'NORMAL')))
n_pneumonia = len(os.listdir(os.path.join(TRAIN_DIR, 'PNEUMONIA')))

weight_normal    = total / (2 * n_normal)
weight_pneumonia = total / (2 * n_pneumonia)

class_weights = {0: weight_normal, 1: weight_pneumonia}
print(f"\nClass weights: Normal={weight_normal:.2f}, Pneumonia={weight_pneumonia:.2f}")

# ─── MODEL ARCHITECTURE ──────────────────────────────
def build_pneumonet(img_size=128):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="xray_input")

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', name="conv1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', name="conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 4 - Grad-CAM target layer
    x = layers.Conv2D(256, 3, padding='same', name="conv4_gradcam")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu', name="fc2")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', name="prediction")(x)

    return Model(inputs, output, name="PneumoNet")

model = build_pneumonet(IMG_SIZE)
model.summary()

# ─── COMPILE ─────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# ─── CALLBACKS ───────────────────────────────────────
callbacks = [
    # Save best model automatically
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'pneumonet.keras'),
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Stop early if no improvement for 5 epochs
    EarlyStopping(
        monitor='val_auc',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    # TensorBoard logging
    TensorBoard(log_dir='logs/', histogram_freq=1)
]

# ─── TRAIN ───────────────────────────────────────────
print("\n" + "="*50)
print("Starting Training...")
print("="*50)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ─── EVALUATE ON TEST SET ────────────────────────────
print("\n" + "="*50)
print("Evaluating on Test Set...")
print("="*50)

results = model.evaluate(test_gen, verbose=1)
# metrics = dict(zip(model.metrics_names, results))
metrics = dict(zip(model.metrics_names, results))
print("Available metrics:", metrics.keys())  # add this line

# print(f"\n  Test Accuracy:  {metrics['accuracy']*100:.2f}%")
# print(f"  Test AUC:       {metrics['auc']:.4f}")
# print(f"  Test Precision: {metrics['precision']*100:.2f}%")
# print(f"  Test Recall:    {metrics['recall']*100:.2f}%")
# Get metric values safely by position
test_loss      = results[0]
test_accuracy  = results[1]
test_auc       = results[2]
test_precision = results[3]  
test_recall    = results[4]

print(f"\n  Test Loss:      {test_loss:.4f}")
print(f"  Test Accuracy:  {test_accuracy*100:.2f}%")
print(f"  Test AUC:       {test_auc:.4f}")
print(f"  Test Precision: {test_precision*100:.2f}%")
print(f"  Test Recall:    {test_recall*100:.2f}%")

# ─── SAVE METADATA ───────────────────────────────────
meta = {
    "model_name": "PneumoNet",
    "architecture": "Custom 4-Block CNN",
    "input_shape": [IMG_SIZE, IMG_SIZE, 3],
    "classes": ["Normal", "Pneumonia"],
    "threshold": 0.5,
    "grad_cam_layer": "conv4_gradcam",
    "version": "2.0.0",
    "trained_on": "Kaggle Chest X-Ray Dataset",
    "train_samples": train_gen.samples,
    "test_accuracy": round(test_accuracy*100, 2),
    "test_auc": round(test_auc, 4)
#     "test_accuracy": round(metrics['accuracy']*100, 2),
#     "test_auc": round(metrics['auc'], 4)
}

with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print("\nMetadata saved!")

# ─── PLOT TRAINING CURVES ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(history.history['auc'], label='Train')
axes[2].plot(history.history['val_auc'], label='Validation')
axes[2].set_title('AUC Score')
axes[2].set_xlabel('Epoch')
axes[2].legend()
axes[2].grid(True)

plt.suptitle('PneumoNet Training Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Training curves saved → training_curves.png")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print(f"Model saved → {MODEL_DIR}/pneumonet.keras")
print(f"Test Accuracy: {metrics['accuracy']*100:.2f}%")
print("="*50)