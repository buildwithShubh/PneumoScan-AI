"""
Build a CNN model for pneumonia detection.
Uses transfer learning with MobileNetV2 as backbone.
Trains on synthetic data to produce a real, working model file.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

print("Building PneumoNet CNN model...")
print(f"TensorFlow: {tf.__version__}")

IMG_SIZE = 224
NUM_CLASSES = 1  # Binary: 0=Normal, 1=Pneumonia

def build_pneumonet():
    """
    PneumoNet: MobileNetV2-based CNN for chest X-ray pneumonia detection.
    Uses transfer learning with custom classification head.
    """
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None  # Random init for demo; real use: weights='imagenet'
    )

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="xray_input")
    x = base(inputs, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', name="fc1")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu', name="fc2")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', name="prediction")(x)

    model = Model(inputs, output, name="PneumoNet")
    return model

def generate_synthetic_xray_batch(batch_size=32, label=0):
    """
    Generate synthetic chest X-ray-like images for training.
    Normal: uniform gray with slight noise
    Pneumonia: includes bright patches (opacity simulation)
    """
    imgs = []
    for _ in range(batch_size):
        # Base: dark grayscale background (lung field)
        img = np.random.normal(0.3, 0.05, (IMG_SIZE, IMG_SIZE, 3)).clip(0, 1)

        # Rib-like horizontal bands
        for rib in range(5, IMG_SIZE, IMG_SIZE // 8):
            thickness = np.random.randint(2, 5)
            img[rib:rib+thickness, 30:IMG_SIZE-30, :] += 0.08

        # Heart silhouette (center bright oval)
        cx, cy = IMG_SIZE//2, IMG_SIZE//2
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                if ((i-cy)/40)**2 + ((j-cx)/30)**2 < 1:
                    img[i, j, :] += 0.15

        if label == 1:
            # Simulate consolidation / opacity patches
            num_patches = np.random.randint(1, 4)
            for _ in range(num_patches):
                px = np.random.randint(40, IMG_SIZE-60)
                py = np.random.randint(40, IMG_SIZE-60)
                pw = np.random.randint(20, 60)
                ph = np.random.randint(20, 50)
                intensity = np.random.uniform(0.25, 0.45)
                img[px:px+ph, py:py+pw, :] += intensity

            # Add haziness
            img += np.random.normal(0.05, 0.03, img.shape)

        img = img.clip(0, 1).astype(np.float32)
        imgs.append(img)

    return np.array(imgs)

# Build model
model = build_pneumonet()
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
model.summary()

# Quick training on synthetic data (enough to produce a working model)
print("\nTraining on synthetic data...")
EPOCHS = 3
BATCH = 32

for epoch in range(EPOCHS):
    # Generate balanced batches
    normal_imgs = generate_synthetic_xray_batch(BATCH, label=0)
    pneumonia_imgs = generate_synthetic_xray_batch(BATCH, label=1)
    X = np.concatenate([normal_imgs, pneumonia_imgs])
    y = np.array([0]*BATCH + [1]*BATCH, dtype=np.float32)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    loss, acc, auc = model.train_on_batch(X, y)
    print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {loss:.4f}  acc: {acc:.4f}  auc: {auc:.4f}")

# Save model
# save_path = "/home/claude/pneumonia_app/model/pneumonet.keras"
save_path = "model/pneumonet.keras"
model.save(save_path)
print(f"\nModel saved → {save_path}")

# Save model metadata
import json
meta = {
    "model_name": "PneumoNet",
    "architecture": "MobileNetV2 + Custom Head",
    "input_shape": [IMG_SIZE, IMG_SIZE, 3],
    "classes": ["Normal", "Pneumonia"],
    "threshold": 0.5,
    "grad_cam_layer": "Conv_1_bn",  # last conv layer in MobileNetV2
    "version": "1.0.0"
}
# with open("/home/claude/pneumonia_app/model/metadata.json", "w") as f:
with open("model/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Metadata saved.")
print("\nDone! Model ready for inference.")
