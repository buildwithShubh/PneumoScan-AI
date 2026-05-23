"""Lightweight CNN model - memory efficient"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import json

print("Building lightweight PneumoNet...")
IMG_SIZE = 128

def build_pneumonet_lite():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="xray_input")
    
    # Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name="conv1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name="conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Block 4 - last conv (for Grad-CAM)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name="conv4_gradcam")(x)
    x = layers.BatchNormalization()(x)
    
    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu', name="fc2")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', name="prediction")(x)
    
    return Model(inputs, output, name="PneumoNet_Lite")

model = build_pneumonet_lite()
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
print(f"Parameters: {model.count_params():,}")

# Generate small synthetic training data
def make_batch(n=16, label=0):
    imgs = []
    for _ in range(n):
        img = np.random.normal(0.3, 0.05, (IMG_SIZE, IMG_SIZE, 3)).clip(0,1).astype(np.float32)
        if label == 1:
            # Simulate opacity patches
            for _ in range(np.random.randint(1, 4)):
                px, py = np.random.randint(20, IMG_SIZE-40, 2)
                pw, ph = np.random.randint(10, 35, 2)
                img[px:px+ph, py:py+pw, :] = np.clip(img[px:px+ph, py:py+pw, :] + 0.35, 0, 1)
        imgs.append(img)
    return np.array(imgs)

print("Training...")
for ep in range(5):
    X = np.concatenate([make_batch(16, 0), make_batch(16, 1)])
    y = np.array([0]*16 + [1]*16, dtype=np.float32)
    idx = np.random.permutation(32)
    X, y = X[idx], y[idx]
    loss, acc, auc = model.train_on_batch(X, y)
    print(f"  Epoch {ep+1}/5 — loss:{loss:.4f}  acc:{acc:.4f}  auc:{auc:.4f}")

os.makedirs("model", exist_ok=True)
model.save("model/pneumonet.keras")

meta = {
    "model_name": "PneumoNet Lite",
    "architecture": "Custom 4-Block CNN",
    "input_shape": [IMG_SIZE, IMG_SIZE, 3],
    "classes": ["Normal", "Pneumonia"],
    "threshold": 0.5,
    "grad_cam_layer": "conv4_gradcam",
    "version": "1.0.0"
}
with open("model/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nModel saved successfully!")
print(f"Model size: {os.path.getsize('model/pneumonet.keras') / 1024:.1f} KB")
