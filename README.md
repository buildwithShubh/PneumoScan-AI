# PneumoScan AI — Pneumonia Detection System

AI-powered chest X-ray pneumonia detection using Flask + TensorFlow/Keras + OpenCV + Grad-CAM.

## Project Structure

```
pneumonia_app/
├── app.py                  # Flask web server & routes
├── inference.py            # CNN prediction + Grad-CAM engine
├── build_model.py          # Model architecture & training script
├── requirements.txt
├── model/
│   ├── pneumonet.keras     # Trained CNN model weights
│   └── metadata.json       # Model config & class info
├── templates/
│   └── index.html          # Full-stack frontend (HTML/CSS/JS)
└── static/
    ├── uploads/            # Uploaded X-rays (temp)
    └── results/            # Saved result images
```

## Architecture: PneumoNet Lite

```
Input (128×128×3)
    ↓
Conv Block 1: Conv2D(32) → BN → MaxPool
    ↓
Conv Block 2: Conv2D(64) → BN → MaxPool
    ↓
Conv Block 3: Conv2D(128) → BN → MaxPool
    ↓
Conv Block 4: Conv2D(256) → BN  ← Grad-CAM target layer
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, relu) → Dropout(0.4)
    ↓
Dense(64, relu) → Dropout(0.3)
    ↓
Dense(1, sigmoid) → Prediction
```

**Total parameters:** ~431,000

## OpenCV Preprocessing Pipeline

1. **Image Decode** — OpenCV reads raw bytes (PNG/JPG/WEBP/BMP/TIFF)
2. **Grayscale Conversion** — Remove color artifacts
3. **CLAHE** — Contrast Limited Adaptive Histogram Equalization (clipLimit=2.0, tileGrid=8×8)
4. **Bilateral Filter** — Edge-preserving denoising (d=9, σcolor=75, σspace=75)
5. **Lanczos Resize** — Resize to 128×128 with high-quality interpolation
6. **Normalization** — Pixel values scaled to [0.0, 1.0]
7. **RGB Stacking** — Grayscale → 3-channel tensor for model input

## Grad-CAM Visualization

Uses **Gradient-weighted Class Activation Mapping**:

1. Build gradient model from input → [conv4_gradcam output, prediction]
2. Record gradients of prediction w.r.t. `conv4_gradcam` feature maps via `tf.GradientTape`
3. Pool gradients globally → importance weights per feature channel
4. Weighted sum of feature maps → activation heatmap
5. Apply ReLU → normalize → resize to input dimensions
6. Overlay on original X-ray using `cv2.applyColorMap(COLORMAP_JET)` + alpha blending

**Color interpretation:**
- 🔵 Blue/Cool = Low activation (normal lung tissue)
- 🟡 Yellow = Moderate activation
- 🔴 Red/Hot = High activation (potential pneumonia region)

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the CNN model (run once)
python build_model.py

# 3. Start the Flask server
python app.py

# 4. Open browser
open http://localhost:5000
```

## API Endpoint

**POST /analyze**
- **Input:** `multipart/form-data` with `file` field (image)
- **Output:** JSON

```json
{
  "success": true,
  "result": {
    "label": "Pneumonia",
    "probability": 0.8712,
    "confidence": 87.12,
    "is_pneumonia": true,
    "risk_level": { "level": "High", "color": "#ff4757", "description": "..." },
    "enhanced_b64": "<base64 JPEG>",
    "gradcam_b64": "<base64 PNG>",
    "findings": ["..."],
    "inference_time_ms": 312.4,
    "model_name": "PneumoNet Lite",
    "architecture": "Custom 4-Block CNN",
    "preprocessing": ["CLAHE contrast enhancement", "..."]
  }
}
```
## Features

- Chest X-ray upload
- AI pneumonia prediction
- Confidence score
- OpenCV preprocessing
- Grad-CAM heatmap visualization
- Flask REST API
- TensorFlow/Keras CNN model
- Real-time inference


## Tech Stack

- Python
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib  


## For Production / Real Training

To train on real chest X-ray data:
1. Download the [Kaggle Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Update `build_model.py` to load from disk using `tf.keras.utils.image_dataset_from_directory`
3. Enable `weights='imagenet'` in MobileNetV2 for transfer learning
4. Train for 20–50 epochs with data augmentation (rotation, flip, zoom)
5. Expected accuracy: ~92–95% on real data

## ⚠ Disclaimer

This system is for **research and demonstration purposes only**.
It must NOT be used for actual clinical diagnosis.
All medical imaging must be interpreted by licensed radiologists and physicians.
**Developer** : Shubham Gupta 
Guthub: https://github.com/buildwithShubh
Repository :
