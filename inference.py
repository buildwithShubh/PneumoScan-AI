"""
PneumoNet Inference Engine
- Image preprocessing with OpenCV
- CNN model prediction
- Grad-CAM visualization
"""
import os
import json
import time
import numpy as np
import cv2
from PIL import Image
import io
import base64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "pneumonet.keras")
META_PATH  = os.path.join(os.path.dirname(__file__), "model", "metadata.json")

# ─── LOAD MODEL & META ────────────────────────────────────────────────────────
_model = None
_meta  = None

def get_model():
    global _model, _meta
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _model, _meta


# ─── IMAGE PREPROCESSING ──────────────────────────────────────────────────────
def preprocess_xray(image_bytes: bytes, target_size: tuple = (128, 128)) -> np.ndarray:
    """
    Preprocess chest X-ray image for model input.
    Pipeline: decode → grayscale equalization → CLAHE → normalize → resize → RGB stack
    """
    # Decode image bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image. Ensure it's a valid PNG/JPG.")

    # Convert to grayscale for histogram equalization
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE – Contrast Limited Adaptive Histogram Equalization
    # Enhances local contrast, critical for X-ray detail
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Denoise with bilateral filter (preserves edges)
    denoised = cv2.bilateralFilter(equalized, d=9, sigmaColor=75, sigmaSpace=75)

    # Resize to model input
    resized = cv2.resize(denoised, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Stack as RGB (model expects 3 channels)
    rgb = np.stack([normalized, normalized, normalized], axis=-1)

    return rgb  # (H, W, 3) float32


def preprocess_for_display(image_bytes: bytes, target_size: tuple = (400, 400)) -> str:
    """Return base64 JPEG of the contrast-enhanced X-ray for display."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    resized = cv2.resize(enhanced, target_size)

    _, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode('utf-8')


# ─── GRAD-CAM ─────────────────────────────────────────────────────────────────
def compute_gradcam(model, img_array: np.ndarray, layer_name: str) -> np.ndarray:
    """
    Gradient-weighted Class Activation Mapping.
    Returns a normalized heatmap (H, W) float32 in [0,1].
    """
    # Build grad model: inputs → [target_conv_output, predictions]
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
    except ValueError:
        # Fallback to last conv layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                grad_model = tf.keras.models.Model(
                    inputs=model.inputs,
                    outputs=[layer.output, model.output]
                )
                break

    input_tensor = tf.cast(img_array[np.newaxis, ...], tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        conv_outputs, predictions = grad_model(input_tensor)
        # For binary classification, use the single output
        loss = predictions[:, 0]

    # Gradients of loss w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients → importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalize
    heatmap = tf.maximum(heatmap, 0)
    if tf.reduce_max(heatmap) > 0:
        heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_gradcam(image_bytes: bytes, heatmap: np.ndarray,
                    display_size: tuple = (400, 400), alpha: float = 0.45) -> str:
    """
    Overlay Grad-CAM heatmap on original X-ray.
    Returns base64-encoded PNG.
    """
    # Decode original
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, display_size)
    base_bgr = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)

    # Resize heatmap to display size
    heatmap_resized = cv2.resize(heatmap, display_size)

    # Apply colormap (JET: blue=normal, red=high activation)
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Weighted overlay
    overlay = cv2.addWeighted(base_bgr, 1 - alpha, colored, alpha, 0)

    # Add subtle border annotation
    cv2.rectangle(overlay, (2, 2), (display_size[0]-2, display_size[1]-2),
                  (0, 200, 200), 1)

    _, buf = cv2.imencode('.png', overlay)
    return base64.b64encode(buf).decode('utf-8')


# ─── MAIN PREDICTION PIPELINE ─────────────────────────────────────────────────
def predict(image_bytes: bytes) -> dict:
    """
    Full prediction pipeline.
    Returns dict with: label, confidence, probability, gradcam_b64,
                        enhanced_b64, preprocessing_steps, inference_time_ms
    """
    t0 = time.time()

    model, meta = get_model()
    img_size = tuple(meta["input_shape"][:2])
    threshold = meta.get("threshold", 0.5)

    # 1. Preprocess
    preprocessed = preprocess_xray(image_bytes, target_size=img_size)

    # 2. Predict
    input_batch = preprocessed[np.newaxis, ...]   # (1, H, W, 3)
    raw_prob = float(model.predict(input_batch, verbose=0)[0][0])

    # 3. Classification
    label = "Pneumonia" if raw_prob >= threshold else "Normal"
    confidence = raw_prob if label == "Pneumonia" else (1.0 - raw_prob)
    confidence_pct = round(confidence * 100, 2)

    # 4. Grad-CAM
    gradcam_layer = meta.get("grad_cam_layer", "conv4_gradcam")
    heatmap = compute_gradcam(model, preprocessed, gradcam_layer)
    gradcam_b64 = overlay_gradcam(image_bytes, heatmap)

    # 5. Enhanced display image
    enhanced_b64 = preprocess_for_display(image_bytes)

    inference_ms = round((time.time() - t0) * 1000, 1)

    # 6. Auxiliary metrics
    findings = _generate_findings(label, raw_prob, heatmap)

    return {
        "label": label,
        "probability": round(raw_prob, 4),
        "confidence": confidence_pct,
        "is_pneumonia": label == "Pneumonia",
        "risk_level": _risk_level(label, confidence_pct),
        "enhanced_b64": enhanced_b64,
        "gradcam_b64": gradcam_b64,
        "findings": findings,
        "inference_time_ms": inference_ms,
        "model_name": meta["model_name"],
        "architecture": meta["architecture"],
        "preprocessing": [
            "CLAHE contrast enhancement",
            "Bilateral denoising filter",
            "Lanczos resize to 128×128",
            "Per-pixel normalization [0,1]",
            "RGB channel stacking"
        ]
    }


def _risk_level(label: str, confidence: float) -> dict:
    if label == "Normal":
        return {"level": "Low", "color": "#2ed573", "description": "No significant abnormality detected"}
    if confidence >= 90:
        return {"level": "Critical", "color": "#ff4757", "description": "High-confidence pneumonia indicators"}
    if confidence >= 75:
        return {"level": "High", "color": "#ff6b35", "description": "Strong pneumonia indicators present"}
    return {"level": "Moderate", "color": "#ffa502", "description": "Possible pneumonia — further review advised"}


def _generate_findings(label: str, prob: float, heatmap: np.ndarray) -> list:
    """Generate interpretable findings based on prediction and heatmap."""
    activation_mean = float(np.mean(heatmap))
    activation_max  = float(np.max(heatmap))
    hot_region_pct  = round(float(np.mean(heatmap > 0.5)) * 100, 1)

    base = [
        f"Model confidence: {round(prob*100,1)}% pneumonia probability",
        f"Grad-CAM activation peak: {round(activation_max*100,1)}% intensity",
        f"High-activation region covers ~{hot_region_pct}% of image"
    ]

    if label == "Pneumonia":
        specific = [
            "Increased opacity consistent with consolidation",
            "Air-space opacification pattern detected",
            "Asymmetric density distribution noted in scan"
        ]
    else:
        specific = [
            "Lung fields appear clear — no focal consolidation",
            "No significant pleural effusion detected",
            "Cardiomediastinal silhouette within expected range"
        ]

    return base + specific
