"""
PneumoScan AI — Flask Web Application
Chest X-Ray Pneumonia Detection with Grad-CAM
"""
import os
import uuid
import json
import traceback
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ─── APP CONFIG ───────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
ALLOWED_EXT   = {"png", "jpg", "jpeg", "bmp", "webp", "tiff"}
MAX_MB        = 15

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
app.config["SECRET_KEY"] = "pneumoscan_secret_2024"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint — accepts image, returns prediction JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXT).upper()}"}), 400

    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"error": "Empty file uploaded"}), 400

        # Run inference (lazy import to keep startup fast)
        from inference import predict
        result = predict(image_bytes)

        return jsonify({"success": True, "result": result})

    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Analysis failed: " + str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "PneumoNet Lite", "version": "1.0.0"})


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  PneumoScan AI — Pneumonia Detection System")
    print("  http://localhost:5000")
    print("="*55)
    app.run(debug=True, host="0.0.0.0", port=5000)
