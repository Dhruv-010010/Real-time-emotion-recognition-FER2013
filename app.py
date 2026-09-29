"""
Flask API — Real-Time Facial Emotion Recognition
Endpoints:
    GET  /health         ? liveness check
    POST /predict        ? accepts image upload, returns emotion JSON
    POST /predict/base64 ? accepts {"image": "<base64 string>"}
"""

import os
import base64
import logging

import cv2
import numpy as np
from flask import Flask, request, jsonify
from realtime_emotion_detection import (
    load_model,
    load_face_detector,
    preprocess_face,
    EMOTION_LABELS,
)

# -- Logging ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# -- App init -----------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

# -- Load model & detector once at startup ------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "final_trained_model.keras")

log.info("Loading model...")
model = load_model(MODEL_PATH)
log.info("Loading face detector...")
face_cascade = load_face_detector()
log.info("Ready.")


# -- Helpers -------------------------------------------------------------------

def decode_image_bytes(image_bytes):
    """Decode raw image bytes into a BGR ndarray."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Send a valid JPEG/PNG.")
    return img


def run_inference(img):
    """Run face detection + emotion inference on a BGR image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    results = []
    for (x, y, w, h) in faces:
        face_roi = img[y:y + h, x:x + w]
        face_input = preprocess_face(face_roi)

        probs = model.predict(face_input, verbose=0)[0]
        emotion_idx = int(np.argmax(probs))
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = float(probs[emotion_idx])

        results.append({
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "bounding_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "all_scores": {
                label: round(float(prob), 4)
                for label, prob in zip(EMOTION_LABELS, probs)
            },
        })

    return results


# -- Routes --------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Liveness check for Render."""
    return jsonify({"status": "ok", "model": "CNN FER-2013"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a multipart/form-data image upload.
    Field name: image

    Returns:
        { "faces_detected": N, "results": [ {...}, ... ] }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in form-data"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = decode_image_bytes(file.read())
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    try:
        results = run_inference(img)
    except Exception as e:
        log.exception("Inference error")
        return jsonify({"error": f"Inference failed: {e}"}), 500

    return jsonify({"faces_detected": len(results), "results": results}), 200


@app.route("/predict/base64", methods=["POST"])
def predict_base64():
    """
    Accept a JSON body with a base64-encoded image.
    Body: { "image": "<base64 string>" }
    """
    body = request.get_json(silent=True)
    if not body or "image" not in body:
        return jsonify({"error": "JSON body must contain an image key with a base64 string"}), 400

    try:
        image_bytes = base64.b64decode(body["image"])
        img = decode_image_bytes(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not decode base64 image: {e}"}), 422

    try:
        results = run_inference(img)
    except Exception as e:
        log.exception("Inference error")
        return jsonify({"error": f"Inference failed: {e}"}), 500

    return jsonify({"faces_detected": len(results), "results": results}), 200


# -- Entry point (local dev only — Render uses gunicorn) ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

