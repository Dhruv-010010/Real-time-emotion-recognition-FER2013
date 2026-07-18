"""
Real-Time Emotion Detection using OpenCV + Trained Keras Model
Usage:
    python realtime_emotion_detection.py --model "path/to/your/model.keras"

Controls:
    Q / ESC  → Quit
"""

import argparse
import glob
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf


# ── Emotion labels (must match training order) ──────────────────────────────
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Keep the bundled model portable when the repository is cloned elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Models" / "final_trained_model.keras"

# ── Colour for each emotion (BGR format for OpenCV) ─────────────────────────
EMOTION_COLORS = {
    "angry":    (0, 0, 255),       # red
    "disgust":  (0, 140, 255),     # dark orange
    "fear":     (0, 165, 255),     # orange
    "happy":    (0, 255, 0),       # green
    "neutral":  (255, 200, 0),     # cyan-ish
    "sad":      (255, 0, 0),       # blue
    "surprise": (255, 0, 255),     # magenta
}

# ── Emoji / text symbols for each emotion ───────────────────────────────────
EMOTION_EMOJI = {
    "angry":    "[Angry]",
    "disgust":  "[Disgust]",
    "fear":     "[Fear]",
    "happy":    "[Happy]",
    "neutral":  "[Neutral]",
    "sad":      "[Sad]",
    "surprise": "[Surprise]",
}


def load_model(model_path: str):
    """Load the trained Keras model from disk."""
    print(f"[INFO] Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("[INFO] Model loaded successfully!")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)


def load_face_detector():
    """Load OpenCV's Haar Cascade face detector."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("[ERROR] Could not load Haar Cascade face detector.")
        sys.exit(1)
    print("[INFO] Face detector loaded.")
    return face_cascade


def preprocess_face(face_roi):
    """
    Preprocess a detected face ROI for model prediction.
    - Convert to grayscale (if needed)
    - Resize to 48×48
    - Reshape to (1, 48, 48, 1) batch
    - The model's built-in Rescaling layer handles normalisation
    """
    # Convert to grayscale if the face ROI is in colour
    if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_roi

    # Resize to model input size
    face_resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)

    # Reshape to (1, 48, 48, 1) and cast to float32
    face_input = face_resized.astype(np.float32)
    face_input = np.expand_dims(face_input, axis=-1)   # (48, 48, 1)
    face_input = np.expand_dims(face_input, axis=0)     # (1, 48, 48, 1)

    return face_input


def draw_results(frame, x, y, w, h, emotion, confidence, all_probs):
    """Draw bounding box, emotion label, confidence bar on the frame."""
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    emoji = EMOTION_EMOJI.get(emotion, "")

    # ── Bounding box ────────────────────────────────────────────────────
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # ── Label background ────────────────────────────────────────────────
    label = f"{emoji} {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Label box above the face
    label_y = max(y - 10, text_h + 10)
    cv2.rectangle(frame,
                  (x, label_y - text_h - 8),
                  (x + text_w + 8, label_y + 4),
                  color, cv2.FILLED)
    cv2.putText(frame, label, (x + 4, label_y - 2),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # ── Mini confidence bars (right side of face box) ───────────────────
    bar_x_start = x + w + 8
    bar_width = 100
    bar_height = 12
    bar_gap = 3

    for i, (em_label, prob) in enumerate(zip(EMOTION_LABELS, all_probs)):
        bar_y = y + i * (bar_height + bar_gap)
        em_color = EMOTION_COLORS.get(em_label, (200, 200, 200))

        # Background bar
        cv2.rectangle(frame,
                      (bar_x_start, bar_y),
                      (bar_x_start + bar_width, bar_y + bar_height),
                      (50, 50, 50), cv2.FILLED)

        # Filled portion
        fill_w = int(bar_width * prob)
        cv2.rectangle(frame,
                      (bar_x_start, bar_y),
                      (bar_x_start + fill_w, bar_y + bar_height),
                      em_color, cv2.FILLED)

        # Label
        short_label = em_label[:3].upper()
        cv2.putText(frame, f"{short_label} {prob:.0%}",
                    (bar_x_start + bar_width + 5, bar_y + bar_height - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def scan_cameras(max_index: int = 5):
    """Scan common webcam device indices and print available cameras."""
    print("[INFO] Scanning camera indices...")
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        print(f"[INFO] Camera index {idx}: {'available' if ok else 'not available'}")
        if ok:
            available.append(idx)
            cap.release()
    if not available:
        print("[INFO] No cameras were found. Try checking camera permissions or closing other apps that use the camera.")
    return available


def open_camera(camera_index: int):
    """Attempt to open a webcam using DirectShow on Windows."""
    print(f"[INFO] Trying camera index {camera_index}")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[WARN] Camera index {camera_index} not available.")
        return None
    return cap


def run_realtime_detection(model, face_cascade, camera_index=0):
    """Main loop: capture frames → detect faces → predict emotions → display."""

    cap = open_camera(camera_index)
    if cap is None:
        # Try a few common alternate camera indices.
        for idx in range(1, 4):
            cap = open_camera(idx)
            if cap is not None:
                camera_index = idx
                break

    if cap is None:
        print("[ERROR] Cannot open webcam. Try a different --camera index, close other apps using the camera, or check Windows camera permissions.")
        sys.exit(1)

    print("[INFO] Webcam opened. Press 'Q' or ESC to quit.")

    # Try to set a reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame. Exiting.")
            break

        # Flip horizontally for mirror view
        frame = cv2.flip(frame, 1)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            # Extract face ROI from the colour frame
            face_roi = frame[y:y+h, x:x+w]

            # Preprocess
            face_input = preprocess_face(face_roi)

            # Predict
            predictions = model.predict(face_input, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = predictions[emotion_idx]

            # Draw results
            draw_results(frame, x, y, w, h, emotion, confidence, predictions)

        # ── HUD info ────────────────────────────────────────────────────
        faces_text = f"Faces: {len(faces)}"
        cv2.putText(frame, faces_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "Press Q / ESC to quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # Show
        cv2.imshow("Real-Time Emotion Detection", frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam released. Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Emotion Detection with OpenCV + Keras"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained Keras model file (.keras or .h5)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default: 0)"
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Scan common webcam indices and exit without running detection"
    )

    args = parser.parse_args()

    if args.probe:
        scan_cameras()
        return

    # Load model and face detector
    model = load_model(args.model)
    face_cascade = load_face_detector()

    # Run
    run_realtime_detection(model, face_cascade, camera_index=args.camera)


if __name__ == "__main__":
    main()


    
