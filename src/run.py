import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np

from Inference import (
    DEFAULT_MODEL_PATH,
    load_model,
    load_face_detector,
    preprocess_face,
    draw_results,
    EMOTION_LABELS,
)

st.title("Real-Time Facial Emotion Recognition using CNN model")

# Load once at module level — not inside recv(), not inside __init__
model = load_model(str(DEFAULT_MODEL_PATH))
face_cascade = load_face_detector()


class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            # Color ROI in — preprocess_face() does its own grayscale conversion
            face_roi = img[y:y + h, x:x + w]

            face_input = preprocess_face(face_roi)

            predictions = model.predict(face_input, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = predictions[emotion_idx]

            draw_results(img, x, y, w, h, emotion, confidence, predictions)

        return frame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
)
