
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Correct trained model path
MODEL_PATH = ROOT / "models" / "emotion_model.keras"

# Emotion class names
CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    print("Loading trained emotion recognition model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully!")

    return model


def prepare_face(face_gray):
    # Resize face image to 48x48
    face = cv2.resize(
        face_gray,
        (48, 48),
        interpolation=cv2.INTER_AREA
    )

    # Normalize pixel values
    face = face.astype("float32") / 255.0

    # Reshape image for CNN model
    face = np.expand_dims(face, axis=(0, -1))

    return face


def predict_face(model, face_gray):
    # Prepare face image
    processed_face = prepare_face(face_gray)

    # Predict facial expression
    probabilities = model.predict(
        processed_face,
        verbose=0
    )[0]

    # Get highest probability class
    index = int(np.argmax(probabilities))

    label = CLASS_NAMES[index]
    confidence = float(probabilities[index])

    return label, confidence, probabilities
