from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from src.predict_utils import load_model, predict_face, CLASS_NAMES

ROOT = Path(__file__).resolve().parent
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

st.set_page_config(page_title="FaceSense AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.main {background: #f6f8fc;}
.block-container {padding-top: 2rem;}
.hero {padding: 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#172554,#2563eb); color:white;}
.small-muted {color:#64748b;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>FaceSense AI</h1>
<p>Facial Expression Recognition | CNN + OpenCV</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model()

try:
    model = get_model()
except Exception as error:
    st.error(str(error))
    st.stop()

st.write("")
tab1, tab2, tab3 = st.tabs(["Image Analysis", "Camera Snapshot", "About"])

def analyze_image(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    output = rgb.copy()
    results = []

    for (x, y, w, h) in faces:
        label, confidence, probabilities = predict_face(model, gray[y:y+h, x:x+w])
        results.append((label, confidence, probabilities))
        cv2.rectangle(output, (x, y), (x+w, y+h), (20, 190, 110), 2)
        cv2.putText(output, f"{label} {confidence*100:.1f}%",
                    (x, max(y-8, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (20, 190, 110), 2)
    return output, results

def show_results(image):
    annotated, results = analyze_image(image)
    left, right = st.columns([1.2, 1])
    with left:
        st.image(annotated, caption="Detected faces and predicted expressions", use_container_width=True)
    with right:
        if not results:
            st.warning("No face detected. Try a clear, front-facing image with good lighting.")
        else:
            st.metric("Faces detected", len(results))
            for i, (label, confidence, probs) in enumerate(results, 1):
                st.subheader(f"Face {i}: {label}")
                st.progress(float(confidence), text=f"Model score: {confidence*100:.1f}%")
                st.caption("Class scores are model outputs, not calibrated probabilities of a person's true feelings.")
                st.bar_chart({name: float(value) for name, value in zip(CLASS_NAMES, probs)})

with tab1:
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded:
        show_results(Image.open(uploaded))

with tab2:
    st.info("Take a still image using your device camera. For continuous live webcam detection, run webcam.py.")
    camera_image = st.camera_input("Take a camera snapshot")
    if camera_image:
        show_results(Image.open(camera_image))

with tab3:
    st.subheader("Project overview")
    st.write("""
    FaceSense AI uses a CNN trained on FER-2013 to classify visible facial expressions
    into seven categories. OpenCV detects face regions before classification.
    """)
    st.warning("A facial-expression classifier cannot reliably determine a person's internal emotional state, intent, or mental health.")
    st.write("Model: CNN | Dataset: FER-2013 | Input: 48 × 48 grayscale face images")
