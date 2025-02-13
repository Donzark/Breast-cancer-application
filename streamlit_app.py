import os
os.system("pip install tensorflow-hub")
import sys
sys.path.append("/home/appuser/.local/lib/python3.12/site-packages")

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import tf_keras

# Streamlit Layout
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to preprocess and predict
@st.cache_resource
def load_prediction_model():
    """Loads the trained model and caches it."""
    MODEL_PATH = "./model/resnet_breast_cancer_model"
    model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')
    return model

model = load_prediction_model()

@st.cache_resource
def predict_image(image_path, model):
    """Processes image and makes predictions."""
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    preds = model(image)
    
    if isinstance(preds, dict):  
        key = list(preds.keys())[0]  
        preds = preds[key]  
    
    preds = preds.numpy() if hasattr(preds, "numpy") else np.array(preds)

    pred_class = "⚠️ Potential Malignant Tumor Detected. Immediate medical evaluation is advised!" if preds[0] > 0.5 else "✅ No Malignant Tumor Indicators Detected."
    pred_conf = preds[0]  

    return pred_class, pred_conf

# Sidebar
with st.sidebar:
    st.title("🎗️ Breast Cancer Detection App")
    st.markdown("""
    This application uses an advanced **ResNet-based CNN model** to analyze mammogram images and detect possible signs of **Breast Cancer**.
    
    ### 🚀 Features:
    - **Deep Learning Model:** ResNet-based CNN
    - **Dataset:** Custom Mammogram Dataset
    - **Accuracy:** High-performance cancer detection with **94% Accuracy**
    - **Fast & Secure:** AI-powered real-time diagnosis
    """)
    st.markdown("---")
    st.markdown("### 🙌 Acknowledgment")
    st.markdown("""
    This project is supported by **TetFund** through the **Institution-Based Research (IBR)**.  
    Special recognition to **Dr. Obasa, Adekunle Isiaka** for pioneering this research.
    """)

# Title 
st.title("🎗️ AI-Powered Breast Cancer Detection")
st.markdown("Upload a **mammogram image** to detect signs of breast cancer.")

# Session State to Handle File Persistence
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# File Uploader
uploaded_file = st.file_uploader(
    "📤 Upload a mammogram image (JPG, PNG, or JPEG)", 
    type=["jpg", "jpeg", "png"]
)

# Store in session state
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

# After Upload: Two-Column Layout
if st.session_state.uploaded_file:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(st.session_state.uploaded_file, caption="📷 Uploaded Image", width=300)

    with col2:
        pred_button = st.button("🔍 Analyze Image")

        if pred_button:
            with st.spinner("🔄 Analyzing the CT scan..."):
                with open("temp_image.jpg", "wb") as f:
                    f.write(st.session_state.uploaded_file.getbuffer())

                pred_class, pred_conf = predict_image("temp_image.jpg", model)

                adjusted_conf = 100 - (pred_conf[0] * 100) if "No Malignant Tumor" in pred_class else pred_conf[0] * 100

                st.session_state.prediction = (pred_class, adjusted_conf)

# Display Prediction (Persists after rerun)
if st.session_state.prediction:
    pred_class, adjusted_conf = st.session_state.prediction

    with col2:
        if "No Stroke" in pred_class:
            st.success(f"✅ **{pred_class}**\n\nConfidence: **{adjusted_conf:.2f}%**")
        else:
            st.error(f"⚠️ **{pred_class}**\n\nConfidence: **{adjusted_conf:.2f}%**")

# Warning Message (Only Show if No Image is Uploaded)
if not st.session_state.uploaded_file:
    st.warning("⚠️ Please upload a **mammogram image** to proceed.")
