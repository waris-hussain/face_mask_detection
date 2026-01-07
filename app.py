import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(page_title="Mask Detector", layout="centered")

st.title("😷 Face Mask Detection App")
st.write("Upload an image to check for face masks.")

# Load model
@st.cache_resource
def load_model():
    model = YOLO("best (1).pt")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect"):
        with st.spinner("Detecting..."):
            # Perform inference
            results = model(image)
            
            # Visualize results
            res_plotted = results[0].plot()
            res_image = Image.fromarray(res_plotted[..., ::-1]) # Convert BGR to RGB
            
            st.image(res_image, caption="Detected Image", use_column_width=True)
            
            # Optional: Show detection details
            # st.write(results[0].boxes)
