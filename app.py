# oil_spill_app.py
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# -------------------
# CONFIG
# -------------------
IMG_HEIGHT = 256
IMG_WIDTH = 256
MODEL_PATH = "Unet_OilSpill.keras"  # path to your trained model

# Load trained model
@st.cache_resource
def load_unet_model():
    model = load_model(MODEL_PATH, compile=False)
    return model

model = load_unet_model()

# -------------------
# HELPER FUNCTIONS
# -------------------
def preprocess_image(uploaded_file):
    """Read and preprocess uploaded image."""
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image) / 255.0  # normalize
    return img_array, np.array(image)

def predict_mask(image_array):
    """Run prediction on image."""
    input_img = np.expand_dims(image_array, axis=0)  # add batch dimension
    pred_mask = model.predict(input_img)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # threshold
    return pred_mask

def overlay_mask(original_img, mask):
    """Overlay segmentation mask on original image."""
    overlay = original_img.copy()
    mask_colored = np.zeros_like(original_img)
    mask_colored[:, :, 0] = mask[:, :, 0] * 255  # Red channel
    overlay = cv2.addWeighted(original_img, 0.7, mask_colored, 0.3, 0)
    return overlay

# -------------------
# STREAMLIT UI
# -------------------
st.set_page_config(page_title="AI SpillGuard - Oil Spill Detection", layout="wide")
st.title("🌊 AI SpillGuard: Oil Spill Detection System")
st.write("Upload a satellite image and detect oil spill regions using deep learning (U-Net).")

uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    # Preprocess
    img_array, orig_img = preprocess_image(uploaded_file)
    st.image(orig_img, caption="Uploaded Image", use_column_width=True)

    # Predict
    with st.spinner("Detecting oil spills..."):
        mask = predict_mask(img_array)
        overlay_img = overlay_mask(orig_img, mask)

    # Show Results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(orig_img, caption="Original Image")
    with col2:
        st.image(mask * 255, caption="Predicted Mask", clamp=True)
    with col3:
        st.image(overlay_img, caption="Overlay Result")

    # Optional: Save output
    if st.button("Download Result"):
        result = Image.fromarray(overlay_img)
        result.save("oilspill_result.png")
        st.success("Result saved as oilspill_result.png")
