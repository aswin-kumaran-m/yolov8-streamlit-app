import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import time
import tempfile
import gdown

st.set_page_config(page_title="YOLOv8 Image Detector", layout="centered")
st.title("🧠 YOLOv8 Object Detection App")
st.markdown("Upload a YOLOv8 model and an image to detect objects!")

# ---------- OPTIONAL MODEL UPLOAD ----------
st.markdown("🔁 **Optional:** Upload your own YOLOv8 model, or skip to use the default model.")

model_file = st.file_uploader("📦 Upload YOLOv8 model (.pt)", type=["pt"])
model = None

if model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        model_path = tmp.name
        model = YOLO(model_path)
        st.success("✅ Custom model loaded successfully!")
else:
    # Download default model from Google Drive if not uploaded
    default_model_id = "1nRvORtYtEbtvQXIvzyEdwsvHsTcZ18-_"  # Replace with your own ID if needed
    default_model_url = f"https://drive.google.com/uc?id={default_model_id}"
    default_model_path = os.path.join(tempfile.gettempdir(), "best.pt")

    if not os.path.exists(default_model_path):
        st.info("No model uploaded. Downloading default model...")
        gdown.download(default_model_url, default_model_path, quiet=False)

    model = YOLO(default_model_path)
    st.success("✅ Default model loaded successfully!")

# ---------- IMAGE UPLOAD ----------
uploaded_image = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image and model:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    img = Image.open(uploaded_image)
    image_path = os.path.join(tempfile.gettempdir(), "input.jpg")
    img.save(image_path)

    if st.button("🚀 Detect"):
        with st.spinner("Detecting objects..."):
            results = model(image_path)

            output_path = os.path.join(tempfile.gettempdir(), "output.jpg")
            results[0].save(filename=output_path)

            time.sleep(1)
            st.success("✅ Detection complete!")
            st.image(output_path, caption="Detected Objects", use_column_width=True)

            # Bounding box info
            boxes = results[0].boxes
            if boxes is not None:
                st.subheader("📦 Bounding Box Details")
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    st.write(f"**Object {i+1}:** Class: {model.names[cls]}, Confidence: {conf:.2f}, Box: {xyxy}")

            # Download button
            with open(output_path, "rb") as file:
                st.download_button(
                    label="📥 Download Output Image",
                    data=file,
                    file_name="output.jpg",
                    mime="image/jpeg"
                )
