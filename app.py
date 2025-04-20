import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import time
import tempfile

st.set_page_config(page_title="YOLOv8 Image Detector", layout="centered")
st.title("🧠 YOLOv8 Object Detection App")
st.markdown("Upload a YOLOv8 model and an image to detect objects!")

# Upload YOLOv8 model
model_file = st.file_uploader("📦 Upload YOLOv8 model (.pt)", type=["pt"])
model = None

if model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        model_path = tmp.name
        model = YOLO(model_path)
        st.success("✅ Model loaded successfully!")

# Upload Image
uploaded_image = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image and model:
    # Show uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Save the image temporarily
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

            boxes = results[0].boxes
            if boxes is not None:
                st.subheader("📦 Bounding Box Details")
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    st.write(f"**Object {i+1}:** Class: {model.names[cls]}, Confidence: {conf:.2f}, Box: {xyxy}")

            with open(output_path, "rb") as file:
                st.download_button(
                    label="📥 Download Output Image",
                    data=file,
                    file_name="output.jpg",
                    mime="image/jpeg"
                )
