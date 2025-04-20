import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import time

# Load YOLOv8 model
model = YOLO("C:/Users/Aswin/Desktop/YOLO_Project/yolov8_flask_final_project/best.pt")


st.set_page_config(page_title="YOLOv8 Image Detector", layout="centered")
st.title("🧠 YOLOv8 Object Detection App")
st.markdown("Upload an image and click **Detect** to see object detection in action.")

uploaded_image = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Show uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Save the image locally
    img = Image.open(uploaded_image)
    image_path = "input.jpg"
    img.save(image_path)

    # Add a Detect button
    if st.button("🚀 Detect"):
        with st.spinner("Detecting objects..."):
            results = model(image_path)

            # Save output image
            output_path = "output.jpg"
            results[0].save(filename=output_path)

            time.sleep(1)
            st.success("Detection complete!")

            # Show result
            st.image(output_path, caption="Detected Objects", use_column_width=True)

            # Show bounding boxes and confidences
            boxes = results[0].boxes
            if boxes is not None:
                st.subheader("Bounding Box Details")
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    st.write(f"**Object {i+1}:** Class: {model.names[cls]}, Confidence: {conf:.2f}, Box: {xyxy}")

            # Download button
            with open(output_path, "rb") as file:
                btn = st.download_button(
                    label="📥 Download Output Image",
                    data=file,
                    file_name="output.jpg",
                    mime="image/jpeg"
                )
