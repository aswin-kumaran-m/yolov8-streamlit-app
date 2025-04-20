YOLOv8 Flask App
================

This is a simple Flask web app that lets you upload an image and uses a YOLOv8 model to detect objects.

✔ Model: best.pt (already included)
✔ Upload form: at http://127.0.0.1:5000

Instructions:
-------------
1. Open PowerShell and navigate to the folder:
   cd YOUR_EXTRACTED_FOLDER_PATH

2. Run the app:
   python inference.py

3. Open your browser:
   http://127.0.0.1:5000

4. Upload an image and click Detect.

Dependencies:
-------------
- Flask
- opencv-python
- ultralytics (YOLOv8)

Install them with:
   pip install flask opencv-python ultralytics