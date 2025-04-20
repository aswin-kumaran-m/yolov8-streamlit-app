from flask import Flask, request, render_template, url_for
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

model = YOLO('best.pt')  # Ensure best.pt is in the same directory

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image = request.files['image']
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')

    image.save(input_path)
    img = cv2.imread(input_path)

    results = model(img)
    results[0].save(filename=output_path)

    return render_template('index.html', result=True)

if __name__ == '__main__':
    app.run(port=5000)