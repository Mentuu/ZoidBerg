from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
from PIL import Image
import cv2
import joblib

# Load the trained model
pipeline = joblib.load('trained_model.pkl')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def crop_lungs_from_image(img):
    img_cv = np.array(img)
    _, thresh = cv2.threshold(img_cv, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img_cv = img_cv[y:y + h, x:x + w]
    cropped_img = Image.fromarray(cropped_img_cv)
    return cropped_img

def preprocess_image(image_path):
    image_size = (64, 64)
    img = Image.open(image_path).convert('L')
    img = crop_lungs_from_image(img)
    img = img.resize(image_size)
    img_array = np.array(img).flatten()
    return img_array.reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image_array = preprocess_image(file_path)
            prediction = pipeline.predict(image_array)
            label = 'Pneumonia' if prediction[0] == 1 else 'Normal'
            return render_template('result.html', label=label)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
