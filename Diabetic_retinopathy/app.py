import tensorflow as tf
from flask import Flask, render_template, request
import os
import cv2
from image_validation import validate_retina_image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No selected file')
    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img = cv2.imread(file_path)
        if not validate_retina_image(img):
            return render_template('predict.html', diagnosis=" Invalid Image", probabilities=[0, 0], user_image=None)
                            
        diagnosis, probabilities = predict_class(file_path)
        return render_template('predict.html', diagnosis=diagnosis, probabilities=probabilities, user_image=file_path)
    
    return render_template('index.html', message='Invalid file type')

if __name__ == '__main__':
    app.run(debug=True)
