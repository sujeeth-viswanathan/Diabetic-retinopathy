import tensorflow as tf
import cv2
import numpy as np

MODEL_PATH = "CapsNet.Model"

model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

def predict_class(path):
    img = cv2.imread(path)
    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    RGBImg = cv2.resize(RGBImg, (224, 224))
    image = np.array(RGBImg) / 255.0

    prediction = infer(tf.constant([image], dtype=tf.float32))
    probabilities = prediction['dense_1'].numpy()[0].tolist()
    predicted_index = int(np.argmax(probabilities))

    diagnosis = "No Diabetic Retinopathy Detected" if predicted_index == 1 else "Diabetic Retinopathy Detected"
    return diagnosis, probabilities