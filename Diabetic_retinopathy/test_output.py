import tensorflow as tf

MODEL_PATH = r"C:/diabetic/diabetic/Diabetic_retinopathy/CapsNet.Model"

model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

print("Model Output Signature:")
print(infer.structured_outputs)
