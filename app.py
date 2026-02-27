from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
import firebase_admin
from firebase_admin import credentials, db
import time

app = Flask(__name__)
CORS(app)

# ------------------------------
# Firebase Setup
# ------------------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://raspberrypi1-652f4-default-rtdb.firebaseio.com/'
})

# ------------------------------
# Load TFLite Model
# ------------------------------
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]
input_dtype = input_details[0]['dtype']

# ------------------------------
# Load Labels
# ------------------------------
with open("labels.json", "r") as f:
    label_map = json.load(f)

index_to_label = {v: k for k, v in label_map.items()}

# ------------------------------
# Image Preprocessing
# ------------------------------
def preprocess_image(image):
    image = image.resize((input_width, input_height))
    image = np.array(image)

    if input_dtype == np.float32:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.uint8)

    image = np.expand_dims(image, axis=0)
    return image

# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def home():
    return "AgriSight Backend Running"

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_data = preprocess_image(image)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = int(np.argmax(output_data))
        confidence = float(np.max(output_data) * 100)

        disease_name = index_to_label[predicted_index]

        # ------------------------------
        # SAVE TO FIREBASE
        # ------------------------------
        ref = db.reference("detections")
        new_id = int(time.time())

        ref.child(new_id).set({
            "disease": disease_name,
            "confidence": round(confidence, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_url": "",
            "action_status": "Not Treated"
        })

        return jsonify({
            "disease": disease_name,
            "confidence": round(confidence, 2),
            "saved": True
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
