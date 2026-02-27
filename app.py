from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
import time
import firebase_admin
from firebase_admin import credentials, db

# --------------------------------------------------
# Initialize Flask
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Initialize Firebase (Railway Safe)
# --------------------------------------------------
firebase_json = json.loads(os.environ["FIREBASE_CREDENTIALS"])
cred = credentials.Certificate(firebase_json)

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://raspberrypi1-652f4-default-rtdb.firebaseio.com/'
})

db_ref = db.reference("detections")

# --------------------------------------------------
# Load TFLite Model
# --------------------------------------------------
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_H = input_details[0]['shape'][1]
IMG_W = input_details[0]['shape'][2]
INPUT_DTYPE = input_details[0]['dtype']

# --------------------------------------------------
# Load Labels
# --------------------------------------------------
with open("labels.json", "r") as f:
    label_map = json.load(f)

index_to_label = {v: k for k, v in label_map.items()}

# --------------------------------------------------
# Preprocess Image
# --------------------------------------------------
def preprocess_image(image):
    image = image.resize((IMG_W, IMG_H))
    image = np.array(image)

    if INPUT_DTYPE == np.float32:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.uint8)

    image = np.expand_dims(image, axis=0)
    return image

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def home():
    return "AgriSight Backend Running"

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # Read Image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_data = preprocess_image(image)

        # Run Model
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_index = int(np.argmax(output))
        disease = index_to_label[pred_index]
        confidence = round(float(output[pred_index]) * 100, 2)

        # Create unique ID
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        detection_id = str(int(time.time() * 1000))

        # Save to Firebase Realtime DB
        db_ref.child(detection_id).set({
            "disease": disease,
            "confidence": confidence,
            "timestamp": timestamp,
            "image_url": "",  # (Pi already uploads image to storage)
            "action_status": "Not Treated"
        })

        return jsonify({
            "disease": disease,
            "confidence": confidence,
            "saved": True
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# Run App
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
