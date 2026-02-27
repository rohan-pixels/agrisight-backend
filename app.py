from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import json
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# ===============================
# Load TFLite Model
# ===============================
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

# ===============================
# Load Labels
# ===============================
with open("labels.json", "r") as f:
    label_map = json.load(f)

index_to_label = {v: k for k, v in label_map.items()}

# ===============================
# Preprocess Function
# ===============================
def preprocess_image(image):
    image = image.resize((input_width, input_height))
    image = np.array(image)

    # Normalize if model expects float32
    if input_details[0]['dtype'] == np.float32:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.uint8)

    image = np.expand_dims(image, axis=0)
    return image

# ===============================
# Prediction Endpoint
# ===============================
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

        return jsonify({
            "disease": disease_name,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "AgriSight TFLite API Running"


if __name__ == "__main__":
    app.run()
