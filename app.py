from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import librosa
import os
import requests
from utils.feature_extractor import extract_features
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Download model from GitHub if not present
MODEL_URL = "https://github.com/codecomet07/latest.github.io/audio_classification_model.h5"
MODEL_PATH = "audio_classification_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['autotuned', 'deepfake', 'real'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    file = request.files['audio']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    features = extract_features(file_path)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        class_index = np.argmax(prediction)
        result = label_encoder.inverse_transform([class_index])[0]
        return jsonify({'prediction': result})

    return jsonify({'error': 'Could not process the audio'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use Render's default port
    app.run(debug=True, host='0.0.0.0', port=port)

