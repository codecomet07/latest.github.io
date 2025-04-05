from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os
import requests
from utils.feature_extractor import extract_features
from sklearn.preprocessing import LabelEncoder
from supabase import create_client
import uuid
import traceback

app = Flask(__name__)
CORS(app)

# Supabase config
SUPABASE_URL = "https://wfelzbdrtfyapzrmuuis.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndmZWx6YmRydGZ5YXB6cm11dWlzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM4NzQ5NDQsImV4cCI6MjA1OTQ1MDk0NH0.78pcejY1J0Lww8lX-fIhIYUt2nNvhxZJW5Oa0J1Ek3E"
SUPABASE_BUCKET = "audiolyt"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Download model from GitHub if not present
MODEL_URL = "https://github.com/codecomet07/latest.github.io/blob/main/audio_classification_model.h5"
MODEL_PATH = "Audiolyt/Backend/audio_classification_model.h5"



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

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Get original extension
    original_filename = file.filename
    ext = os.path.splitext(original_filename)[1]  # e.g., '.mp3', '.flac'
    if not ext:
        return jsonify({'error': 'File must have an extension'}), 400

    # Generate unique filename with original extension
    unique_filename = f"{uuid.uuid4()}{ext}"

    try:
        file_bytes = file.read()  # Read the uploaded file as bytes
        res = supabase.storage.from_(SUPABASE_BUCKET).upload(unique_filename, file_bytes)

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{unique_filename}"
        return jsonify({'message': 'Uploaded', 'url': public_url}), 200
    except Exception as e:
        # Print error in server logs
        print("UPLOAD ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    audio_url = data.get('audio_url')

    if not audio_url:
        return jsonify({'error': 'No audio URL provided'}), 400

    try:
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)

        # Get file extension from URL
        ext = os.path.splitext(audio_url)[1]
        local_filename = f"temp_audio{ext}"
        local_path = os.path.join(upload_folder, local_filename)

        # Download the audio from Supabase public URL
        response = requests.get(audio_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download audio file from Supabase'}), 500

        with open(local_path, 'wb') as f:
            f.write(response.content)

        # Extract features and predict
        features = extract_features(local_path)
        if features is not None:
            features = features.reshape(1, -1)
            prediction = model.predict(features)
            class_index = np.argmax(prediction)
            result = label_encoder.inverse_transform([class_index])[0]
            return jsonify({'prediction': result})

        return jsonify({'error': 'Could not process the audio'}), 500

    except Exception as e:
        print("PREDICT ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=10000)
