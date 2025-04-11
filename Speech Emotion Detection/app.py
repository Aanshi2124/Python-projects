from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app) 

# Load the trained model
MODEL_PATH = "ravdess_emotion_model.h5"  # Update with your model's file path
model = load_model(MODEL_PATH)

# Emotion labels (must match the order used during training)
EMOTIONS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprise"]

# Audio processing parameters
SAMPLE_RATE = 22050
MAX_PAD_LEN = 174

# Extract features from audio
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = MAX_PAD_LEN - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
        return mfccs
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

# API endpoint for emotion prediction
@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        # Check if an audio file is in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        # Extract features from the uploaded file
        features = extract_features(filepath)
        if features is None:
            return jsonify({"error": "Feature extraction failed"}), 500
        
        # Prepare features for the model
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=-1)

        # Predict emotion
        predictions = model.predict(features)
        emotion = EMOTIONS[np.argmax(predictions)]

        # Cleanup temporary file
        os.remove(filepath)

        return jsonify({"emotion": emotion}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
