import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Dataset path
DATASET_PATH = "./dataset/audio_speech_actors_01-24"  # Replace with the path to your dataset

# Emotions map based on RAVDESS convention
EMOTIONS_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprise"
}
EMOTIONS = list(EMOTIONS_MAP.values())

# Audio processing parameters
SAMPLE_RATE = 22050
MAX_PAD_LEN = 174

# Extract features from an audio file
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
        print(f"Error processing {file_path}: {e}")
        return None

# Load dataset and labels
def load_data():
    X, y = [], []
    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                # Extract emotion from filename (4th element in the split name)
                emotion_code = file.split("-")[2]
                emotion = EMOTIONS_MAP.get(emotion_code)
                if emotion:
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(EMOTIONS.index(emotion))
    return np.array(X), np.array(y)

# Load data
X, y = load_data()

# Reshape data for CNN
X = np.expand_dims(X, axis=-1)
y = to_categorical(y, num_classes=len(EMOTIONS))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(40, MAX_PAD_LEN, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(len(EMOTIONS), activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("ravdess_emotion_model.h5")
print("Model saved as ravdess_emotion_model.h5")
