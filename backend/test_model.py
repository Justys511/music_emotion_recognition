import numpy as np
import tensorflow as tf
import librosa
import joblib
import os
import threading
from pathlib import Path
# Paths

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "Emotion_Voice_Detection_Model_Emotify.h5"


model = None
model_lock = threading.Lock()


def get_model():
    global model
    if model is None:
        with model_lock:
            if model is None:
                model = tf.keras.models.load_model(
                    str(MODEL_PATH),
                    compile=False,
                )
    return model

# List of emotions that the model is trained to predict
emotion_labels = [
    "amazement", "solemnity", "tenderness", "nostalgia", "calmness",
    "power", "joyful_activation", "tension", "sadness"
]

###################################################
# FEATURE EXTRACTION FUNCTIONS (as in extract_features.py)
###################################################
def compute_features(audio, sr, max_pad_length=100):
    """
    Extracts and concatenates several types of features:
      - Mel-spectrogram (64 bands)
      - MFCC (40)
      - Chroma (12)
      - Spectral Contrast (7)
    Total of 64 + 40 + 12 + 7 = 123 rows.
    Then we pad/trim in time to max_pad_length and do Z-score normalization per component.
    Returns shape (123, max_pad_length).
    """
    # 1) Mel
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # shape: (64, T)

    # 2) MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # (40, T)

    # 3) Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)    # (12, T)

    # 4) Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)  # (7, T)

    # Concatenate everything along axis 0
    feat_array = np.concatenate([mel_db, mfcc, chroma, spec_contrast], axis=0)  
    # now shape -> (123, T)

    # Pad/trim along the time dimension
    time_frames = feat_array.shape[1]
    if time_frames < max_pad_length:
        pad_width = max_pad_length - time_frames
        feat_array = np.pad(feat_array, ((0, 0), (0, pad_width)), mode='constant')
    else:
        feat_array = feat_array[:, :max_pad_length]

    # Z-score normalization (per component, across each of the 123 rows)
    for i in range(feat_array.shape[0]):
        row = feat_array[i, :]
        mean, std = row.mean(), row.std()
        if std < 1e-6:
            feat_array[i, :] = 0.
        else:
            feat_array[i, :] = (row - mean) / (std + 1e-6)

    return feat_array  # shape: (123, max_pad_length)

def extract_features_for_inference(file_path, segment_length=60.0, max_pad_length=100):
    """
    1) Loads the audio
    2) Splits it into chunks of length `segment_length`
    3) Extracts (Mel, MFCC, Chroma, Contrast) for each chunk
    4) Pads/normalizes => shape (123, max_pad_length)
    5) Adds the channel dimension => (123, max_pad_length, 1)
    6) Combines all chunks into one numpy array => (num_segments, 123, max_pad_length, 1)
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        total_duration = librosa.get_duration(y=audio, sr=sr)

        features_list = []
        # Iterate over segments
        for start in np.arange(0, total_duration, segment_length):
            end = min(start + segment_length, total_duration)
            segment = audio[int(start*sr):int(end*sr)]

            # Extract multi-feature features
            feat_array = compute_features(segment, sr, max_pad_length=max_pad_length)
            # Add the channel dimension
            feat_array = np.expand_dims(feat_array, axis=-1)  # (123, max_pad_length, 1)
            features_list.append(feat_array)

        # Combine
        if len(features_list) == 0:
            return None

        return np.array(features_list)  # shape -> (num_segments, 123, max_pad_length, 1)

    except Exception as e:
        print(f"❌ Error while processing {file_path}: {e}")
        return None

def predict_emotion(file_path):
    features = extract_features_for_inference(file_path)
    if features is None or len(features) == 0:
        return {"error": "Could not extract features."}

    predictions = get_model().predict(features)
    avg_prediction = np.mean(predictions, axis=0)

    if len(avg_prediction) != len(emotion_labels):
        return {"error": "Model output size does not match."}

    best_prob = np.max(avg_prediction)
    EPS = 0.02
    top_emotions = {
        emotion: float(prob)
        for emotion, prob in zip(emotion_labels, avg_prediction)
        if prob >= best_prob - EPS
    }

    return {
        "emotion": max(top_emotions, key=top_emotions.get),
        "probabilities": {k: float(f"{v:.4f}") for k, v in zip(emotion_labels, avg_prediction)}
    }
