# extract_features.py

import librosa
import numpy as np
import pandas as pd
import joblib
import os
import random
import time
from tqdm import tqdm
from joblib import Parallel, delayed

# Paths to the input CSV and output joblib files
DATASET_CSV = "data/processed_dataset.csv"
OUTPUT_FEATURES_X = "joblib_features/X_emotify.joblib"
OUTPUT_FEATURES_Y = "joblib_features/y_emotify.joblib"

# List of emotion columns (must match what's in the CSV)
emotion_columns = [
    "amazement", "solemnity", "tenderness", "nostalgia", "calmness",
    "power", "joyful_activation", "tension", "sadness"
]

# ------------------------------------------------------------------------------------
# FEATURE EXTRACTION FUNCTIONS
# ------------------------------------------------------------------------------------
def compute_features(audio, sr, max_pad_length=100):
    """
    Extracts several types of spectral features and stacks them vertically:
      1) Mel-spectrogram (log-mel)
      2) MFCC
      3) Chroma
      4) Spectral Contrast

    Final shape: (num_features_total, time_frames).
    Then we pad/trim to max_pad_length and return the shape (num_features_total, max_pad_length).

    Important: each block is normalized (Z-score) independently.
    """

    # 1) Mel-spectrogram (64 bands)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # shape: (64, T)

    # 2) MFCC (40)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # (40, T)

    # 3) Chroma (12)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)    # (12, T)

    # 4) Spectral Contrast (7)
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)  # (7, T)

    # Concatenate along axis 0
    feat_array = np.concatenate([mel_db, mfcc, chroma, spec_contrast], axis=0)
    # shape -> (64 + 40 + 12 + 7 = 123, T)

    time_frames = feat_array.shape[1]
    if time_frames < max_pad_length:
        pad_width = max_pad_length - time_frames
        feat_array = np.pad(feat_array,
                            pad_width=((0, 0), (0, pad_width)),
                            mode='constant')
    else:
        feat_array = feat_array[:, :max_pad_length]

    # Z-score normalization by component
    for i in range(feat_array.shape[0]):
        row = feat_array[i, :]
        mean, std = row.mean(), row.std()
        if std < 1e-6:
            feat_array[i, :] = 0.
        else:
            feat_array[i, :] = (row - mean) / (std + 1e-6)

    return feat_array

def load_and_augment(file_path, duration=60.0):
    """Loads audio + random augmentation (time/pitch/none)."""
    audio, sr = librosa.load(file_path, sr=None, duration=duration)

    aug_type = random.choice(['time', 'pitch', 'none'])
    if aug_type == 'time':
        rate = random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=rate)
    elif aug_type == 'pitch':
        n_steps = random.choice([-2, -1, 1, 2])
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    return audio, sr

# ------------------------------------------------------------------------------------
# MAIN FUNCTION FOR PROCESSING A SINGLE ROW
# ------------------------------------------------------------------------------------
def process_row(row, augment=False, max_pad_length=100):
    """
    1) Load audio (with augmentation if augment=True)
    2) Extract several features (Mel, MFCC, Chroma, Spectral Contrast)
    3) Return (features, active_emotions) or (None, None) if an error occurs
    """
    try:
        file_path = row["file_path"]
        if augment:
            audio, sr = load_and_augment(file_path, duration=60.0)
        else:
            audio, sr = librosa.load(file_path, sr=None, duration=60.0)

        feat_array = compute_features(audio, sr, max_pad_length=max_pad_length)
        active_emotions = [emo for emo in emotion_columns if row[emo] == 1]
        return feat_array, active_emotions

    except Exception as e:
        print(f"❌ Error while processing {row['file_path']}: {e}")
        return None, None

# ------------------------------------------------------------------------------------
# SPECIAL FUNCTION FOR OVERSAMPLING
# ------------------------------------------------------------------------------------
def oversample_task(row, repeats=2, max_pad_length=100):
    """
    Called in Parallel for a single row that contains "rare" emotions.
    Creates several copies (repeats) with augmentation.
    Returns a list of (feat, emos).
    """
    out = []
    for _ in range(repeats):
        feat, emos = process_row(row, augment=True, max_pad_length=max_pad_length)
        if feat is not None:
            out.append((feat, emos))
    return out

# ------------------------------------------------------------------------------------
# FEATURE EXTRACTION SCRIPT
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    start_time_total = time.time()
    print("🔄 Starting feature extraction...")

    df = pd.read_csv(DATASET_CSV)
    max_pad_length = 100

    # Emotions considered "rare" (based on statistics)
    rare_emotions = {"amazement", "power", "sadness", "tenderness"}

    results = []

    # ------------------------
    # 1) Pass without augmentation
    # ------------------------
    print("--- Extracting (without augmentation) ---")
    start_time_no_aug = time.time()
    partial_no_aug = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_row)(row, augment=False, max_pad_length=max_pad_length)
        for _, row in tqdm(df.iterrows(), total=len(df))
    )
    partial_no_aug = [res for res in partial_no_aug if res[0] is not None]
    results.extend(partial_no_aug)
    end_time_no_aug = time.time()
    print(f"✔ Done no-aug pass in {end_time_no_aug - start_time_no_aug:.2f} s, got {len(partial_no_aug)} samples.")

    # ------------------------
    # 2) Pass with augmentation (general)
    # ------------------------
    print("--- Extracting (with augmentation) ---")
    start_time_aug = time.time()
    partial_aug = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_row)(row, augment=True, max_pad_length=max_pad_length)
        for _, row in tqdm(df.iterrows(), total=len(df))
    )
    partial_aug = [res for res in partial_aug if res[0] is not None]
    results.extend(partial_aug)
    end_time_aug = time.time()
    print(f"✔ Done aug pass in {end_time_aug - start_time_aug:.2f} s, got {len(partial_aug)} samples.")

    # ------------------------
    # 3) Additional pass: oversampling for rare emotions
    # ------------------------
    print("--- Oversampling rare classes ---")
    start_time_rare = time.time()

    # First collect rows that have at least one rare_emotion
    rare_rows = []
    for _, row in df.iterrows():
        for re in rare_emotions:
            if row[re] == 1:
                rare_rows.append(row)
                break

    print(f"Found {len(rare_rows)} rows with rare emotions. Will oversample each row x2 with augmentation.")

    partial_rare = Parallel(n_jobs=-1, backend="loky")(
        delayed(oversample_task)(row, repeats=2, max_pad_length=max_pad_length) 
        for row in tqdm(rare_rows, total=len(rare_rows))
    )
    # partial_rare is a list of lists
    rare_results = []
    for sublist in partial_rare:
        rare_results.extend(sublist)

    results.extend(rare_results)
    end_time_rare = time.time()
    print(f"✔ Done oversampling pass in {end_time_rare - start_time_rare:.2f} s, got {len(rare_results)} samples.")

    # ------------------------
    # Final assembly
    # ------------------------
    X_list, y_list = zip(*results)
    X = np.array(X_list)
    y = np.array(y_list, dtype=object)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FEATURES_X), exist_ok=True)
    joblib.dump(X, OUTPUT_FEATURES_X)
    joblib.dump(y, OUTPUT_FEATURES_Y)

    end_time_total = time.time()
    print(f"\n✅ Extracted {len(X)} samples total.")
    print(f"   Feature shape = {X.shape}, e.g. (N, 123, {max_pad_length})")
    print(f"   Emotions shape = {y.shape}")
    print(f"Features saved to:\n  {OUTPUT_FEATURES_X}\n  {OUTPUT_FEATURES_Y}")
    print(f"🏁 Total time: {end_time_total - start_time_total:.2f} seconds.")
