import numpy as np
import joblib
import tensorflow as tf
import time
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
)
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# Paths
X_PATH = "joblib_features/X_emotify.joblib"
Y_PATH = "joblib_features/y_emotify.joblib"
MODEL_PATH = "model/Emotion_Voice_Detection_Model_Emotify.h5"

########################
# Optional function for threshold tuning
########################
def find_optimal_thresholds(y_true, y_prob, classes, step=0.01):
    """
    For each class separately, we iterate over thresholds in the [0..1] range
    to maximize the F1-score on the validation data.
    Returns an array of optimal thresholds for each class.
    """
    import numpy as np
    from sklearn.metrics import f1_score

    n_classes = len(classes)
    thresholds = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.0, 1.01, step):
            y_pred_t = (y_prob[:, c] >= t).astype(int)
            score_t = f1_score(y_true[:, c], y_pred_t, zero_division=0)
            if score_t > best_f1:
                best_f1 = score_t
                best_t = t
        thresholds[c] = best_t
    return thresholds


if __name__ == '__main__':
    start_time = time.time()

    print("📂 Loading data...")
    X = joblib.load(X_PATH)   # Assumed shape ~ (N, 123, 100) after combined features
    y = joblib.load(Y_PATH)   # list of lists (multi-label)

    # Add a channel dimension so that Conv2D expects (batch, height, width, channels)
    # Here height=123 (mel+mfcc+chroma+contrast), width=100 (max_pad_length)
    X = np.expand_dims(X, axis=-1)
    print("X shape after expand_dims:", X.shape)  
    # Should become (N, 123, 100, 1)

    # List of emotions
    emotion_labels = [
        "amazement", "solemnity", "tenderness", "nostalgia", "calmness",
        "power", "joyful_activation", "tension", "sadness"
    ]

    print("🎭 Encoding emotions...")
    mlb = MultiLabelBinarizer(classes=emotion_labels)
    y_encoded = mlb.fit_transform(y)
    print(f"🔍 Example of encoded labels:\n{y_encoded[:5]}")
    num_classes = y_encoded.shape[1]
    print(f"✅ Number of classes: {num_classes}")

    # Split into train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    batch_size = 16

    # Use tf.data.Dataset for faster loading
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
                                .shuffle(len(X_train)) \
                                .batch(batch_size) \
                                .prefetch(tf.data.experimental.AUTOTUNE)

    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
                               .batch(batch_size) \
                               .prefetch(tf.data.experimental.AUTOTUNE)

    print("🧠 Creating model...")

    # Deeper CNN (3 convolutional blocks + 2 Dense layers)
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2], 1)),  # (123, 100, 1)

        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        BatchNormalization(),

        Dense(128, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),

        Dense(num_classes, activation='sigmoid')  # multi-label => sigmoid
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    model_checkpoint = ModelCheckpoint(
        "model/best_model.h5",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    epochs = 100
    print("🚀 Starting training...")
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr, model_checkpoint],
        verbose=1
    )

    # Save the final model (after the last training epoch)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    print("\n🏁 Evaluating the model on the test set...")
    test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
    print(f"Test Loss = {test_loss:.4f}")
    print(f"Test Accuracy = {test_accuracy:.4f}")

    # Gather predictions
    all_y_true = []
    all_y_pred_prob = []
    for X_batch, y_batch in test_data:
        preds = model.predict(X_batch)
        all_y_true.append(y_batch.numpy())
        all_y_pred_prob.append(preds)

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred_prob = np.concatenate(all_y_pred_prob, axis=0)

    # --- (1) STANDARD APPROACH (threshold=0.5) ---
    all_y_pred_05 = (all_y_pred_prob >= 0.5).astype(int)
    print("\nClassification Report (threshold=0.5):")
    print(classification_report(
        all_y_true,
        all_y_pred_05,
        target_names=emotion_labels,
        zero_division=0
    ))

    # --- (2) FINE-TUNE THE THRESHOLD (optional) ---
    print("\n🔎 Searching for optimal thresholds per class...")
    best_thresholds = find_optimal_thresholds(all_y_true, all_y_pred_prob, emotion_labels, step=0.01)
    print("Best thresholds:", best_thresholds)

    # Apply them
    all_y_pred_custom = np.zeros_like(all_y_pred_prob, dtype=int)
    for c in range(num_classes):
        all_y_pred_custom[:, c] = (all_y_pred_prob[:, c] >= best_thresholds[c]).astype(int)

    print("\nClassification Report (custom thresholds):")
    print(classification_report(
        all_y_true,
        all_y_pred_custom,
        target_names=emotion_labels,
        zero_division=0
    ))

    end_time = time.time()
    print(f"\n✅ Model trained and saved at {MODEL_PATH}")
    print(f"⏳ Training time: {end_time - start_time:.2f} seconds")
