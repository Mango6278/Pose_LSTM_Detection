"""
Author: Tom Gelhorn, Mika Laubert
train_LSTM.py (c) 2026
Desc: description
Created:  2026-01-07T12:06:33.826Z
Modified: 2026-01-07T21:28:49.804Z
"""

import logging
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

DATASET_PATH            = 'datasets/HAR/pose_dataset_full.npz'
MODEL_SAVE_PATH         = 'model/pose_lstm_model.keras'
MODEL_CONFIG_SAVE_PATH  = 'model/pose_lstm_model_config.json'
MAX_SEQ_LENGTH          = 10  # Max number of frames per video
NUM_FEATURES            = 132*2

callbacks = [
    EarlyStopping(patience=10, verbose=1, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001, verbose=1, monitor='val_loss'),
    ModelCheckpoint('model/best_pose_model.keras', save_best_only=True, monitor='val_loss')
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def add_noise(X_data, noise_factor=0.02):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=X_data.shape)
    return X_data + noise

def load_and_preprocess_data():

    if not tf.io.gfile.exists(DATASET_PATH):
        raise FileNotFoundError(f"{DATASET_PATH} not found.")
        
    data = np.load(DATASET_PATH, allow_pickle=True)
    X_raw = data['X']
    y_raw = data['y']  # Labels
    labels = data['labels'] # Label Names
    
    logging.info(f"Found {len(X_raw)} samples in Data.")

    # Padding
    # Videos < MAX_SEQ_LENGTH filled with 0, above truncated
    X_pad = pad_sequences(X_raw, maxlen=MAX_SEQ_LENGTH, dtype='float32', padding='post', truncating='post')
    
    # 3. One-Hot Encoding
    y_cat = to_categorical(y_raw)
    
    return X_pad, y_cat, labels

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape)) # zero padding mask
    
    # LSTM Layer 1
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)) # prevent overfitting
    
    # LSTM Layer 2
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Dense Layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def augment_data(X_data, y_data):
    X_aug = []
    y_aug = []
    
    for x, y in zip(X_data, y_data):
        X_aug.append(x)
        y_aug.append(y)
        
        # Noise
        noise = np.random.normal(0, 0.02, x.shape)
        X_aug.append(x + noise)
        y_aug.append(y)
        
        # Time Shift
        shift = np.random.randint(low=-5, high=5)
        x_shifted = np.roll(x, shift, axis=0)

        # Correct Padding 
        if shift > 0: x_shifted[:shift] = 0
        else: x_shifted[shift:] = 0
        X_aug.append(x_shifted)
        y_aug.append(y)

    return np.array(X_aug), np.array(y_aug)

def main():
    X, y, class_names = load_and_preprocess_data()
    num_classes = len(class_names)
    
    # Train/Test Split (80% Training, 20% Validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Data Augmentation
    # X_train, y_train = augment_data(X_train, y_train)
    
    logging.info(f"Training started with {len(X_train)} samples. Classes: {class_names}")

    model = build_model((MAX_SEQ_LENGTH, NUM_FEATURES), num_classes)
    model.summary()

    # Training
    history = model.fit(
        X_train, y_train,
        epochs=100,           # How many times through all data?
        batch_size=16,        # How many videos at once?
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    model.save(MODEL_SAVE_PATH)
    logging.info(f"Model saved at {MODEL_SAVE_PATH}")

    config = {
        "classes": class_names.tolist(),
        "sequence_length": MAX_SEQ_LENGTH,
        "num_features": NUM_FEATURES
    }

    with open(MODEL_CONFIG_SAVE_PATH, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f"Class names saved at {MODEL_CONFIG_SAVE_PATH}")

    loss, accuracy = model.evaluate(X_test, y_test)
    logging.info(f"Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()