"""
Author: Tom Gelhorn, Mika Laubert
analyze_video.py (c) 2026
Desc: description
Created:  2026-01-07T12:18:23.853Z
Modified: 2026-01-11T15:49:21.347Z
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pose_Utils import preprocess_landmarks, draw_landmarks_on_image, compute_velocity
from fileIo_Utils import load_config

MODEL_PATH          = 'model/pose_lstm_model.keras'
MODEL_CONFIG_PATH   = 'model/pose_lstm_model_config.json'
LANDMARKER_PATH     = 'model/pose_landmarker_full.task'
INPUT_VIDEO         = 'datasets/prototypingData/videos/VID_20260108_150009.mp4'
OUTPUT_VIDEO        = 'datasets/prototypingData/analyzed_result.mp4'
THRESHOLD           = 0.7 # above: show class

SHOW_SEGMENTATION   = True
SHOW_LANDMARKS      = True

COL_DARK_GREY   = (50, 50, 50)
COL_LIGHT_GREY  = (200, 200, 200)
COL_WHITE       = (255, 255, 255)

def init_detector(model_path):
    # similar to Pose_LSTM_Detection.py
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=SHOW_SEGMENTATION,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)

def main():

    config = load_config(MODEL_CONFIG_PATH)

    current_classes = config.get("classes", ['category1', 'noCategory'])
    seq_length = config.get("sequence_length", 100)

    model = tf.keras.models.load_model(MODEL_PATH)
    
    detector = init_detector(LANDMARKER_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    sequence_buffer = []
    current_label = "Waiting..."
    current_prob = 0.0

    frame_index = 0
    
    print(f"Starting analysis for '{INPUT_VIDEO}'...")

    prev_inference_feature = np.zeros(33 * 4)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # media pipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((frame_index * 1000) / fps)
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            base_features = preprocess_landmarks(landmarks)
        else:
            base_features = np.zeros(33 * 4) # Zero-Feature-Vektor
        
        velocity = compute_velocity(base_features, prev_inference_feature)
        full_features = np.concatenate((base_features, velocity))

        prev_inference_feature = base_features
        sequence_buffer.append(full_features)

        if detection_result.segmentation_masks:
            mask = detection_result.segmentation_masks[0].numpy_view() # Shape (H, W, 1)
            
            mask = np.squeeze(mask)

            # (H, W, 3)
            mask_3d = np.stack((mask,) * 3, axis=-1)

            OVERLAY_COLOR = (255, 255, 0) 
            OPACITY = 0.3 

            colored_overlay = np.zeros_like(frame, dtype=np.uint8)
            colored_overlay[:] = OVERLAY_COLOR

            frame_float = frame.astype(float)
            overlay_float = colored_overlay.astype(float)
            
            weight = mask_3d * OPACITY
            blended = frame_float * (1 - weight) + overlay_float * weight
            
            frame = blended.astype(np.uint8)
        
        if len(sequence_buffer) > seq_length:
            sequence_buffer.pop(0)

        # Prediction (buffer full?)
        if len(sequence_buffer) == seq_length:
            # (1, 40, 132) -> Batch size 1
            input_data = np.expand_dims(sequence_buffer, axis=0)
            
            prediction = model.predict(input_data, verbose=0)[0] # Prediction per class, e.g.: [0.1, 0.9, ...]
            class_idx = np.argmax(prediction) # class with highest probability
            
            current_prob = prediction[class_idx]
            
            if current_prob > THRESHOLD:
                current_label = current_classes[class_idx]
            else:
                current_label = "Unsure..."

        if detection_result.pose_landmarks:
            if SHOW_LANDMARKS:
                frame = draw_landmarks_on_image(frame, detection_result)

        # visuals
        text_color_status = COL_WHITE if current_prob > THRESHOLD else COL_DARK_GREY

        cv2.rectangle(frame, (0,0), (300, 60), COL_LIGHT_GREY, -1)
        cv2.putText(frame, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DARK_GREY, 1, cv2.LINE_AA)
        cv2.putText(frame, 'CLASS', (120,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DARK_GREY, 1, cv2.LINE_AA)

        prob_text = f"{current_prob:.2f}"
        label_text = current_label if current_label else "..."

        cv2.putText(frame, prob_text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color_status, 2, cv2.LINE_AA)
        cv2.putText(frame, label_text, (120,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color_status, 2, cv2.LINE_AA)

        cv2.imshow('Analysis', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Video saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()