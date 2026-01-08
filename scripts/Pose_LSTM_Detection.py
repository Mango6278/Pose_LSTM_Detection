"""
Author: Tom Gelhorn, Mika Laubert
Pose_LSTM_Detection.py (c) 2026
Desc: description
Created:  2026-01-07T08:18:20.984Z
Modified: 2026-01-07T10:08:04.409Z
"""

import logging
import traceback
import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pose_Utils import preprocess_landmarks, draw_landmarks_on_image
from fileIo_Utils import load_video_jobs, save_dataset

CONFIG_FILE = 'conf/video_jobs.json'
MODEL_PATH = 'model/pose_landmarker_full.task'
OUTPUT_FILE = 'datasets/prototypingData/pose_dataset.npz'

DEBUG_VISUALIZE = True # draws pose landmarks on video and saves it
DEBUG_VISUALIZE_FOLDER = 'debug_visualizations/'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

all_training_data = []

def init_detector(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)

def handle_missing_pose(last_valid_pose):
    # zero order hold, eventually interpolation later
    if last_valid_pose is not None:
        return last_valid_pose
    else:
        return np.zeros(33 * 4)

def process_video_batch(video_jobs, detector):
    dataset = []
    
    for i, job in enumerate(video_jobs):
        print(f"Processing {i+1}/{len(video_jobs)}: {job.get('input')}")
        
        try:
            result = process_single_video(job, detector)
            if result:
                dataset.append(result)
        except Exception as e:
            print(f"Error processing {job.get('input')}: {e}")
            
    return dataset

def process_single_video(job_config, detector):
    input_path = job_config["input"]

    parent_dir_path = os.path.dirname(input_path)
    output_path = os.path.join(DEBUG_VISUALIZE_FOLDER, os.path.basename(parent_dir_path), os.path.basename(input_path))

    if not os.path.exists(input_path):
        logging.warning(f"File '{input_path}' not found. Skipping...")
        return None

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error(f"Could not open '{input_path}'.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video Writer
    out = None
    if DEBUG_VISUALIZE:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_data = []
    frame_index = 0
    last_valid_feature = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((frame_index * 1000) / fps)

        # Detection
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        
        # Feature Extraction
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            feature_vector = preprocess_landmarks(landmarks)
            last_valid_feature = feature_vector
        else:
            # Fallback
            feature_vector = handle_missing_pose(last_valid_feature)

        frame_data.append(feature_vector)

        # Visualization
        if DEBUG_VISUALIZE and out:
            annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

        frame_index += 1

    cap.release()
    if out: out.release()
    
    if not frame_data:
        logging.warning(f"No Pose found in {input_path}.")
        return None

    logging.info(f"  -> Finished {input_path}: {len(frame_data)} Frames extracted.")
    
    return {
        "label": job_config.get("label", "unknown"),
        "features": np.array(frame_data, dtype=np.float32),
        "source_file": input_path
    }

def run_extraction_pipeline(video_jobs):
    dataset = []
    total = len(video_jobs)
    
    logging.info(f"Stareting extraction for {total} videos...")

    for i, job in enumerate(video_jobs):
        logging.info(f"[{i+1}/{total}] Processing {job.get('input')}...")
        
        try:
            with init_detector(MODEL_PATH) as detector:
                result = process_single_video(job, detector)
                
            if result:
                dataset.append(result)
                
        except Exception as e:
            logging.error(f"Error processing {job.get('input')}: {e}")
            traceback.print_exc()
            
    return dataset

def main():
    try:
        video_jobs = load_video_jobs(CONFIG_FILE)
    except Exception as e:
        logging.critical(f"Config Error: {e}")
        return

    try:
        dataset = run_extraction_pipeline(video_jobs)

        if dataset:
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            save_dataset(dataset, OUTPUT_FILE)
        else:
            logging.warning("Dataset empty. Nothing saved.")
                
    except Exception as e:
        logging.critical(f"Pipeline Failed: {e}")

if __name__ == "__main__":
    main()