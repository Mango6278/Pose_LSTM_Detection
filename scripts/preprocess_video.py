"""
Author: Tom Gelhorn, Mika Laubert
preprocess_video.py (c) 2026
Desc: Feature extraction pipeline with FPS Normalization
Created:  2026-01-07T08:18:20.984Z
Modified: 2026-01-10T15:12:38.652Z
"""

import logging
import traceback
import cv2
import mediapipe as mp
import numpy as np
import os
import concurrent.futures
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pose_Utils import preprocess_landmarks, draw_landmarks_on_image, compute_velocity
from fileIo_Utils import load_video_jobs, save_dataset

CONFIG_FILE = 'conf/video_jobs.json'
MODEL_PATH = 'model/pose_landmarker_full.task'
OUTPUT_FILE = 'datasets/prototypingData/walk_dataset.npz'

DEBUG_VISUALIZE = False # draws pose landmarks on video and saves it
DEBUG_VISUALIZE_FOLDER = 'debug_visualizations/'

TARGET_FPS = 30 

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

def process_single_wrapper(job):
    try:
        with init_detector(MODEL_PATH) as detector:
            return process_single_video(job, detector)
    except Exception as e:
        logging.error(f"Worker Error at {job.get('input')}: {e}")
        return None

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

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or np.isnan(source_fps):
        source_fps = 30 # Fallback
        logging.warning(f"FPS invalid for {input_path}, assuming 30 FPS.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps_ratio = source_fps / TARGET_FPS

    # Video Writer
    out = None
    if DEBUG_VISUALIZE:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (width, height))
    
    frame_data = []
    
    current_frame_idx = 0       # Counter for original frames
    next_target_frame_idx = 0.0 # counter for target frames
    
    last_valid_feature = None
    prev_loop_feature  = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        if current_frame_idx >= int(next_target_frame_idx):
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            extracted_count = len(frame_data)
            timestamp_ms = int((extracted_count * 1000) / TARGET_FPS)

            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                feature_vector = preprocess_landmarks(landmarks)
                last_valid_feature = feature_vector
            else:
                feature_vector = handle_missing_pose(last_valid_feature)

            velocity = compute_velocity(feature_vector, prev_loop_feature)
            combined_features = np.concatenate((feature_vector, velocity))
            
            frame_data.append(combined_features)
            prev_loop_feature = feature_vector

            if DEBUG_VISUALIZE and out:
                annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            
            next_target_frame_idx += fps_ratio

        current_frame_idx += 1

    cap.release()
    if out: out.release()
    
    if not frame_data:
        logging.warning(f"No Pose found in {input_path}.")
        return None

    logging.info(f"  -> Finished {input_path}: {len(frame_data)} Frames extracted (Resampled {source_fps:.1f}->{TARGET_FPS}).")
    
    return {
        "label": job_config.get("label", "unknown"),
        "features": np.array(frame_data, dtype=np.float32),
        "source_file": input_path
    }

def run_extraction_pipeline(video_jobs):
    dataset = []
    total = len(video_jobs)
    
    MAX_WORKERS = os.cpu_count() - 2 if os.cpu_count() > 2 else 1
    
    logging.info(f"Starting extraction for {total} videos on {MAX_WORKERS} cores...")
    # paralell processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        future_to_job = {executor.submit(process_single_wrapper, job): job for job in video_jobs}        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_job)):
            job = future_to_job[future]
            try:
                result = future.result()
                if result:
                    dataset.append(result)
                logging.info(f"[{i+1}/{total}] Finished: {job.get('input')}")
            except Exception as e:
                logging.error(f"Exception at {job.get('input')}: {e}")
            
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