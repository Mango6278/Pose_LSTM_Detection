"""
Author: Tom Gelhorn, Mika Laubert
pose_Utils.py (c) 2026
Desc: description
Created:  2026-01-07T08:40:00.459Z
Modified: 2026-01-07T12:06:03.708Z
"""

import numpy as np
import cv2

POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23),
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29),
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
])

def preprocess_landmarks(landmarks):
    
    if not landmarks:
        return np.zeros(33 * 4) # (x, y, z, vis)

    data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])

    # ref: hip center
    hip_x = (data[23, 0] + data[24, 0]) / 2
    hip_y = (data[23, 1] + data[24, 1]) / 2
    
    data[:, 0] -= hip_x 
    data[:, 1] -= hip_y
    
    # torso size normalization
    shoulder_center = (data[11, :3] + data[12, :3]) / 2
    hip_center = (data[23, :3] + data[24, :3]) / 2

    torso_size = np.linalg.norm(shoulder_center - hip_center)

    if torso_size > 0:
        data[:, :3] /= torso_size

    return data.flatten()

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        height, width, _ = annotated_image.shape
        
        for landmark in pose_landmarks:
             cx, cy = int(landmark.x * width), int(landmark.y * height)
             cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)

        for connection in POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = pose_landmarks[start_idx]
            end_point = pose_landmarks[end_idx]
            x1, y1 = int(start_point.x * width), int(start_point.y * height)
            x2, y2 = int(end_point.x * width), int(end_point.y * height)
            cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return annotated_image