# Pose LSTM Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13.2](https://img.shields.io/badge/python-3.13.2-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Academic_Project-green.svg)]()

> **Short Description:** A vision-based gesture recognition pipeline for automated systems in transportation, utilizing MediaPipe for pose estimation and a 2-layer LSTM for classification.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Directory Structure](#-directory-structure)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Usage](#-usage)
  - [Training](#training)
  - [Inference](#inference)
- [Results & Benchmarks](#-results--benchmarks)
- [Pre-trained Models](#-pre-trained-models)
- [Citation](#-citation)

---

## 📖 Overview

This project addresses gesture-based human-machine interaction in transportation, specifically designed to allow Unmanned Aerial Vehicles (UAVs) and Autonomous Vehicles (AVs) to recognize human signals. The system processes RGB video input of any size and framerate.

The architecture follows a two-step procedure:
1.  **Feature Extraction:** Videos are preprocessed using the MediaPipe framework to extract normalized body-landmark sequences.
2.  **Classification:** A Long Short-Term Memory (LSTM) Neural Network predicts the human gesture based on the temporal dynamics of the landmarks.

This implementation was developed as part of the "Machine Learning and Computer Vision" course (Winter 2025/2026) at DHBW Stuttgart.

## ✨ Key Features
- **High Accuracy:** Achieves a validation accuracy of 96.34% across 5 gesture classes.
- Robust Normalization:** The model is invariant to input resolution, aspect ratio, and camera distance due to a translation and scaling pipeline based on torso size.
- **Lightweight Architecture:** Utilizes a lightweight 2-layer stacked LSTM (64 units each) suitable for embedded systems.
- **Negative Class Handling:** Includes a specific "Negative" class to reduce false positives by training on random movements that resemble target gestures (e.g., praying vs. clapping).
- **Data Augmentation:** Implements Gaussian noise and random time shifts to prevent overfitting and ensure robustness against landmark detection jitter.

---

## 📂 Directory Structure

```bash
.
├── conf/                   # Configuration files
├── model/                  # LSTM architecture definitions
├── scripts/                # Includes all Python scripts
