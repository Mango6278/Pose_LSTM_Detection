"""
Author: Hello (World)
fileIo_Utils.py (c) 2026
Desc: description
Created:  2026-01-07T08:41:28.916Z
Modified: 2026-01-07T09:41:20.726Z
"""

import os
import json
import csv
import numpy as np

def load_video_jobs(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file '{filepath}' not found.")
    
    jobs = []
    
    if filepath.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            jobs = json.load(f)
            
    elif filepath.endswith('.csv'):
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('input'): 
                    jobs.append(row)
    else:
        raise ValueError("Config file not of type .json or .csv.")
        
    return jobs

def save_dataset(dataset, filename="dataset.npz"):

    X = []
    y = []
    
    # Label Encoding (String -> Int)
    unique_labels = sorted(list(set(d['label'] for d in dataset)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    for entry in dataset:
        X.append(entry['features'])
        y.append(label_map[entry['label']])
        
    np.savez_compressed(filename, X=np.array(X, dtype=object), y=np.array(y), labels=unique_labels)
    print(f"Dataset saved at '{filename}'. Labels: {label_map}. Number of samples: {len(dataset)}")

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file '{path}' not found.")
        
    with open(path, 'r') as f:
        return json.load(f)