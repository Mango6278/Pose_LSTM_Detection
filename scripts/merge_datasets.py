"""
Author: Hello (World)
merge_datasets.py (c) 2026
Desc: description
Created:  2026-01-08T14:32:01.942Z
Modified: !date!
"""

import numpy as np

def merge_npz_files(file_path_1, file_path_2, output_path):
    print(f"Loading {file_path_1}...")
    data1 = np.load(file_path_1, allow_pickle=True)
    X1, y1, labels1 = data1['X'], data1['y'], data1['labels']

    print(f"Loading {file_path_2}...")
    data2 = np.load(file_path_2, allow_pickle=True)
    X2, y2, labels2 = data2['X'], data2['y'], data2['labels']

    combined_labels = sorted(list(set(labels1.tolist() + labels2.tolist())))
    print(f"New combined label list: {combined_labels}")

    def remap_labels(y_indices, old_label_names, new_label_names_list):
        new_y = []
        for idx in y_indices:
            label_name = old_label_names[int(idx)]
            new_index = new_label_names_list.index(label_name)
            new_y.append(new_index)
        return np.array(new_y)

    print("Remapping labels...")
    y1_new = remap_labels(y1, labels1, combined_labels)
    y2_new = remap_labels(y2, labels2, combined_labels)

    print("Concatenating ...")
    X_combined = np.concatenate((X1, X2), axis=0)
    y_combined = np.concatenate((y1_new, y2_new), axis=0)

    np.savez_compressed(
        output_path, 
        X=X_combined, 
        y=y_combined, 
        labels=np.array(combined_labels)
    )
    print("Saved merged dataset to", output_path)

if __name__ == "__main__":
    FILE_A = 'datasets/prototypingData/pose_dataset_clap_wave_stand.npz'
    FILE_B = 'datasets/prototypingData/walk_dataset.npz'
    OUTPUT = 'datasets/prototypingData/merged_dataset_full.npz'
    
    try:
        merge_npz_files(FILE_A, FILE_B, OUTPUT)
    except FileNotFoundError as e:
        print(f"Error: {e}")