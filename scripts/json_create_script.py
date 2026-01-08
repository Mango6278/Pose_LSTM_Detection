import os
import json
import re

VIDEO_DIR = '../datasets/prototypingData/videos'
OUTPUT_JSON = '../conf/video_jobs.json'
OUTPUT_DIR = '../output'

def extract_label(filename):
    match = re.match(r'([A-Za-z]+)', filename)
    return match.group(1) if match else 'unknown'

def main():
    jobs = []
    for fname in os.listdir(VIDEO_DIR):
        if not (fname.lower().endswith('.mp4') or fname.lower().endswith('.avi')):
            continue
        label = extract_label(fname)
        input_path = os.path.join(VIDEO_DIR, fname)
        output_name = os.path.splitext(fname)[0] + '_out' + os.path.splitext(fname)[1]
        output_path = os.path.join(OUTPUT_DIR, output_name)
        jobs.append({
            'input': input_path,
            'output': output_path,
            'label': label
        })
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(jobs, f, indent=4)
    print(f"{len(jobs)} Jobs geschrieben nach {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
