import os
import cv2
from multiprocessing import Pool


def extract_frames(vid):
    print(f'Processing {vid}...')
    path, _ = os.path.split(vid)
    output_path = f'{path}/frames'
    os.makedirs(output_path, exist_ok=True)

    vidcap = cv2.VideoCapture(vid)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"{output_path}/frame{count}.jpg", image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1


in_data = []
for dirpath, dirnames, filenames in os.walk('/ssd2/duc/people_snapshot_public'):
    for file in filenames:
        if file.endswith('.mp4'):
            in_data.append(os.path.join(dirpath, file))

with Pool(8) as p:
    p.map(extract_frames, in_data)
