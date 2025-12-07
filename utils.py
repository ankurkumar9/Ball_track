import os
import cv2

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def video_writer(out_path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(out_path, fourcc, fps, (width, height))

def list_videos(input_path: str):
    if os.path.isfile(input_path):
        return [input_path]

    vids = []
    for root, _, files in os.walk(input_path):
        for fn in files:
            if fn.lower().endswith((".mp4", ".mov", ".avi", ".mkv")) and not fn.startswith("._"):
                vids.append(os.path.join(root, fn))
    vids.sort()
    return vids
