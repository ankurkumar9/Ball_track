import os
import csv
import argparse
import cv2
from tqdm import tqdm

from ball_detector import BallDetector
from tracker import SimpleTracker
from utils import ensure_dir, video_writer, list_videos


def process_single_video(video_path: str, annotations_dir: str, results_dir: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_video = os.path.join(results_dir, f"{base}_annotated.mp4")
    out_csv = os.path.join(annotations_dir, f"{base}.csv")

    writer = video_writer(out_video, fps, width, height)

    detector = BallDetector(scale=0.5)
    tracker = SimpleTracker(max_history=2000)

    rows = [["frame", "x", "y", "visible"]]
    frame_idx = 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else None
    pbar = tqdm(total=total, desc=f"Processing {base}", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det = detector.detect(frame)

        tracker.update(det)

        if det is not None:
            x, y = det
            visible = 1
            cv2.circle(frame, det, 8, (0, 255, 0), 2)
        else:
            x = y = -1
            visible = 0

        rows.append([frame_idx, float(x), float(y), int(visible)])

        traj = tracker.get_trajectory()
        for i in range(1, len(traj)):
            cv2.line(frame, traj[i - 1], traj[i], (255, 0, 0), 2)

        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"[INFO] Saved CSV:   {out_csv}")
    print(f"[INFO] Saved video: {out_video}")


def main():
    parser = argparse.ArgumentParser(description="EdgeFleet cricket ball tracking pipeline")
    parser.add_argument("--input", required=True, help="Input file or directory of videos")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    annotations_dir = os.path.join(root, "annotations")
    results_dir = os.path.join(root, "results")
    ensure_dir(annotations_dir)
    ensure_dir(results_dir)

    videos = list_videos(args.input)
    print(f"[INFO] Found {len(videos)} video(s).")

    for vp in videos:
        process_single_video(vp, annotations_dir, results_dir)


if __name__ == "__main__":
    main()
