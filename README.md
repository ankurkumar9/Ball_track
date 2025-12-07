# Cricket Ball Detection & Tracking (EdgeFleet.Ai Assignment)

This repo implements a complete CV pipeline to detect and track a cricket ball from single fixed-camera videos.

## Outputs (per video)
- `annotations/<video>.csv` with columns: `frame,x,y,visible`
- `results/<video>_annotated.mp4` with centroid + trajectory overlay

## Setup
```bash
pip install -r requirements.txt
```
(Optional conda)
```bash
conda env create -f environment.yml
conda activate edgefleet-ball-tracker
```

## Run
Put your videos in any folder, e.g. `data/25_nov_2025/`, then:

```bash
cd code
python pipeline.py --input ../data/25_nov_2025
# or single file
python pipeline.py --input ../data/25_nov_2025/1.mp4
```

## CSV format
If the ball is not detected in a frame, we output:
- `visible = 0`
- `x = -1`, `y = -1`
