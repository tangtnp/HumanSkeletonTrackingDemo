# Human Skeleton Tracking Demo

Detects people in a video or webcam stream, estimates each person's body
skeleton, tracks them across frames with stable IDs, and overlays movement
information (joint angles, speed, motion trail, squat repetition count, and
body orientation).

```
Input  ->  Pose Estimation  ->  Tracking  ->  Analysis  ->  Visualization
webcam     YOLO11-pose          ByteTrack     angles /       skeleton + info
/ video    (COCO-17 kpts)       (persistent   speed /        panel on screen
                                 IDs)          reps / trail
```

## Features

| Stage | What it does |
|-------|--------------|
| **Input** | Webcam (`--source 0`) or a video file (`--source path.mp4`) |
| **Pose estimation** | Ultralytics **YOLO11-pose**, 17 COCO keypoints per person |
| **Tracking** | Built-in **ByteTrack** assigns a persistent `ID` to each person |
| **Analysis** | Knee & elbow **joint angles**, body-center **speed** (m/s), motion **trajectory**, a **squat rep counter**, and **body orientation** (facing direction + sideways lean) |
| **Visualization** | Coloured skeleton, ID box, per-person info panel, **facing arrow**, live FPS |

## Requirements

- macOS / Linux / Windows, Python 3.9+
- Apple Silicon (M-series) GPU is auto-used via `mps`; CUDA and CPU also work.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The YOLO-pose weights (`yolo11n-pose.pt`, ~6 MB) download automatically on the
first run, so an internet connection is needed once.

## Run

```bash
# Webcam
python main.py --source 0

# Video file
python main.py --source samples/walk.mp4

# Save an annotated output video (great for the demo recording)
python main.py --source samples/walk.mp4 --save output.mp4
```

Press **`q`** in the window to quit.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Webcam index (e.g. `0`) or path to a video file |
| `--save` | – | Path to write an annotated `.mp4` |

## Project structure

```
main.py           # CLI + the per-frame pipeline loop (run this)
src/
  pose_tracker.py # YOLO-pose + ByteTrack -> stream of tracked Person objects
  analyzer.py     # joint angles, speed, trajectory, squat rep counter
  visualizer.py   # skeleton / box / info-panel drawing
  keypoints.py    # COCO-17 keypoint names + skeleton edges
report/REPORT.md  # 1-page write-up (method, challenges, etc.)
```

## Notes & limitations

- Speed in m/s is **approximate**: pixels are converted to metres assuming each
  person's bounding-box height ≈ 1.70 m, so it is a rough estimate, not a
  calibrated measurement.
- The squat counter is a simple knee-angle state machine; it works best when
  the legs are clearly visible from the side.
- Body orientation (Front / Back / Left / Right + sideways lean) is a geometric
  heuristic from the shoulders, hips, and face keypoints. It is a coarse
  estimate, not a calibrated 3-D pose, so front/back can be ambiguous when the
  face is occluded.
- `yolo11n` (nano) favours speed. Edit `model_path` in `pose_tracker.py` to a
  larger variant (`s`/`m`/`l`/`x`) for crowded or low-quality footage.
