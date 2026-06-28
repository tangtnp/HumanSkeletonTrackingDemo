"""Human Skeleton Tracking Demo - entry point.

Pipeline:  Input -> Pose Estimation -> Tracking -> Analysis -> Visualization

Examples:
    python main.py --source 0                                   # webcam
    python main.py --source 0 --save output.mp4                 # webcam with output
    python main.py --source samples/walk.mp4                    # video file
    python main.py --source samples/walk.mp4 --save output.mp4  # video file with output
"""

import argparse
import time

import cv2

from src import visualizer
from src.analyzer import MovementAnalyzer
from src.pose_tracker import PoseTracker

def parse_args():
    p = argparse.ArgumentParser(description="Human Skeleton Tracking Demo")
    p.add_argument("--source", default="0", help="webcam or video path")
    p.add_argument("--save", default=None, help="output video path")
    return p.parse_args()

def resolve_source(source):
    """int 0 (webcam)/strings (file path)"""
    return int(source) if source.isdigit() else source


def main():
    args = parse_args()
    source = resolve_source(args.source)
    is_webcam = isinstance(source, int)

    fps = 30.0

    tracker = PoseTracker()
    analyzer = MovementAnalyzer()
    writer = None

    print(f"[info] device={tracker.device}  source={source}  fps~={fps:.1f}")
    print("[info] press 'q' in the window to quit")

    last = time.perf_counter()
    disp_fps = fps
    start_wall = last

    for frame in tracker.stream(source):
        if is_webcam:
            t = time.perf_counter() - start_wall
        else:
            t = frame.index / fps

        for person in frame.persons:
            analyzer.update(person, t)

        now = time.perf_counter()
        dt = now - last
        last = now
        if dt > 0:
            disp_fps = 0.9 * disp_fps + 0.1 * (1.0 / dt)

        img = visualizer.render(frame, analyzer, disp_fps)

        if args.save:
            if writer is None:
                h, w = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
            writer.write(img)

        cv2.imshow("Human Skeleton Tracking Demo", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if writer is not None:
        writer.release()
        print(f"[info] saved annotated video -> {args.save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
