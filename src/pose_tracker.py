"""Pose estimation + tracking"""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Person:
    """One tracked person in a single frame."""
    track_id: int
    bbox: np.ndarray
    keypoints: np.ndarray

@dataclass
class Frame:
    """The raw image frame and everyone detected in it."""
    image: np.ndarray
    persons: list
    index: int
    fps: float


class PoseTracker:
    def __init__(self, model_path="yolo11n-pose.pt", device=None, conf=0.5):
        """
        model_path: YOLO-pose
        device:     'mps', 'cuda', 'cpu', or None (auto-pick)
        conf:       minimum detection confidence.
        """
        self.model = YOLO(model_path)
        self.device = device or self._auto_device()
        self.conf = conf

    @staticmethod
    def _auto_device():
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def stream(self, source):
        """Stream frames with tracked poses from the input source."""
        results = self.model.track(
            source=source,
            stream=True,
            persist=True,
            conf=self.conf,
            device=self.device,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        for index, result in enumerate(results):
            persons = self._parse(result)
            yield Frame(image=result.orig_img, persons=persons,
                        index=index, fps=30)

    @staticmethod
    def _parse(result):
        persons = []
        if result.keypoints is None or result.boxes is None:
            return persons
        if result.boxes.id is None:
            return persons

        ids = result.boxes.id.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()
        kpts = result.keypoints.data.cpu().numpy()

        for tid, box, kp in zip(ids, boxes, kpts):
            persons.append(Person(track_id=int(tid), bbox=box, keypoints=kp))
        return persons
