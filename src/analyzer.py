"""Turns tracked keypoints into per-person movement metrics.

For each track ID we keep a little history and compute joint angles, the speed
of the body center, a motion trail, a squat counter, and which way the body is
facing.
"""
from collections import defaultdict, deque
import numpy as np
from src.keypoints import KP

MIN_CONF = 0.5

SQUAT_DOWN_ANGLE = 120
SQUAT_UP_ANGLE = 160

ASSUMED_HEIGHT_M = 1.70
PROFILE_RATIO = 0.33


def joint_angle(a, b, c):
    """Angle in degrees at point b (between the segments b->a and b->c).

    Returns None if any of the three points is missing.
    """
    if a is None or b is None or c is None:
        return None
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

class PersonState:
    """Info of one tracked person between frames."""
    def __init__(self, history=64):
        self.centers = deque(maxlen=history)
        self.trail = deque(maxlen=history)
        self.speed_px = 0.0
        self.speed_ms = 0.0
        self.knee_angle = None
        self.elbow_angle = None
        self.squat_reps = 0
        self.facing = None
        self.lean_angle = None
        self._squat_phase = "up"


class MovementAnalyzer:
    def __init__(self):
        self.states = defaultdict(PersonState)

    @staticmethod
    def _kp(kps, name):
        """Look up a keypoint by name, or None if it isn't confident enough."""
        p = kps[KP[name]]
        return p if p[2] >= MIN_CONF else None

    def update(self, person, t):
        """Update one person's metrics for the frame at time t (seconds)."""
        state = self.states[person.track_id]
        kps = person.keypoints

        # Midpoint of the hips as the body center
        left_hip = self._kp(kps, "left_hip")
        right_hip = self._kp(kps, "right_hip")
        if left_hip is not None and right_hip is not None:
            cx, cy = (left_hip[:2] + right_hip[:2]) / 2
        else:
            x1, y1, x2, y2 = person.bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        state.centers.append((t, cx, cy))
        state.trail.append((int(cx), int(cy)))

        state.speed_px, state.speed_ms = self._speed(state, person.bbox)
        state.knee_angle = self._best_angle(kps, "hip", "knee", "ankle")
        state.elbow_angle = self._best_angle(kps, "shoulder", "elbow", "wrist")
        self._count_squats(state)
        state.facing, state.lean_angle = self._orientation(kps)
        return state

    def _best_angle(self, kps, a, b, c):
        """Angle for whichever side (left or right) is fully visible."""
        for side in ("left", "right"):
            angle = joint_angle(self._kp(kps, f"{side}_{a}"),
                                self._kp(kps, f"{side}_{b}"),
                                self._kp(kps, f"{side}_{c}"))
            if angle is not None:
                return angle
        return None

    def _speed(self, state, bbox):
        if len(state.centers) < 2:
            return 0.0, 0.0
        (t0, x0, y0), (t1, x1, y1) = state.centers[0], state.centers[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0, 0.0
        speed_px = np.hypot(x1 - x0, y1 - y0) / dt
        m_per_px = ASSUMED_HEIGHT_M / max(bbox[3] - bbox[1], 1.0)
        return speed_px, speed_px * m_per_px

    def _count_squats(self, state):
        angle = state.knee_angle
        if angle is None:
            return
        if state._squat_phase == "up" and angle < SQUAT_DOWN_ANGLE:
            state._squat_phase = "down"
        elif state._squat_phase == "down" and angle > SQUAT_UP_ANGLE:
            state._squat_phase = "up"
            state.squat_reps += 1

    def _orientation(self, kps):
        """Direction of the person faces, and how much they lean sideways."""
        left_sh = self._kp(kps, "left_shoulder")
        right_sh = self._kp(kps, "right_shoulder")
        if left_sh is None or right_sh is None:
            return None, None

        shoulder_mid = (left_sh[:2] + right_sh[:2]) / 2
        shoulder_span = abs(left_sh[0] - right_sh[0])

        left_hip = self._kp(kps, "left_hip")
        right_hip = self._kp(kps, "right_hip")
        torso_height = None
        lean = None
        if left_hip is not None and right_hip is not None:
            hip_mid = (left_hip[:2] + right_hip[:2]) / 2
            dx = shoulder_mid[0] - hip_mid[0]
            dy = hip_mid[1] - shoulder_mid[1]
            torso_height = float(np.hypot(dx, dy))
            if torso_height > 1.0:
                lean = float(np.degrees(np.arctan2(dx, dy)))

        nose = self._kp(kps, "nose")
        if torso_height is not None and shoulder_span < PROFILE_RATIO * torso_height:
            ref_x = nose[0] if nose is not None else shoulder_mid[0]
            facing = "Left" if ref_x <= shoulder_mid[0] else "Right"
        else:
            facing = "Front" if (left_sh[0] > right_sh[0] or nose is not None) else "Back"
        return facing, lean
