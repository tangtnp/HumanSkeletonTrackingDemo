KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# index lookup
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

SKELETON = [
    (KP["left_shoulder"], KP["right_shoulder"]),
    (KP["left_shoulder"], KP["left_elbow"]),
    (KP["left_elbow"], KP["left_wrist"]),
    (KP["right_shoulder"], KP["right_elbow"]),
    (KP["right_elbow"], KP["right_wrist"]),
    (KP["left_shoulder"], KP["left_hip"]),
    (KP["right_shoulder"], KP["right_hip"]),
    (KP["left_hip"], KP["right_hip"]),
    (KP["left_hip"], KP["left_knee"]),
    (KP["left_knee"], KP["left_ankle"]),
    (KP["right_hip"], KP["right_knee"]),
    (KP["right_knee"], KP["right_ankle"]),
    (KP["nose"], KP["left_shoulder"]),
    (KP["nose"], KP["right_shoulder"]),
]
