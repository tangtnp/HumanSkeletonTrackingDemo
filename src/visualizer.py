"""Draws the overlay: skeleton, detection boxes, motion trail and info cards."""

import cv2

from src.analyzer import MIN_CONF
from src.keypoints import KP, SKELETON

FACING_ARROW = {
    "Right": (1, 0),
    "Left": (-1, 0),
    "Front": (0, 1),
    "Back": (0, -1),
}

PALETTE = [
    (255, 196, 0),
    (0, 200, 255),
    (128, 255, 64),
    (255, 144, 160),
    (200, 120, 255),
    (96, 255, 208),
]

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE = (245, 245, 245)
SHADOW = (0, 0, 0)
CARD_BG = (24, 22, 28)


def _color(track_id):
    return PALETTE[track_id % len(PALETTE)]


def _text(img, text, org, scale, color, thick=1):
    """Draw text with a 1px shadow so it stays readable over any background."""
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale, SHADOW, thick, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _filled_rounded(img, x1, y1, x2, y2, color, r):
    """Filled rectangle with rounded corners (OpenCV has no built-in for this)."""
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def _card(img, x1, y1, x2, y2, accent, alpha=0.55, r=8):
    """Translucent rounded card with a coloured accent strip down the left."""
    overlay = img.copy()
    _filled_rounded(overlay, x1, y1, x2, y2, CARD_BG, r)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1 + r), (x1 + 3, y2 - r), accent, -1)


def draw_skeleton(img, person):
    color = _color(person.track_id)
    kps = person.keypoints

    for a, b in SKELETON:
        if kps[a][2] >= MIN_CONF and kps[b][2] >= MIN_CONF:
            pa = (int(kps[a][0]), int(kps[a][1]))
            pb = (int(kps[b][0]), int(kps[b][1]))
            
            cv2.line(img, pa, pb, SHADOW, 3, cv2.LINE_AA)
            cv2.line(img, pa, pb, color, 1, cv2.LINE_AA)

    for x, y, conf in kps:
        if conf >= MIN_CONF:
            p = (int(x), int(y))
            cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)
            cv2.circle(img, p, 1, WHITE, -1, cv2.LINE_AA)


def draw_box(img, person):
    """Corner brackets and an ID chip instead of a full rectangle."""
    color = _color(person.track_id)
    x1, y1, x2, y2 = person.bbox.astype(int)
    bracket = max(12, int(0.18 * (x2 - x1)))
    for cx, cy, sx, sy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + sx * bracket, cy), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + sy * bracket), color, 2, cv2.LINE_AA)

    label = f"ID {person.track_id}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
    _filled_rounded(img, x1, y1 - th - 10, x1 + tw + 14, y1, color, 6)
    _text(img, label, (x1 + 7, y1 - 7), 0.45, (15, 15, 15), 1)


def draw_trajectory(img, state, track_id):
    color = _color(track_id)
    pts = list(state.trail)
    n = len(pts)
    
    for i in range(1, n):
        f = i / n
        faded = tuple(int(ch * (0.3 + 0.7 * f)) for ch in color)
        cv2.line(img, pts[i - 1], pts[i], faded, 1 + int(f * 1.5), cv2.LINE_AA)
    if pts:
        cv2.circle(img, pts[-1], 3, color, -1, cv2.LINE_AA)


def draw_orientation(img, person, state):
    """Arrow at the shoulders showing which way the body faces."""
    if state.facing is None:
        return
    kps = person.keypoints
    left_sh, right_sh = kps[KP["left_shoulder"]], kps[KP["right_shoulder"]]
    if left_sh[2] < MIN_CONF or right_sh[2] < MIN_CONF:
        return

    color = _color(person.track_id)
    cx = int((left_sh[0] + right_sh[0]) / 2)
    cy = int((left_sh[1] + right_sh[1]) / 2)
    length = max(int(abs(left_sh[0] - right_sh[0]) * 0.6), 16)
    dx, dy = FACING_ARROW[state.facing]
    tip = (cx + dx * length, cy + dy * length)
    cv2.arrowedLine(img, (cx, cy), tip, color, 2, cv2.LINE_AA, tipLength=0.4)


def draw_info_panel(img, person, state, slot=0):
    """One card per person, stacked down the right-hand side."""
    color = _color(person.track_id)

    lines = [f"ID {person.track_id}"]
    if state.knee_angle is not None:
        lines.append(f"Knee  {state.knee_angle:.0f} deg")
    if state.elbow_angle is not None:
        lines.append(f"Elbow {state.elbow_angle:.0f} deg")
    lines.append(f"Speed {state.speed_ms:.1f} m/s")
    lines.append(f"Squats {state.squat_reps}")
    if state.facing is not None:
        lean = f" {state.lean_angle:+.0f}" if state.lean_angle is not None else ""
        lines.append(f"Facing {state.facing}{lean}")

    scale, line_h, pad = 0.4, 17, 8
    text_w = max(cv2.getTextSize(t, FONT, scale, 1)[0][0] for t in lines)
    w = text_w + pad * 2 + 6
    h = line_h * len(lines) + pad * 2
    x = img.shape[1] - w - 12
    y = 12 + slot * (h + 8)

    _card(img, x, y, x + w, y + h, color)
    for i, text in enumerate(lines):
        _text(img, text, (x + pad + 6, y + pad + 12 + i * line_h), scale, WHITE, 1)


def draw_hud(img, fps, n_people):
    """Small status pill in the top-left corner."""
    text = f"FPS {fps:.0f}   |   {n_people} people"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    _card(img, 10, 10, 10 + tw + 24, 10 + th + 16, (120, 120, 120), alpha=0.5)
    _text(img, text, (22, 10 + th + 4), 0.45, WHITE, 1)


def render(frame, analyzer, fps):
    """Draw the whole overlay onto the frame in place and return it."""
    img = frame.image
    for slot, person in enumerate(frame.persons):
        state = analyzer.states[person.track_id]
        draw_trajectory(img, state, person.track_id)
        draw_skeleton(img, person)
        draw_orientation(img, person, state)
        draw_box(img, person)
        draw_info_panel(img, person, state, slot)
    draw_hud(img, fps, len(frame.persons))
    return img
