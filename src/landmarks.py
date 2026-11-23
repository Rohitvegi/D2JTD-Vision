"""Stage 5 — hand landmarks on the RGB frame (MediaPipe Hand Landmarker) + anatomical checks.

Checks follow the failure modes reported by Qiao et al. (2025) for keypoints on FLIR ONE Pro images:
  1/2. finger order reversed (index<->little, middle<->ring)  -> fixed by distance re-ordering
  3.   index MCP (5) and little MCP (17) collapse onto one point -> frame rejected
  4.   keypoints outside the image / hand                       -> frame rejected
"""
from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# 21 MediaPipe landmarks: 0 wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16; little 17-20
FINGER_LM = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
             "ring": [13, 14, 15, 16], "little": [17, 18, 19, 20]}

# 14 joint regions (architecture Phase 3.4). (name, landmark index, finger, kind)
JOINTS = [
    ("thumb_MCP", 2, "thumb", "MCP"), ("thumb_IP", 3, "thumb", "IP"),
    ("index_MCP", 5, "index", "MCP"), ("index_PIP", 6, "index", "PIP"), ("index_DIP", 7, "index", "DIP"),
    ("middle_MCP", 9, "middle", "MCP"), ("middle_PIP", 10, "middle", "PIP"), ("middle_DIP", 11, "middle", "DIP"),
    ("ring_MCP", 13, "ring", "MCP"), ("ring_PIP", 14, "ring", "PIP"), ("ring_DIP", 15, "ring", "DIP"),
    ("little_MCP", 17, "little", "MCP"), ("little_PIP", 18, "little", "PIP"), ("little_DIP", 19, "little", "DIP"),
]
JOINT_NAMES = [j[0] for j in JOINTS]
# Qiao et al.'s 10 joints (MCP + PIP / thumb IP) for comparability
QIAO_JOINTS = [n for n, _, _, k in JOINTS if k in ("MCP", "PIP", "IP")]
DORSUM_LM = [0, 1, 2, 5, 9, 13, 17]
# For the radius rule: (MCP landmark, PIP landmark) per finger
RADIUS_PAIR = {"thumb": (2, 3), "index": (5, 6), "middle": (9, 10), "ring": (13, 14), "little": (17, 18)}


def make_landmarker(model_path: str | Path, video: bool, min_conf: float = 0.5):
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    return vision.HandLandmarker.create_from_options(opts)


def detect(landmarker, rgb: np.ndarray, ts_ms: int | None = None, with_z: bool = False):
    """Return (21x2 landmark pixel coords in the crop, handedness score) or (None, 0).
    with_z=True also returns MediaPipe's relative depth per landmark (pixels; smaller = closer to camera)."""
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = landmarker.detect_for_video(img, int(ts_ms)) if ts_ms is not None else landmarker.detect(img)
    if not res.hand_landmarks:
        return (None, 0.0, None) if with_z else (None, 0.0)
    h, w = rgb.shape[:2]
    pts = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]], dtype=np.float32)
    score = float(res.handedness[0][0].score) if res.handedness else 0.0
    if with_z:
        return pts, score, np.array([p.z * w for p in res.hand_landmarks[0]], dtype=np.float32)
    return pts, score


BONES = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
         (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]


def _bend_deg(p3, a, b, c) -> float:
    u, v = p3[b] - p3[a], p3[c] - p3[b]
    n = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(u @ v / n, -1, 1)))) if n > 0 else 0.0


def joint_visibility(pts: np.ndarray, z: np.ndarray | None = None, fold_deg: float = 100.0,
                     cover_frac: float = 0.07, z_margin_frac: float = 0.03) -> dict:
    """Which of the 14 joints are actually visible in the RGB image (camera looks at the back of the hand).

    Returns {joint: ""} for visible joints and {joint: reason} for hidden ones. Rules:
      1. under the hand : a PIP/DIP lies inside the back-of-hand outline (wrist, thumb base, 4 knuckles)
                          -> in this view it is folded underneath the hand (fist);
      2. covered        : another finger's bone crosses the joint's position and is closer to the camera
                          (MediaPipe relative depth; only when z is available);
      3. finger folded  : the middle joint (PIP) is bent more than `fold_deg` -> the end joint (DIP) is
                          tucked under.
    MediaPipe still outputs positions for hidden joints (it guesses them); those guesses must not be
    used as temperature measurements.
    """
    pts = np.asarray(pts, float)
    size = float(np.linalg.norm(pts[0] - pts[9])) or 1.0
    out = {name: "" for name, *_ in JOINTS}
    poly = pts[DORSUM_LM]
    c = poly.mean(0)
    poly = (c + 0.92 * (poly - c)).astype(np.float32)                    # slight shrink: edges are uncertain
    p3 = np.column_stack([pts, z]) if z is not None else np.column_stack([pts, np.zeros(len(pts))])
    for name, lm, finger, kind in JOINTS:
        if kind == "MCP":
            continue                                                     # knuckles: on the visible back of hand
        if finger != "thumb" and cv2.pointPolygonTest(poly, (float(pts[lm, 0]), float(pts[lm, 1])), False) > 0:
            out[name] = "folded under the hand"
            continue
        if kind == "DIP":
            pip = lm - 1
            if _bend_deg(p3, pip - 1, pip, lm) > fold_deg:
                out[name] = "finger folded (end joint tucked under)"
                continue
        if z is not None:
            for a, b in BONES:
                if lm in (a, b):
                    continue
                ab = pts[b] - pts[a]
                t = float(np.clip((pts[lm] - pts[a]) @ ab / max(ab @ ab, 1e-6), 0, 1))
                if np.linalg.norm(pts[a] + t * ab - pts[lm]) < cover_frac * size:
                    zc = z[a] + t * (z[b] - z[a])
                    if zc < z[lm] - z_margin_frac * size:
                        owner = next(f for f, ch in FINGER_LM.items() if a in ch)
                        out[name] = ("finger folded (covered by its own finger)" if owner == finger
                                     else f"covered by the {owner} finger")
                        break
    return out


def openness(p: np.ndarray) -> float:
    """Mean over the 4 fingers of |wrist-tip| / |wrist-MCP|: ~1.9+ open hand, ~1.0-1.3 fist."""
    return float(np.mean([_d(p, 0, t) / max(_d(p, 0, m), 1e-6) for m, t in ((5, 8), (9, 12), (13, 16), (17, 20))]))


def _d(p, a, b) -> float:
    return float(np.linalg.norm(p[a] - p[b]))


def check_and_fix(pts: np.ndarray, hw: tuple[int, int], margin: float = 0.0) -> tuple[np.ndarray, dict]:
    """Apply the anatomical checks. Returns (possibly re-ordered points, flags)."""
    p = pts.copy()
    flags = dict(reordered_index_little=False, reordered_middle_ring=False,
                 mcp_collapse=False, out_of_image=False)
    # Case 3: index/little MCP collapse (Qiao eq. 2)
    thumb_len = sum(_d(p, i, i + 1) for i in range(4))
    if _d(p, 5, 17) <= 0.25 * thumb_len:
        flags["mcp_collapse"] = True
    # Case 1/2: thumb is reliable -> index must be nearer the thumb than the little finger is
    if _d(p, 2, 5) > _d(p, 2, 17):
        p[[5, 6, 7, 8, 17, 18, 19, 20]] = p[[17, 18, 19, 20, 5, 6, 7, 8]]
        flags["reordered_index_little"] = True
    # then middle must be nearer the index than the ring is
    if _d(p, 5, 9) > _d(p, 5, 13):
        p[[9, 10, 11, 12, 13, 14, 15, 16]] = p[[13, 14, 15, 16, 9, 10, 11, 12]]
        flags["reordered_middle_ring"] = True
    # Case 4 (geometric part): joint centres must lie inside the image
    h, w = hw
    jidx = [j[1] for j in JOINTS] + [0]
    q = p[jidx]
    if (q[:, 0] < -margin).any() or (q[:, 1] < -margin).any() or (q[:, 0] > w + margin).any() or (q[:, 1] > h + margin).any():
        flags["out_of_image"] = True
    flags["ok"] = not (flags["mcp_collapse"] or flags["out_of_image"])
    return p, flags
