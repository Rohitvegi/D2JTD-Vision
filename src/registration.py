"""Stage 7 — RGB -> thermal registration.

No calibration target was recorded, so stereo calibration is impossible for this dataset
(Stage 2 review, section 3.5). Instead, per recording, a similarity transform (uniform scale s
about the frame centre + translation t) is fitted that best overlays the RGB hand mask on the
thermal hand mask:

    p_npy = s * (k * p_crop - c) + c + t,   k = 4/3 (crop -> .npy size), c = .npy centre

The FLIR ONE Pro's two lenses are rigidly mounted, so one transform per recording (median over
sampled frames) is used. Accuracy is checked on frames NOT used for fitting (held-out Dice), and
against hand-clicked joints when data/annotations/thermal_joints.csv exists (src/annotate.py).
"""
from __future__ import annotations

import cv2
import numpy as np

from src.io_utils import NPY_HW
from src.segmentation import dice

C = np.array([NPY_HW[1] / 2.0, NPY_HW[0] / 2.0], dtype=np.float32)   # (x, y) centre of .npy frame


def matrix(k: float, s: float, tx: float, ty: float, angle_deg: float = 0.0) -> np.ndarray:
    """2x3 affine matrix crop -> .npy:  p_npy = s * Rot(angle) (k p - c) + c + t   (angle 0 = v3 model)."""
    a = np.deg2rad(angle_deg)
    Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    A = s * k * Rm
    b = C - s * Rm @ C + np.array([tx, ty])
    return np.c_[A, b].astype(np.float32)


def decompose(M: np.ndarray, k: float) -> tuple[float, float, float, float]:
    """Inverse of matrix(): (s, tx, ty, angle_deg) of a similarity matrix."""
    A = np.asarray(M[:, :2], float)
    s = float(np.sqrt(abs(np.linalg.det(A)))) / k
    a = float(np.arctan2(A[1, 0], A[0, 0]))
    Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    t = np.asarray(M[:, 2], float) - (C - s * Rm @ C)
    return s, float(t[0]), float(t[1]), float(np.rad2deg(a))


def map_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


def warp_mask(mask_crop: np.ndarray, M: np.ndarray, hw=NPY_HW) -> np.ndarray:
    out = cv2.warpAffine(mask_crop.astype(np.uint8), M, (hw[1], hw[0]), flags=cv2.INTER_NEAREST)
    return out > 0


def fit_frame(rgb_m: np.ndarray, th_m: np.ndarray, k: float, scales: np.ndarray, ds: int = 4) -> dict:
    """Best (s, tx, ty) for one frame: scale grid x phase-correlation translation, Dice-scored."""
    hw_s = (NPY_HW[0] // ds, NPY_HW[1] // ds)
    th_s = cv2.resize(th_m.astype(np.float32), (hw_s[1], hw_s[0]), interpolation=cv2.INTER_AREA)
    win = cv2.createHanningWindow((hw_s[1], hw_s[0]), cv2.CV_32F)
    best = dict(s=np.nan, tx=np.nan, ty=np.nan, dice=-1.0)
    for s in scales:
        M = matrix(k / ds, s, 0, 0)
        M[:, 2] = (1 - s) * C / ds                                    # centre in downscaled frame
        w = cv2.warpAffine(rgb_m.astype(np.float32), M, (hw_s[1], hw_s[0]), flags=cv2.INTER_LINEAR)
        (dx, dy), _ = cv2.phaseCorrelate(w * win, th_s * win)
        for sx, sy in ((dx, dy), (-dx, -dy)):
            Ms = M.copy()
            Ms[:, 2] += (sx, sy)
            ws = cv2.warpAffine(rgb_m.astype(np.float32), Ms, (hw_s[1], hw_s[0])) > 0.5
            d = dice(ws, th_s > 0.5)
            if d > best["dice"]:
                best = dict(s=float(s), tx=float(sx * ds), ty=float(sy * ds), dice=d)
    # local refinement at full resolution (+-4 px, step 2)
    s, tx0, ty0 = best["s"], best["tx"], best["ty"]
    for ddx in range(-4, 5, 2):
        for ddy in range(-4, 5, 2):
            d = dice(warp_mask(rgb_m, matrix(k, s, tx0 + ddx, ty0 + ddy)), th_m)
            if d > best["dice"]:
                best.update(tx=tx0 + ddx, ty=ty0 + ddy, dice=d)
    return best


def fit_recording(pairs: list[tuple[np.ndarray, np.ndarray]], k: float, scales: np.ndarray,
                  rig_default: tuple[float, float, float] | None = None) -> dict:
    """pairs = [(rgb_mask_crop, thermal_mask_npy), ...] in time order.

    Fit on even-numbered samples, take the median transform, evaluate Dice on odd (held-out) ones.
    If rig_default (s, tx, ty) scores better on the held-out frames, it is used instead and
    flagged (the per-recording fit failed, e.g. masks disagree in most sampled frames).
    """
    fits = [fit_frame(r, t, k, scales) for r, t in pairs]
    fit_idx = list(range(0, len(pairs), 2))
    test_idx = list(range(1, len(pairs), 2)) or fit_idx
    F = np.array([[fits[i]["s"], fits[i]["tx"], fits[i]["ty"]] for i in fit_idx])
    s, tx, ty = np.median(F, axis=0)

    def held_dice(s_, tx_, ty_):
        M_ = matrix(k, s_, tx_, ty_)
        return [dice(warp_mask(pairs[i][0], M_), pairs[i][1]) for i in test_idx]
    held = held_dice(s, tx, ty)
    used_default = False
    if rig_default is not None:
        held_d = held_dice(*rig_default)
        if np.median(held_d) > np.median(held):
            s, tx, ty = rig_default
            held, used_default = held_d, True
    allF = np.array([[f["s"], f["tx"], f["ty"]] for f in fits])
    return dict(scale=float(s), tx=float(tx), ty=float(ty), used_rig_default=used_default,
                dice_fit_median=float(np.median([f["dice"] for f in fits])),
                dice_heldout_median=float(np.median(held)), dice_heldout_min=float(np.min(held)),
                scale_sd=float(allF[:, 0].std()), tx_sd=float(allF[:, 1].std()), ty_sd=float(allF[:, 2].std()),
                n_pairs=len(pairs))
