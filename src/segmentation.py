"""Stage 6 — hand masks.

Thermal: the hand is warmer than the background -> Otsu threshold on the radiometric frame,
largest connected component, holes filled.
RGB: GrabCut seeded by the landmarks (palm polygon + finger bones = sure hand; far from the
landmark hull = sure background). Skin colour is not assumed, so it works for any skin tone.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.landmarks import DORSUM_LM, FINGER_LM


def _largest_component(m: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(m, dtype=bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def _fill_holes(m: np.ndarray) -> np.ndarray:
    m8 = m.astype(np.uint8) * 255
    h, w = m8.shape
    flood = m8.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return (m8 | cv2.bitwise_not(flood)) > 0


def thermal_mask(T: np.ndarray, low_frac: float = 0.4) -> tuple[np.ndarray, float]:
    """Boolean hand mask on the .npy frame and the low threshold used (deg C).

    Hysteresis: the Otsu region (warm hand) is the seed; it grows into connected pixels above
    background + low_frac * (Otsu threshold - background). This keeps cold fingers (close to
    room temperature, e.g. subject S17) that a single Otsu threshold would cut off.
    """
    t = cv2.GaussianBlur(T, (5, 5), 0)
    lo, hi = float(np.percentile(t, 1)), float(np.percentile(t, 99.5))
    t8 = np.clip((t - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    thr8, m = cv2.threshold(t8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = lo + thr8 / 255.0 * (hi - lo)
    seed = _largest_component(m > 0)
    bg = float(np.median(t[~seed])) if (~seed).any() else lo
    low = bg + low_frac * (otsu - bg)
    n, lab = cv2.connectedComponents((t > low).astype(np.uint8), connectivity=8)
    keep = np.unique(lab[seed])
    keep = keep[keep > 0]
    m = np.isin(lab, keep).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    m = _fill_holes(_largest_component(m > 0))
    return m, low


def hand_size(pts: np.ndarray) -> float:
    """Wrist to middle-finger MCP distance, a stable hand-size scale."""
    return float(np.linalg.norm(pts[0] - pts[9]))


def rgb_mask(rgb: np.ndarray, pts: np.ndarray, iters: int = 3) -> np.ndarray:
    """GrabCut hand mask on the RGB crop, seeded by the 21 landmarks."""
    h, w = rgb.shape[:2]
    s = hand_size(pts)
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    hull = cv2.convexHull(pts.astype(np.int32))
    near = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(near, hull, 1)
    near = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.6 * s) | 1,) * 2))
    gc[near == 0] = cv2.GC_BGD
    hull_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    gc[(hull_mask == 1)] = cv2.GC_PR_FGD
    sure = np.zeros((h, w), np.uint8)
    cv2.fillPoly(sure, [pts[DORSUM_LM].astype(np.int32)], 1)
    sure = cv2.erode(sure, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.15 * s) | 1,) * 2))
    th = max(2, int(0.06 * s))
    for lms in FINGER_LM.values():           # bones up to the last joint (tips often overshoot)
        chain = [0 if lms[0] != 1 else 0] + lms[:-1]
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(sure, tuple(map(int, pts[a])), tuple(map(int, pts[b])), 1, th)
    gc[sure == 1] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return sure.astype(bool)
    m = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    return _fill_holes(_largest_component(m))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else np.nan
