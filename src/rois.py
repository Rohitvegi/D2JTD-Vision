"""Stage 8 — joint and dorsum regions of interest on the radiometric frame, with quality flags.

ROI radius (Qiao et al. 2025): r_finger = beta_finger * |MCP - PIP| of that finger (thumb: MCP-IP),
so ROIs scale with hand size and camera distance. Temperatures are taken only from pixels inside
the thermal hand mask; hot pixels are never removed (they may be the signal).
"""
from __future__ import annotations

import cv2
import numpy as np

from src.landmarks import DORSUM_LM, JOINTS, RADIUS_PAIR

COLD_BAND_C = 1.5        # deg C below the ROI median counted as a possible accessory (ring) band
COLD_FLAG_FRAC = 0.15    # ROI flagged "possibly contaminated" when more than this share is in the cold band


def radii(pts_npy: np.ndarray, beta: dict) -> dict:
    return {f: beta[f] * float(np.linalg.norm(pts_npy[a] - pts_npy[b])) for f, (a, b) in RADIUS_PAIR.items()}


def edge_distance(hand: np.ndarray, c: np.ndarray, axis: np.ndarray, max_len: int = 120) -> float:
    """Distance from c to the NEARER hand-mask edge, measured perpendicular to the finger axis.

    Walks outwards pixel by pixel on both sides until it leaves the mask. Using the nearer side
    keeps the ROI on the finger even when the joint centre is slightly off the finger's midline.
    """
    h, w = hand.shape
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return float(max_len)
    perp = np.array([-axis[1], axis[0]]) / n
    dists = []
    for sgn in (1, -1):
        d = 0
        for step in range(1, max_len + 1):
            x, y = np.rint(c + sgn * step * perp).astype(int)
            if not (0 <= x < w and 0 <= y < h) or not hand[y, x]:
                break
            d = step
        dists.append(d)
    return float(min(dists))


def joint_radii(pts_npy: np.ndarray, hand: np.ndarray, beta: dict, mode: str = "qiao",
                width_frac: float = 0.9, min_r: float = 6.0) -> dict:
    """Radius per joint.

    qiao:          r = beta_finger * |MCP - PIP|  (Qiao et al. 2025)
    finger_width:  r = min(qiao r, width_frac * distance to the nearer finger edge), floored at
                   min_r px (~1.5 native sensor pixels) so a ROI never collapses to nothing.
    """
    rr = radii(pts_npy, beta)
    out = {}
    for name, lm, finger, _ in JOINTS:
        r = rr[finger]
        if mode == "finger_width":
            axis = pts_npy[lm + 1] - pts_npy[lm]          # towards the next distal landmark
            r = max(min_r, min(r, width_frac * edge_distance(hand, pts_npy[lm], axis)))
        out[name] = float(r)
    return out


def _disc(hw, c, r) -> tuple[slice, slice, np.ndarray]:
    """Boolean disc inside its bounding box (fast: avoids full-frame masks)."""
    h, w = hw
    x, y = float(c[0]), float(c[1])
    r = max(r, 1.0)
    x0, x1 = max(0, int(np.floor(x - r))), min(w, int(np.ceil(x + r)) + 1)
    y0, y1 = max(0, int(np.floor(y - r))), min(h, int(np.ceil(y + r)) + 1)
    if x1 <= x0 or y1 <= y0:
        return slice(0, 0), slice(0, 0), np.zeros((0, 0), bool)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return slice(y0, y1), slice(x0, x1), (xx - x) ** 2 + (yy - y) ** 2 <= r * r


def extract(T: np.ndarray, hand: np.ndarray, pts_npy: np.ndarray, beta: dict, usable_frac: float,
            core: np.ndarray | None = None, min_px: int = 32, radius_mode: str = "qiao",
            width_frac: float = 0.9) -> dict:
    """Per-joint and dorsum statistics for one frame.

    hand: thermal hand mask (decides how much of each ROI lies on the hand).
    core: hand mask shrunk by ~1 sensor pixel; temperatures are sampled only here, so the
          cooler partial-volume pixels on the hand's outline do not bias joints downwards.
    """
    core = hand if core is None else core
    hw = T.shape
    rj = joint_radii(pts_npy, hand, beta, radius_mode, width_frac)
    out = {}
    excl = np.zeros(hw, bool)
    full_area = 0
    for name, lm, finger, _ in JOINTS:
        r = rj[name]
        out[f"{name}_r"] = r
        ys, xs, disc = _disc(hw, pts_npy[lm], r)
        area_theory = np.pi * r * r
        full_area += area_theory
        if disc.size == 0:
            out.update({f"{name}_T": np.nan, f"{name}_sd": np.nan, f"{name}_in": 0.0, f"{name}_npx": 0,
                        f"{name}_mean": np.nan, f"{name}_p90": np.nan, f"{name}_valid": 0.0, f"{name}_cold": np.nan})
            continue
        on = hand[ys, xs] & disc
        frac = on.sum() / max(area_theory, 1.0)          # off-image part counts as off-hand
        vals = T[ys, xs][core[ys, xs] & disc]
        vals = vals[np.isfinite(vals)]                   # invalid pixels never enter a statistic
        usable = frac >= usable_frac and vals.size >= min_px
        med = float(np.median(vals)) if vals.size else np.nan
        out[f"{name}_T"] = med if usable else np.nan
        out[f"{name}_sd"] = float(vals.std()) if usable else np.nan      # HDI-like spread
        out[f"{name}_mean"] = float(vals.mean()) if usable else np.nan
        out[f"{name}_p90"] = float(np.percentile(vals, 90)) if usable else np.nan
        out[f"{name}_in"] = float(min(frac, 1.0))
        out[f"{name}_npx"] = int(vals.size)
        out[f"{name}_valid"] = float(min(vals.size / max(area_theory, 1.0), 1.0))   # share of ROI usable
        # contamination check: a band clearly COLDER than the ROI median (metal ring: low emissivity).
        # Hot pixels are never removed or flagged — they may be the signal.
        out[f"{name}_cold"] = float(np.mean(vals < med - COLD_BAND_C)) if vals.size else np.nan
        excl[ys, xs] |= disc
    poly = np.zeros(hw, np.uint8)
    cv2.fillPoly(poly, [pts_npy[DORSUM_LM].astype(np.int32)], 1)
    d = (poly > 0) & core & ~excl & np.isfinite(T)
    out["dorsum_T"] = float(np.median(T[d])) if d.sum() >= 50 else np.nan
    out["dorsum_npx"] = int(d.sum())
    hv = T[core & np.isfinite(T)]
    out["hand_T"] = float(np.median(hv)) if hv.size >= 50 else np.nan     # whole visible hand
    out["invalid_px"] = int((~np.isfinite(T)).sum())
    out["roi_radius_mean"] = float(np.mean(list(rj.values())))
    return out
