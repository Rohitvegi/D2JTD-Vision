"""
Stage 1 — Dataset audit for the FLIR One Pro paired RGB / thermal hand dataset.

What it does (one pass over the dataset, never loading a whole .npy into RAM):
  1.  Finds every subject, hand and task folder and checks all 3 files exist.
  2.  Reads RGB and thermal MP4 metadata (resolution, FPS, frame count, duration).
  3.  Optionally scans every video frame's timestamp to get the TRUE frame count
      and frame-interval statistics (catches variable frame rate and dropped frames).
  4.  Opens each radiometric .npy with mmap (header, dtype, shape, axis order,
      truncation check) and computes temperature statistics on sampled frames.
  5.  Checks NaN / inf / out-of-range values and frozen (repeated) frames.
  6.  Tests whether thermal MP4 frame i matches .npy frame i (correlation at
      lags -k..+k) — i.e. whether the streams are index-aligned.
  7.  Detects duplicate recordings (identical files under different names).
  8.  Saves sample figures (RGB | thermal MP4 | radiometric | overlay) per recording.
  9.  Writes audit CSV + Markdown report.

Run from the project root:
    python -m src.data_audit                 # uses config.yaml
    python -m src.data_audit --quick         # skip the per-frame timestamp scan
    python -m src.data_audit --limit 2       # only the first 2 subjects (smoke test)
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")                      # no window needed; just save PNGs
import matplotlib.pyplot as plt

LOG = logging.getLogger("audit")

TASKS = ("static", "open_close")
HANDS = ("left", "right")
STREAMS = {
    "rgb": "{task}_RGB.mp4",
    "thermal_mp4": "{task}_Thermal.mp4",
    "npy": "{task}_Thermal_Radiometric_Sequence.npy",
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    """Read config.yaml. Deployment overrides: RGBT_DATASET_ROOT (original FLIROne dataset folder)."""
    import os
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("RGBT_DATASET_ROOT"):
        cfg["dataset_root"] = os.environ["RGBT_DATASET_ROOT"]
    return cfg


def subject_sort_key(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 10**6


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
@dataclass
class Recording:
    subject: str
    hand: str
    task: str
    folder: Path
    paths: dict = field(default_factory=dict)      # stream -> Path
    missing: list = field(default_factory=list)

    @property
    def rec_id(self) -> str:
        return f"{self.subject}_{self.hand}_{self.task}"


def discover(root: Path, limit: int | None = None) -> tuple[list[Recording], list[str]]:
    """Return expected recordings (present or not) and a list of unexpected items."""
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    subjects = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: subject_sort_key(p.name))
    if limit:
        subjects = subjects[:limit]
    recs, unexpected = [], []
    for s in subjects:
        for hand in HANDS:
            hdir = s / f"{hand} hand"
            if not hdir.is_dir():
                for task in TASKS:
                    r = Recording(s.name, hand, task, hdir, missing=list(STREAMS))
                    recs.append(r)
                continue
            expected_names = set()
            for task in TASKS:
                r = Recording(s.name, hand, task, hdir)
                for stream, pattern in STREAMS.items():
                    fname = pattern.format(task=task)
                    expected_names.add(fname)
                    p = hdir / fname
                    if p.is_file():
                        r.paths[stream] = p
                    else:
                        r.missing.append(stream)
                recs.append(r)
            for extra in hdir.iterdir():
                if extra.name not in expected_names:
                    unexpected.append(str(extra.relative_to(root)))
        for extra in s.iterdir():
            if extra.is_dir() and extra.name not in {f"{h} hand" for h in HANDS}:
                unexpected.append(str(extra.relative_to(root)))
    return recs, unexpected


# --------------------------------------------------------------------------- #
# Video metadata
# --------------------------------------------------------------------------- #
def video_metadata(path: Path, scan_timestamps: bool) -> dict:
    out = {"ok": False}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        out["error"] = "cannot open"
        return out
    out.update(
        ok=True,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps_header=float(cap.get(cv2.CAP_PROP_FPS)),
        frames_header=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if scan_timestamps:
        ts = []
        while cap.grab():                       # grab = advance without full colour conversion
            ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        ts = np.asarray(ts, dtype=np.float64)
        out["frames_decoded"] = int(ts.size)
        if ts.size > 1:
            d = np.diff(ts)
            out["duration_s"] = float((ts[-1] - ts[0]) / 1000.0 + np.median(d) / 1000.0)
            out["fps_measured"] = float(1000.0 / np.median(d)) if np.median(d) > 0 else np.nan
            out["dt_ms_min"] = float(d.min())
            out["dt_ms_max"] = float(d.max())
            out["dt_ms_std"] = float(d.std())
            out["non_increasing_ts"] = int((d <= 0).sum())
            # a gap > 2x median interval suggests dropped frames
            out["gaps_gt_2x"] = int((d > 2 * np.median(d)).sum())
            out["_timestamps"] = ts
    else:
        fh = out["frames_header"]
        out["duration_s"] = fh / out["fps_header"] if out["fps_header"] > 0 else np.nan
    cap.release()
    return out


def read_video_frames(path: Path, indices) -> dict[int, np.ndarray]:
    """Return {index: frame} by decoding the video IN ORDER.

    Never use cap.set(CAP_PROP_POS_FRAMES, i) on these files: they have a variable
    frame rate, so OpenCV's seek lands up to ~15 frames away from frame i.
    """
    wanted = sorted({int(i) for i in indices if int(i) >= 0})
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    cap = cv2.VideoCapture(str(path))
    i = 0
    while i <= wanted[-1]:
        if i in wanted:
            ok, frame = cap.read()
            if not ok:
                break
            out[i] = frame
        elif not cap.grab():                 # skip without decoding to an image
            break
        i += 1
    cap.release()
    return out


def read_video_frame(path: Path, index: int) -> np.ndarray | None:
    return read_video_frames(path, [index]).get(int(index))


def seek_offset(path: Path, index: int, search: int = 30) -> float:
    """How far OpenCV's frame seek lands from the true frame `index` (0 = accurate)."""
    ref = read_video_frames(path, range(max(0, index - search), index + search + 1))
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, f = cap.read()
    cap.release()
    if not ok or index not in ref:
        return np.nan
    if np.array_equal(f, ref[index]):
        return 0.0
    diffs = {j: float(np.abs(f.astype(np.int16) - g.astype(np.int16)).mean()) for j, g in ref.items()}
    return float(min(diffs, key=diffs.get) - index)


# --------------------------------------------------------------------------- #
# Radiometric .npy
# --------------------------------------------------------------------------- #
def npy_header(path: Path) -> dict:
    """Read the .npy header without touching the data, and check for truncation."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        header_len = f.tell()
    expected = header_len + int(np.prod(shape)) * dtype.itemsize
    actual = path.stat().st_size
    return dict(
        npy_version=f"{version[0]}.{version[1]}",
        shape=tuple(int(x) for x in shape),
        dtype=str(dtype),
        fortran_order=bool(fortran),
        header_bytes=header_len,
        size_expected=expected,
        size_actual=actual,
        truncated=actual < expected,
        trailing_bytes=max(0, actual - expected),
    )


def frame_axes(shape: tuple) -> tuple[int, int, int, str]:
    """Return (n_frames, H, W, note). Images are conventionally (H, W) = (480, 640)."""
    if len(shape) != 3:
        return (shape[0] if shape else 0), 0, 0, f"unexpected ndim={len(shape)}"
    n, a, b = shape
    if (a, b) == (480, 640):
        return n, a, b, "(N, H=480, W=640) — standard image order"
    if (a, b) == (640, 480):
        return n, a, b, "(N, H=640, W=480) — portrait; MP4s show it letterboxed (resize into box, do not transpose)"
    return n, a, b, f"(N, {a}, {b}) — non-standard size"


def npy_stats(seq: np.ndarray, n_sample: int, valid_range: tuple[float, float],
              rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    n = seq.shape[0]
    idx = np.unique(np.linspace(0, n - 1, num=min(n_sample, n)).round().astype(int))
    mins, maxs, means, p01, p99 = [], [], [], [], []
    nan_px = inf_px = out_px = total_px = 0
    frozen = 0
    pixel_pool = []
    for i in idx:
        fr = np.asarray(seq[i], dtype=np.float32)
        total_px += fr.size
        nan_px += int(np.isnan(fr).sum())
        inf_px += int(np.isinf(fr).sum())
        finite = fr[np.isfinite(fr)]
        if finite.size == 0:
            continue
        out_px += int(((finite < valid_range[0]) | (finite > valid_range[1])).sum())
        mins.append(finite.min()); maxs.append(finite.max()); means.append(finite.mean())
        p01.append(np.percentile(finite, 1)); p99.append(np.percentile(finite, 99))
        pixel_pool.append(rng.choice(finite, size=min(2000, finite.size), replace=False))
        if i + 1 < n:                                   # frozen-frame check on the next frame
            nxt = np.asarray(seq[i + 1], dtype=np.float32)
            if np.array_equal(fr, nxt, equal_nan=True):
                frozen += 1
    stats = dict(
        frames_sampled=len(idx),
        t_min=float(np.min(mins)) if mins else np.nan,
        t_max=float(np.max(maxs)) if maxs else np.nan,
        t_mean=float(np.mean(means)) if means else np.nan,
        t_p01=float(np.median(p01)) if p01 else np.nan,
        t_p99=float(np.median(p99)) if p99 else np.nan,
        frame_mean_drift=float(means[-1] - means[0]) if len(means) > 1 else np.nan,
        nan_frac=nan_px / max(total_px, 1),
        inf_frac=inf_px / max(total_px, 1),
        out_of_range_frac=out_px / max(total_px, 1),
        frozen_pairs=frozen,
    )
    pool = np.concatenate(pixel_pool) if pixel_pool else np.empty(0, np.float32)
    return stats, pool


def content_box(bgr: np.ndarray, thr: int = 8) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1) of the non-black area of a letterboxed video frame.

    The FLIR One exports put a portrait 3:4 image in the middle of a 480x640 frame
    with black bars left and right; this finds that image area.
    """
    g = bgr.max(axis=2) if bgr.ndim == 3 else bgr
    rows = np.where(g.max(axis=1) > thr)[0]
    cols = np.where(g.max(axis=0) > thr)[0]
    if rows.size == 0 or cols.size == 0:
        return 0, g.shape[0], 0, g.shape[1]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def fit_to_box(T: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Resize a radiometric frame (H, W) to the size of a video content box. No transpose."""
    y0, y1, x0, x1 = box
    return cv2.resize(np.asarray(T, dtype=np.float32), (x1 - x0, y1 - y0),
                      interpolation=cv2.INTER_AREA)


def place_in_frame(T: np.ndarray, hw: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    """Radiometric frame resized into the video's content box; NaN in the black bars."""
    canvas = np.full(hw, np.nan, dtype=np.float32)
    y0, y1, x0, x1 = box
    canvas[y0:y1, x0:x1] = fit_to_box(T, box)
    return canvas


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def mp4_npy_correspondence(thermal_mp4: Path, seq: np.ndarray, n_test: int, max_lag: int,
                           min_motion_corr: float = 0.3, min_margin: float = 0.1) -> dict:
    """Check that thermal-MP4 frame i shows the same image as .npy frame i.

    1. Spatial match: the .npy frame resized into the MP4's content box should correlate
       strongly with the MP4 grey level (corr_lag0, median over n_test frames; expect > 0.9).
    2. Index alignment: neighbouring frames are nearly identical, so single-frame
       correlation cannot tell lag 0 from lag 1. Instead build a "motion energy" signal for
       each stream (mean absolute change between consecutive frames, on 120x90 thumbnails)
       over the WHOLE recording and find the shift -k..+k that lines the two signals up.
       If the hand barely moves the result is reported as undetermined, not as a lag.
    """
    n = seq.shape[0]
    empty = dict(corr_lag0=np.nan, best_lag=np.nan, best_lag_score=np.nan,
                 lag_margin=np.nan, lag_status="not tested", thermal_box="")
    cap = cv2.VideoCapture(str(thermal_mp4))
    if not cap.isOpened() or n < 2 * max_lag + 3:
        cap.release()
        return empty
    test_set = set(np.linspace(0, n - 1, num=n_test).round().astype(int).tolist())
    thumb = (90, 120)                                    # (W, H) for cv2.resize
    box, spatial, mp4_thumbs = None, [], []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if box is None:
            box = content_box(bgr)
        y0, y1, x0, x1 = box
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1].astype(np.float32)
        mp4_thumbs.append(cv2.resize(g, thumb, interpolation=cv2.INTER_AREA))
        if i in test_set and i < n:
            spatial.append(_corr(fit_to_box(seq[i], box), g))
        i += 1
    cap.release()
    if box is None:
        return empty
    out = dict(empty, corr_lag0=float(np.nanmedian(spatial)) if spatial else np.nan,
               thermal_box=f"y{box[0]}-{box[1]} x{box[2]}-{box[3]}")
    npy_thumbs = np.stack([cv2.resize(np.asarray(seq[j], dtype=np.float32), thumb,
                                      interpolation=cv2.INTER_AREA) for j in range(n)])
    # every frame: does MP4 frame i show the same image as .npy frame i?
    per_frame = np.array([_corr(mp4_thumbs[j], npy_thumbs[j])
                          for j in range(min(len(mp4_thumbs), n))])
    if per_frame.size and np.isfinite(per_frame).any():
        bad = np.where(per_frame < 0.95)[0]
        out.update(frame_corr_min=float(np.nanmin(per_frame)),
                   frame_corr_min_at=int(np.nanargmin(per_frame)),
                   frames_mismatched=int(bad.size),
                   frames_mismatched_first=(",".join(map(str, bad[:10])) if bad.size else ""))
    mg = np.abs(np.diff(np.stack(mp4_thumbs), axis=0)).mean(axis=(1, 2))
    mt = np.abs(np.diff(npy_thumbs, axis=0)).mean(axis=(1, 2))
    k = min(len(mg), len(mt))
    mg, mt = mg[:k], mt[:k]
    scores = {}
    for lag in range(-max_lag, max_lag + 1):         # lag L: mp4 change[i+L] ~ npy change[i]
        a = mg[max(0, lag):k + min(0, lag)]
        b = mt[max(0, -lag):k - max(0, lag)]
        c = _corr(a, b)
        if np.isfinite(c):
            scores[lag] = c
    if not scores:
        return dict(out, lag_status="undetermined (no frame change)")
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    margin = best_score - ranked[1][1] if len(ranked) > 1 else np.nan
    out.update(best_lag_score=best_score, lag_margin=margin)
    if best_score < min_motion_corr or not (margin >= min_margin):
        out["lag_status"] = "undetermined (too little motion)"
    else:
        out.update(best_lag=best, lag_status="aligned" if best == 0 else f"offset {best:+d}")
    return out


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #
def file_fingerprint(path: Path, nbytes: int) -> str:
    """Hash of size + first and last nbytes. Fast and reliable for large files."""
    h = hashlib.md5()
    size = path.stat().st_size
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
        if size > nbytes:
            f.seek(max(0, size - nbytes))
            h.update(f.read(nbytes))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def thermal_hand_contour(T: np.ndarray) -> np.ndarray | None:
    finite = np.where(np.isfinite(T), T, np.nanmin(T))
    t8 = cv2.normalize(finite, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t8 = cv2.medianBlur(t8, 5)
    _, m = cv2.threshold(t8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def save_sample_figure(rec: Recording, seq: np.ndarray, n_frames: int, out_png: Path) -> None:
    idx = [0, n_frames // 2, n_frames - 1]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    rgb_frames = read_video_frames(rec.paths["rgb"], idx) if "rgb" in rec.paths else {}
    thm_frames = read_video_frames(rec.paths["thermal_mp4"], idx) if "thermal_mp4" in rec.paths else {}
    for r, i in enumerate(idx):
        rgb, thm = rgb_frames.get(i), thm_frames.get(i)
        T = np.asarray(seq[min(i, seq.shape[0] - 1)], dtype=np.float32)
        ax = axes[r]
        if rgb is not None:
            ax[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        ax[0].set_title(f"RGB frame {i}")
        if thm is not None:
            ax[1].imshow(cv2.cvtColor(thm, cv2.COLOR_BGR2RGB))
        ax[1].set_title(f"Thermal MP4 frame {i} (visual only)")
        im = ax[2].imshow(T, cmap="inferno")
        fig.colorbar(im, ax=ax[2], fraction=0.046, label="°C")
        ax[2].set_title(f"Radiometric .npy frame {i}")
        if rgb is not None:
            over = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).copy()
            Tr = place_in_frame(T, over.shape[:2], content_box(rgb))
            c = thermal_hand_contour(np.nan_to_num(Tr, nan=float(np.nanmin(T))))
            if c is not None:
                cv2.drawContours(over, [c], -1, (0, 255, 0), 3)
            ax[3].imshow(over)
        ax[3].set_title("Thermal hand outline on RGB\n(.npy fitted to letterbox, unregistered)")
        for a in ax:
            a.axis("off")
    fig.suptitle(rec.rec_id, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def save_timestamp_figure(ts_rgb, ts_thm, rec_id: str, out_png: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    if ts_rgb is not None and len(ts_rgb) > 1:
        ax[0].plot(np.diff(ts_rgb), lw=0.7, label="RGB")
    if ts_thm is not None and len(ts_thm) > 1:
        ax[0].plot(np.diff(ts_thm), lw=0.7, label="Thermal MP4")
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("interval (ms)"); ax[0].legend(); ax[0].set_title("Frame intervals")
    if ts_rgb is not None and ts_thm is not None and len(ts_rgb) and len(ts_thm):
        k = min(len(ts_rgb), len(ts_thm))
        ax[1].plot(ts_rgb[:k] - ts_thm[:k], lw=0.7)
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("RGB − thermal (ms)"); ax[1].set_title("Timestamp difference")
    fig.suptitle(rec_id); fig.tight_layout(); fig.savefig(out_png, dpi=80); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main audit
# --------------------------------------------------------------------------- #
def audit_recording(rec: Recording, cfg: dict, fig_dir: Path, rng) -> tuple[dict, np.ndarray]:
    a = cfg["audit"]
    row = dict(rec_id=rec.rec_id, subject=rec.subject, hand=rec.hand, task=rec.task,
               missing_files=";".join(rec.missing), issues=[])
    if rec.missing:
        row["issues"].append("missing: " + ", ".join(rec.missing))
    pool = np.empty(0, np.float32)

    for stream in ("rgb", "thermal_mp4", "npy"):
        if stream in rec.paths:
            row[f"{stream}_bytes"] = rec.paths[stream].stat().st_size
            row[f"{stream}_fingerprint"] = file_fingerprint(rec.paths[stream], a["hash_bytes"])

    ts = {}
    for stream in ("rgb", "thermal_mp4"):
        if stream not in rec.paths:
            continue
        md = video_metadata(rec.paths[stream], a["scan_timestamps"])
        ts[stream] = md.pop("_timestamps", None)
        if not md.get("ok"):
            row["issues"].append(f"{stream}: {md.get('error')}")
        for k, v in md.items():
            if k != "ok":
                row[f"{stream}_{k}"] = v

    seq = None
    if "npy" in rec.paths:
        try:
            hdr = npy_header(rec.paths["npy"])
            row.update({f"npy_{k}": (str(v) if k == "shape" else v) for k, v in hdr.items()})
            n, h, w, note = frame_axes(hdr["shape"])
            row.update(npy_frames=n, npy_h=h, npy_w=w, npy_axis_note=note)
            if hdr["truncated"]:
                row["issues"].append("npy truncated (file shorter than header says)")
            else:
                seq = np.load(rec.paths["npy"], mmap_mode="r")
        except Exception as e:                          # corrupted header, etc.
            row["issues"].append(f"npy unreadable: {e}")

    if seq is not None:
        stats, pool = npy_stats(seq, a["n_sample_frames"], tuple(a["valid_temp_range_c"]), rng)
        row.update(stats)
        if stats["nan_frac"] > 0 or stats["inf_frac"] > 0:
            row["issues"].append("NaN/inf temperatures present")
        if stats["frozen_pairs"] > 0:
            row["issues"].append(f"{stats['frozen_pairs']} frozen frame pair(s) in sample")
        if "thermal_mp4" in rec.paths and seq.shape[0] > 2 * a["max_corr_lag"] + 1:
            row.update(mp4_npy_correspondence(rec.paths["thermal_mp4"], seq,
                                              a["n_corr_frames"], a["max_corr_lag"]))
            if np.isfinite(row.get("corr_lag0", np.nan)) and row["corr_lag0"] < 0.9:
                row["issues"].append(f"thermal MP4 and .npy match weakly (r={row['corr_lag0']:.2f})")
                f0 = read_video_frame(rec.paths["rgb"], 0) if "rgb" in rec.paths else None
                if f0 is not None:
                    b = content_box(f0)
                    g = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)[b[0]:b[1], b[2]:b[3]].astype(np.float32)
                    r_rgb = _corr(fit_to_box(seq[0], b), g)
                    row["rgb_file_corr_npy"] = r_rgb
                    if np.isfinite(r_rgb) and r_rgb > 0.9:
                        row["issues"].append(f"RGB and thermal MP4 file names appear SWAPPED (RGB file matches .npy, r={r_rgb:.2f}); "
                                             "add to pipeline.swapped_video_files")
            if np.isfinite(row.get("best_lag", np.nan)) and row["best_lag"] != 0:
                row["issues"].append(f"thermal MP4 is offset from .npy by {int(row['best_lag']):+d} frame(s)")
            if row.get("frames_mismatched", 0) > 0:
                row["issues"].append(f"{row['frames_mismatched']} thermal MP4 frame(s) differ from .npy "
                                     f"(r<0.95; first: {row['frames_mismatched_first']})")

    # frame-count agreement (decoded count when available, else header count)
    def count(stream):
        return row.get(f"{stream}_frames_decoded", row.get(f"{stream}_frames_header"))
    counts = {"rgb": count("rgb"), "thermal_mp4": count("thermal_mp4"), "npy": row.get("npy_frames")}
    known = {k: v for k, v in counts.items() if v is not None}
    row["frame_counts_equal"] = len(set(known.values())) == 1 if known else False
    if known and not row["frame_counts_equal"]:
        row["issues"].append("frame counts differ: " + ", ".join(f"{k}={v}" for k, v in known.items()))
    if "rgb_frames_decoded" in row and row["rgb_frames_decoded"] != row.get("rgb_frames_header"):
        row["issues"].append("RGB header frame count != decoded count")

    if seq is not None:
        rgb_hw = (row.get("rgb_height"), row.get("rgb_width"))
        row["npy_matches_rgb_size"] = (row.get("npy_h"), row.get("npy_w")) == rgb_hw
        row["npy_portrait"] = (row.get("npy_h"), row.get("npy_w")) == (640, 480)
        if "rgb" in rec.paths:
            f0 = read_video_frame(rec.paths["rgb"], 0)
            if f0 is not None:
                b = content_box(f0)
                row["rgb_box"] = f"y{b[0]}-{b[1]} x{b[2]}-{b[3]}"
    # does cv2 frame seeking land on the right frame? (middle frame, both videos)
    n_mid = row.get("npy_frames") or row.get("rgb_frames_header")
    for stream in ("rgb", "thermal_mp4"):
        if stream in rec.paths and n_mid:
            row[f"{stream}_seek_offset_mid"] = seek_offset(rec.paths[stream], int(n_mid) // 2)
        try:
            save_sample_figure(rec, seq, int(seq.shape[0]), fig_dir / f"{rec.rec_id}_samples.png")
        except Exception as e:
            row["issues"].append(f"figure failed: {e}")
    if ts.get("rgb") is not None or ts.get("thermal_mp4") is not None:
        rgb_ts, thm_ts = ts.get("rgb"), ts.get("thermal_mp4")
        if rgb_ts is not None and thm_ts is not None:
            k = min(len(rgb_ts), len(thm_ts))
            row["ts_rgb_minus_thermal_max_ms"] = float(np.max(np.abs(rgb_ts[:k] - thm_ts[:k]))) if k else np.nan
        save_timestamp_figure(rgb_ts, thm_ts, rec.rec_id, fig_dir / f"{rec.rec_id}_timestamps.png")

    row["issues"] = " | ".join(row["issues"])
    return row, pool


def find_duplicates(df: pd.DataFrame) -> list[str]:
    """Report identical files and tag the affected rows' issues in place."""
    msgs = []
    for stream in ("npy", "rgb", "thermal_mp4"):
        col = f"{stream}_fingerprint"
        if col not in df:
            continue
        for fp, grp in df.dropna(subset=[col]).groupby(col):
            if len(grp) > 1:
                ids = list(grp["rec_id"])
                msgs.append(f"{stream}: identical files in {', '.join(ids)}")
                for i in grp.index:
                    others = [x for x in ids if x != df.at[i, "rec_id"]]
                    tag = f"{stream} identical to {', '.join(others)}"
                    df.at[i, "issues"] = (df.at[i, "issues"] + " | " + tag) if df.at[i, "issues"] else tag
    return msgs


def write_report(df: pd.DataFrame, unexpected: list[str], dups: list[str], pool: np.ndarray,
                 out_dir: Path, elapsed: float, root: Path) -> Path:
    fig_hist = out_dir / "dataset_temperature_histogram.png"
    if pool.size:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(pool, bins=120, color="#c0504d")
        ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("sampled pixels")
        ax.set_title("Radiometric temperatures, all recordings (sampled pixels)")
        fig.tight_layout(); fig.savefig(fig_hist, dpi=100); plt.close(fig)

    def rng_str(col, fmt="{:.2f}"):
        if col not in df or df[col].dropna().empty:
            return "n/a"
        s = df[col].dropna()
        return f"{fmt.format(s.min())} – {fmt.format(s.max())} (median {fmt.format(s.median())})"

    n_subj = df["subject"].nunique()
    complete = df["missing_files"].eq("").sum()
    lines = [
        "# Dataset audit report", "",
        f"Dataset root: `{root}`  ",
        f"Audit time: {time.strftime('%Y-%m-%d %H:%M')} · run took {elapsed/60:.1f} min", "",
        "## Summary", "",
        "| Item | Value |", "| --- | --- |",
        f"| Subjects found | {n_subj} ({', '.join(sorted(df['subject'].unique(), key=subject_sort_key))}) |",
        f"| Recordings expected / complete | {len(df)} / {complete} |",
        f"| Recordings with any issue | {(df['issues'] != '').sum()} |",
        f"| RGB resolution | {', '.join(sorted({f'{int(w)}×{int(h)}' for w, h in zip(df.get('rgb_width', []), df.get('rgb_height', [])) if pd.notna(w)}))} |",
        f"| Radiometric shape (per frame) | {', '.join(sorted(set(df['npy_axis_note'].dropna())))} |",
        f"| Radiometric dtype | {', '.join(sorted(set(df.get('npy_dtype', pd.Series(dtype=str)).dropna())))} |",
        f"| Frames per recording (.npy) | {rng_str('npy_frames', '{:.0f}')} |",
        f"| RGB FPS (measured) | {rng_str('rgb_fps_measured')} |",
        f"| RGB FPS (header) | {rng_str('rgb_fps_header')} |",
        f"| RGB duration (s) | {rng_str('rgb_duration_s', '{:.1f}')} |",
        f"| Frame counts equal (RGB = thermal MP4 = .npy) | {int(df['frame_counts_equal'].sum())} of {len(df)} |",
        f"| Temperature min / max (°C) | {rng_str('t_min')} / {rng_str('t_max')} |",
        f"| Mean frame temperature (°C) | {rng_str('t_mean')} |",
        f"| Recordings with NaN/inf | {int(((df.get('nan_frac', 0) > 0) | (df.get('inf_frac', 0) > 0)).sum())} |",
        f"| Thermal MP4 ↔ .npy spatial match r (fitted to letterbox) | {rng_str('corr_lag0', '{:.3f}')} |",
        f"| Thermal MP4 ↔ .npy frame alignment | {df['lag_status'].value_counts(dropna=False).to_dict() if 'lag_status' in df else 'n/a'} |",
        f"| Thermal MP4 ↔ .npy worst single frame r (all frames checked) | {rng_str('frame_corr_min', '{:.3f}')} |",
        f"| Frames where thermal MP4 ≠ .npy (r<0.95), total | {int(df['frames_mismatched'].sum()) if 'frames_mismatched' in df else 'n/a'} |",
        f"| cv2 seek error at middle frame, RGB / thermal (frames) | {rng_str('rgb_seek_offset_mid', '{:+.0f}')} / {rng_str('thermal_mp4_seek_offset_mid', '{:+.0f}')} |",
        f"| Thermal MP4 content box | {df['thermal_box'].value_counts().to_dict() if 'thermal_box' in df else 'n/a'} |",
        f"| RGB content box | {df['rgb_box'].value_counts().to_dict() if 'rgb_box' in df else 'n/a'} |",
        "",
        "## Duplicates", "",
        *(["- " + d for d in dups] or ["None found."]), "",
        *(["Identical radiometric files mean one recording was saved twice under two task names."
           " Exclude the copy from task-level analysis (keep one) and record this in the report.", ""]
          if any(d.startswith("npy") for d in dups) else []),
        "## Unexpected files or folders", "",
        *(["- " + u for u in unexpected] or ["None."]), "",
        "## Recordings with issues", "",
    ]
    bad = df[df["issues"] != ""][["rec_id", "issues"]]
    if bad.empty:
        lines.append("None.")
    else:
        lines += ["| Recording | Issues |", "| --- | --- |"]
        lines += [f"| {r.rec_id} | {r.issues} |" for r in bad.itertuples()]
    lines += [
        "", "## Frame geometry", "",
        "- Each .npy frame is a portrait image, H=640 x W=480, already upright. Do NOT transpose it.",
        "- Both MP4s are 480x640 frames with black bars; the real image sits in the content box above",
        "  (a 3:4 portrait area). To compare with a video, resize the .npy frame into that box.",
        "- Never jump to a frame with cv2 `CAP_PROP_POS_FRAMES`: on these files the seek lands on",
        "  the wrong frame (see 'cv2 seek error' above). Decode in order and count frames; then",
        "  video frame i corresponds to .npy frame i.",
        "",
        "## How to read the figures", "",
        "- `*_samples.png`: frames 0, middle and last. Column 4 draws the thermal hand outline on the RGB",
        "  frame WITHOUT registration, so any offset you see is what Stage 5 must correct.",
        "- `*_timestamps.png`: frame intervals (variable frame rate shows as spikes) and the",
        "  RGB − thermal timestamp difference (flat at 0 = same clock).",
        f"- `{fig_hist.name}`: background and hand form two humps; their gap sets the mask threshold.",
        "", "## Not available in this dataset", "",
        "Emissivity, reflected temperature, camera-to-hand distance, room temperature, humidity and",
        "acclimatisation time are not recorded. No diagnosis or inflammation labels are present.",
    ]
    rpt = out_dir / "audit_report.md"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    return rpt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1 dataset audit")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--root", default=None, help="override dataset_root in config")
    ap.add_argument("--quick", action="store_true", help="skip per-frame timestamp scan")
    ap.add_argument("--limit", type=int, default=None, help="audit only the first N subjects")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    if args.quick:
        cfg["audit"]["scan_timestamps"] = False
    root = Path(args.root or cfg["dataset_root"])
    out_dir = Path(cfg["audit"]["output_dir"]); fig_dir = out_dir / "figures"
    meta_dir = Path(cfg["audit"]["metadata_dir"])
    for d in (out_dir, fig_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out_dir / "audit.log", mode="w", encoding="utf-8")])
    rng = np.random.default_rng(cfg.get("seed", 42))
    t0 = time.time()

    recs, unexpected = discover(root, args.limit)
    LOG.info("Found %d subjects, %d expected recordings", len({r.subject for r in recs}), len(recs))
    rows, pools = [], []
    for k, rec in enumerate(recs, 1):
        t = time.time()
        try:
            row, pool = audit_recording(rec, cfg, fig_dir, rng)
        except Exception as e:                       # never let one bad file stop the audit
            LOG.exception("Failed on %s", rec.rec_id)
            row, pool = dict(rec_id=rec.rec_id, subject=rec.subject, hand=rec.hand, task=rec.task,
                             missing_files=";".join(rec.missing), issues=f"audit crashed: {e}"), np.empty(0)
        rows.append(row); pools.append(pool)
        LOG.info("[%2d/%d] %-24s %5.1fs  %s", k, len(recs), rec.rec_id, time.time() - t,
                 row["issues"] or "OK")

    df = pd.DataFrame(rows)
    dups = find_duplicates(df)
    csv_path = out_dir / "dataset_audit.csv"
    df.to_csv(csv_path, index=False)
    df.to_csv(meta_dir / "dataset_audit.csv", index=False)
    pool = np.concatenate([p for p in pools if p.size]) if any(p.size for p in pools) else np.empty(0)
    rpt = write_report(df, unexpected, dups, pool, out_dir, time.time() - t0, root)
    LOG.info("Duplicates: %s", dups or "none")
    LOG.info("Wrote %s and %s", csv_path, rpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
