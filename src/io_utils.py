"""Stage 4 — shared input/output for the FLIROne dataset.

Rules learned in Stage 1 (see reports/audit/audit_report.md):
* The radiometric .npy frame is an upright portrait image, H=640 x W=480, in deg C. Never transpose.
* Both MP4s are 480x640 letterboxed frames; the real image is the box y0-480, x140-500 (3:4).
  Resizing that box by 4/3 gives exactly the .npy frame size.
* Video frame i == .npy frame i, but ONLY when frames are decoded in order.
  cv2 CAP_PROP_POS_FRAMES seeking lands up to +-30 frames off on these files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd

from src.data_audit import Recording, discover, load_config  # noqa: F401  (re-exported)

NPY_HW = (640, 480)            # radiometric frame (H, W)


@dataclass
class RecInfo:
    rec: Recording
    rgb_path: Path
    thermal_mp4_path: Path
    npy_path: Path
    fps: float                  # measured frames per second (Stage 1 audit), for real time axes
    n_frames: int

    @property
    def rec_id(self) -> str:
        return self.rec.rec_id


def load_recordings(cfg: dict, limit: int | None = None, only: list[str] | None = None) -> list[RecInfo]:
    """All complete recordings, with the known RGB/thermal file swaps corrected."""
    root = Path(cfg["dataset_root"])
    recs, _ = discover(root, limit)
    swapped = set(cfg["pipeline"].get("swapped_video_files", []))
    audit_csv = Path(cfg["audit"]["metadata_dir"]) / "dataset_audit.csv"
    fps_map, n_map = {}, {}
    if audit_csv.is_file():
        a = pd.read_csv(audit_csv)
        fps_map = dict(zip(a["rec_id"], a.get("rgb_fps_measured", pd.Series(dtype=float))))
        n_map = dict(zip(a["rec_id"], a.get("npy_frames", pd.Series(dtype=float))))
    out = []
    for r in recs:
        if r.missing or (only and r.rec_id not in only):
            continue
        rgb, thm = r.paths["rgb"], r.paths["thermal_mp4"]
        if r.rec_id in swapped:
            rgb, thm = thm, rgb
        seq = np.load(r.paths["npy"], mmap_mode="r")
        fps = fps_map.get(r.rec_id)
        if fps is None or not np.isfinite(fps):
            cap = cv2.VideoCapture(str(rgb))
            fps = cap.get(cv2.CAP_PROP_FPS) or 8.3
            cap.release()
        out.append(RecInfo(r, rgb, thm, r.paths["npy"], float(fps), int(n_map.get(r.rec_id, seq.shape[0]))))
    return out


def open_npy(info: RecInfo) -> np.ndarray:
    seq = np.load(info.npy_path, mmap_mode="r")
    assert seq.shape[1:] == NPY_HW, f"{info.rec_id}: unexpected npy frame shape {seq.shape}"
    return seq


def iter_frames(info: RecInfo, indices, box) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (i, rgb_crop[H,W,3] in RGB order, T[640,480] float32) for the wanted indices, in order.

    Video frames are decoded sequentially (grab() for skipped frames), never seeked.
    """
    wanted = sorted({int(i) for i in indices})
    if not wanted:
        return
    y0, y1, x0, x1 = box
    seq = open_npy(info)
    n = min(seq.shape[0], info.n_frames)
    cap = cv2.VideoCapture(str(info.rgb_path))
    want = set(wanted)
    i = 0
    try:
        while i <= wanted[-1] and i < n:
            if i in want:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = np.ascontiguousarray(bgr[y0:y1, x0:x1, ::-1])
                yield i, rgb, np.asarray(seq[i], dtype=np.float32)
            elif not cap.grab():
                break
            i += 1
    finally:
        cap.release()


def crop_to_npy_scale(box) -> float:
    """Scale factor from video crop pixels to .npy pixels (480/360 = 640/480 = 4/3)."""
    y0, y1, x0, x1 = box
    return NPY_HW[1] / (x1 - x0)
