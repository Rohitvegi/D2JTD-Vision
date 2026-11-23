"""v8 — Live acquisition with timestamp-based RGB/thermal pairing.

Real cameras run at different rates (e.g. RGB ~30-60 fps, thermal ~8.7 Hz). Every RGB frame is
paired with the most recent thermal frame whose timestamp is within `max_age_ms`; the age of that
thermal frame is reported and RGB frames without a fresh thermal frame are NOT measured.

Sources
  * OpenCVCamera    — any webcam / USB RGB camera (real hardware, RGB only).
  * FlirOneSource   — NOT AVAILABLE: the FLIR ONE Pro streams to a phone app through FLIR's mobile
                      SDK; no desktop driver was available for this project. Raises with an
                      explanation so the GUI can say so.
  * SimulatedLive   — SIMULATION: replays a recorded dataset recording on two independent clocks
                      (RGB at the recording's rate, thermal at 8.7 Hz with timing jitter) to exercise
                      the pairing and the live measurement path. Clearly labelled as simulation.

Usage:  .venv312\\Scripts\\python.exe -m src.live_acquisition --simulate S2_right_static --seconds 20
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    t_ms: float
    data: np.ndarray
    index: int = -1                  # source index (simulation: ground-truth frame number)


class PairingBuffer:
    """Keeps recent thermal frames; pairs an RGB timestamp with the newest thermal frame not newer
    than the RGB frame (+tolerance) and not older than max_age_ms."""

    def __init__(self, max_age_ms: float = 150.0, tolerance_ms: float = 20.0, size: int = 32):
        self.buf: deque[Frame] = deque(maxlen=size)
        self.max_age, self.tol = max_age_ms, tolerance_ms
        self.stats = dict(rgb=0, paired=0, unpaired=0, ages=[])

    def push_thermal(self, f: Frame):
        self.buf.append(f)

    def pair(self, rgb_t_ms: float) -> tuple[Frame | None, float]:
        self.stats["rgb"] += 1
        cands = [f for f in self.buf if f.t_ms <= rgb_t_ms + self.tol]
        if not cands:
            self.stats["unpaired"] += 1
            return None, np.nan
        f = max(cands, key=lambda x: x.t_ms)
        age = rgb_t_ms - f.t_ms
        if age > self.max_age:
            self.stats["unpaired"] += 1
            return None, age
        self.stats["paired"] += 1
        self.stats["ages"].append(age)
        return f, age


class OpenCVCamera:
    """Real RGB camera (index 0 = default webcam)."""
    simulated = False

    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"RGB camera {index} not available")
        self.t0 = time.perf_counter()

    def read(self) -> Frame | None:
        ok, bgr = self.cap.read()
        return Frame(1000 * (time.perf_counter() - self.t0), bgr[:, :, ::-1].copy()) if ok else None

    def close(self):
        self.cap.release()


class FlirOneSource:
    simulated = False

    def __init__(self, *a, **k):
        raise NotImplementedError(
            "FLIR ONE Pro live streaming needs FLIR's mobile SDK (Android/iOS); no desktop driver was available. "
            "Record with the FLIR ONE app, export RGB + radiometric data and use Dataset Upload, or use Simulation mode.")


class SimulatedLive:
    """SIMULATION: replay one recorded recording on two clocks."""
    simulated = True

    def __init__(self, info, box, thermal_hz: float = 8.7, jitter_ms: float = 8.0, seed: int = 0):
        from src.io_utils import open_npy
        self.info, self.box = info, box
        self.seq = open_npy(info) if hasattr(info, "rec") and info.npy_path else np.load(info.npy_path, mmap_mode="r")
        self.n = min(int(info.n_frames), self.seq.shape[0])
        self.rgb_period = 1000.0 / info.fps
        self.th_period = 1000.0 / thermal_hz
        self.rng = np.random.default_rng(seed)
        self.cap = cv2.VideoCapture(str(info.rgb_path))
        self.i = 0

    def frames(self):
        """Yields ('thermal'|'rgb', Frame) in time order. Recorded RGB frame i was captured at the same
        moment as radiometric frame i, so thermal frames are released on the thermal clock at the time
        of the RGB frame they belong to (plus jitter) — the pairing must recover index i."""
        y0, y1, x0, x1 = self.box
        next_th = 0.0
        while self.i < self.n:
            ok, bgr = self.cap.read()
            if not ok:
                break
            t_rgb = self.i * self.rgb_period
            while next_th <= t_rgb:                       # thermal clock ticks up to now
                k = min(int(round(next_th / self.rgb_period)), self.n - 1)
                t = k * self.rgb_period + self.rng.uniform(-self.rgb_period * 0.1, 0) if k else 0.0
                yield "thermal", Frame(t, np.asarray(self.seq[k], np.float32), k)
                next_th += self.th_period + self.rng.normal(0, 2)
            yield "rgb", Frame(t_rgb, np.ascontiguousarray(bgr[y0:y1, x0:x1, ::-1]), self.i)
            self.i += 1

    def close(self):
        self.cap.release()


def run_simulation(rec_id: str, seconds: float = 20.0, cfg_path: str = "config.yaml", progress=None) -> dict:
    """Simulated live session: pairing + measurement via the edge screener."""
    from src.edge_demo import EdgeScreener
    from src.io_utils import load_config, load_recordings
    cfg = load_config(Path(cfg_path))
    info = load_recordings(cfg, only=[rec_id])[0]
    src = SimulatedLive(info, tuple(cfg["pipeline"]["video_box"]))
    buf = PairingBuffer(max_age_ms=1.5 * src.th_period)
    eng = EdgeScreener(cfg)
    t_wall = time.perf_counter()
    index_match, measured, last = [], 0, {}
    for kind, f in src.frames():
        if f.t_ms > 1000 * seconds:
            break
        if kind == "thermal":
            buf.push_thermal(f)
            continue
        th, age = buf.pair(f.t_ms)
        if th is None:
            continue
        index_match.append(abs(th.index - f.index))
        last = eng.step(f.data, th.data, int(f.t_ms))
        measured += 1
        if progress:
            progress(f, th, age, last)
    src.close()
    wall = time.perf_counter() - t_wall
    ages = np.array(buf.stats["ages"])
    return dict(mode="SIMULATION (recorded data replayed; no live camera)", recording=rec_id, seconds_simulated=seconds,
                rgb_frames=buf.stats["rgb"], paired=buf.stats["paired"], unpaired=buf.stats["unpaired"],
                thermal_age_ms_median=float(np.median(ages)) if ages.size else np.nan,
                thermal_age_ms_max=float(ages.max()) if ages.size else np.nan,
                pairs_exact_frame=float(np.mean(np.array(index_match) == 0)) if index_match else np.nan,
                pairs_within_1_frame=float(np.mean(np.array(index_match) <= 1)) if index_match else np.nan,
                measured_frames=measured, processing_fps=measured / wall if wall else np.nan,
                last_status={k: v["label"] for k, v in last.get("status", {}).items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", default="S2_right_static")
    ap.add_argument("--seconds", type=float, default=20)
    args = ap.parse_args(argv)
    r = run_simulation(args.simulate, args.seconds)
    Path("reports/v8_live").mkdir(parents=True, exist_ok=True)
    json.dump(r, open("reports/v8_live/simulation.json", "w"), indent=2, default=float)
    print(json.dumps(r, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
