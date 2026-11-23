"""Stage 12 — edge-style inference and benchmark.

Runs the live path the way a device would: frame in -> landmarks -> FIXED rig transform ->
thermal mask -> joint ROIs -> rolling D2JTD -> comparison with the healthy reference ->
per-joint status + measurement quality. Nothing is fitted at run time; the rig transform is the
median of the offline registration (data/processed/registration.csv), the reference comes from
models/healthy_reference.json (written by src.analysis).

Wording: output is "outside healthy reference range", never a diagnosis.

Usage:
  .venv312\\Scripts\\python.exe -m src.edge_demo --rec S2_right_static            # demo + figure
  .venv312\\Scripts\\python.exe -m src.edge_demo --rec S2_right_static --benchmark # timing / RAM
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil

from src import landmarks as L
from src import registration as R
from src import rois
from src import landmark_correction as LC
from src.io_utils import crop_to_npy_scale, iter_frames, load_config, load_recordings
from src.segmentation import thermal_mask

CORE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


class EdgeScreener:
    def __init__(self, cfg: dict, window: int = 25, z_flag: float = 2.0):
        pc = cfg["pipeline"]
        self.pc = pc
        box = tuple(pc["video_box"])
        k = crop_to_npy_scale(box)
        reg = pd.read_csv(Path(pc["processed_dir"]) / "registration.csv")
        self.M = R.matrix(k, reg.scale.median(), reg.tx.median(), reg.ty.median())
        self.ref = json.load(open("models/healthy_reference.json"))["joints"]
        self.corr = LC.load() if pc.get("use_landmark_correction", True) else None
        self.lm = L.make_landmarker(pc["landmark_model"], video=True, min_conf=pc["min_detection_confidence"])
        self.buf = deque(maxlen=window)                  # last N valid frames of D2JTD per joint
        self.z_flag = z_flag
        self.timing = {k: [] for k in ("landmarks", "thermal_mask", "rois", "total")}

    def step(self, rgb: np.ndarray, T: np.ndarray, ts_ms: int) -> dict:
        t0 = time.perf_counter()
        pts, _, zz = L.detect(self.lm, rgb, ts_ms, with_z=True)
        t1 = time.perf_counter()
        out = dict(quality="no hand")
        if pts is not None:
            pts, fl = L.check_and_fix(pts, rgb.shape[:2])
            hand, _ = thermal_mask(T)
            core = cv2.erode(hand.astype(np.uint8), CORE_KERNEL) > 0
            t2 = time.perf_counter()
            p = LC.apply(R.map_points(pts, self.M), self.corr)
            if fl["ok"]:
                r = rois.extract(T, hand, p, self.pc["roi"]["beta"], self.pc["roi"]["usable_frac"], core,
                                 self.pc["roi"]["min_px"], self.pc["roi"].get("radius_mode", "qiao"),
                                 self.pc["roi"].get("width_frac", 0.9))
                vis = L.joint_visibility(pts, zz)
                d = {j: (r["dorsum_T"] - r[f"{j}_T"]) if not vis[j] else float("nan") for j in L.JOINT_NAMES}
                self.buf.append(d)
                out = dict(quality="ok", pts=p, rois=r)
            else:
                out = dict(quality="landmark check failed", pts=p)
            t3 = time.perf_counter()
            self.timing["thermal_mask"].append(t2 - t1)
            self.timing["rois"].append(t3 - t2)
        self.timing["landmarks"].append(t1 - t0)
        self.timing["total"].append(time.perf_counter() - t0)
        out["status"] = self.status()
        return out

    def status(self) -> dict:
        """Per joint: rolling median D2JTD and whether it is outside the healthy range (z < -z_flag)."""
        if len(self.buf) < max(5, self.buf.maxlen // 2):
            return {}
        s = {}
        for j in L.JOINT_NAMES:
            v = np.nanmedian([b[j] for b in self.buf])
            ref = self.ref.get(j)
            z = (v - ref["mean"]) / ref["sd"] if ref and ref["sd"] > 0 and np.isfinite(v) else np.nan
            s[j] = dict(d2jtd=float(v), z=float(z),
                        label=("measurement unreliable" if not np.isfinite(z) else
                               "warmer than healthy range" if z < -self.z_flag else "within healthy range"))
        return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--rec", default="S2_right_static")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--warm-joint", default=None, help="add +2 C to this joint ROI (demo of a flag)")
    args = ap.parse_args(argv)
    os.environ.setdefault("GLOG_minloglevel", "2")
    cfg = load_config(Path(args.config))
    info = load_recordings(cfg, only=[args.rec])[0]
    box = tuple(cfg["pipeline"]["video_box"])
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    eng = EdgeScreener(cfg)
    rss_model = proc.memory_info().rss
    last = None
    peak = rss_model
    for i, rgb, T in iter_frames(info, range(min(args.frames, info.n_frames)), box):
        if args.warm_joint and last is not None and last.get("pts") is not None:
            name, lmi, finger, _ = next(j for j in L.JOINTS if j[0] == args.warm_joint)
            c = last["pts"][lmi]
            r = last["rois"][f"{name}_r"] + 4 if last.get("rois") else 25.0          # cover the whole ROI
            yy, xx = np.mgrid[0:T.shape[0], 0:T.shape[1]]
            T = T + 2.0 * (((xx - c[0]) ** 2 + (yy - c[1]) ** 2) < r ** 2)
        res = eng.step(rgb, T, int(1000 * i / info.fps))
        if res.get("pts") is not None:
            last = dict(res, T=T, i=i)
        peak = max(peak, proc.memory_info().rss)
    tm = {k: 1000 * np.median(v) for k, v in eng.timing.items() if v}
    out_dir = Path(cfg["pipeline"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    bench = dict(recording=args.rec, frames=args.frames, median_ms=tm, fps_equivalent=1000 / tm["total"],
                 rss_baseline_MB=rss0 / 2**20, rss_after_model_load_MB=rss_model / 2**20, rss_peak_MB=peak / 2**20,
                 landmark_model_MB=Path(cfg["pipeline"]["landmark_model"]).stat().st_size / 2**20,
                 reference_model_KB=Path("models/healthy_reference.json").stat().st_size / 1024,
                 cpu=os.environ.get("PROCESSOR_IDENTIFIER", "unknown"), logical_cpus=os.cpu_count())
    if args.benchmark:
        json.dump(bench, open(out_dir / "edge_benchmark.json", "w"), indent=2)
    print(json.dumps(bench, indent=1))
    st = eng.status()
    for j, s in st.items():
        print(f"{j:12s} D2JTD {s['d2jtd']:+.2f} C  z {s['z']:+.1f}  {s['label']}")
    if last is not None:
        save_map(last, st, out_dir / f"edge_demo_{args.rec}{'_warm' if args.warm_joint else ''}.png", eng.pc, args.rec)
    return 0


def save_map(last, st, path, pc, rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T, p = last["T"], last["pts"]
    rr0 = rois.radii(p, pc["roi"]["beta"])
    rr = {n: (last["rois"].get(f"{n}_r", rr0[f]) if last.get("rois") else rr0[f]) for n, _, f, _ in L.JOINTS}
    fig, ax = plt.subplots(figsize=(5, 6.2))
    ax.imshow(T, cmap="inferno")
    for name, lm, finger, _ in L.JOINTS:
        lab = st.get(name, {}).get("label", "measurement unreliable")
        col = {"within healthy range": "#0ca30c", "warmer than healthy range": "#d03b3b"}.get(lab, "#b4b2a9")
        ax.add_patch(plt.Circle(p[lm], rr[name], fill=False, color=col, lw=2))
        if lab == "warmer than healthy range":
            ax.annotate("outside range", p[lm], xytext=(10, -10), textcoords="offset points", color="white", fontsize=8)
    ax.set_title(f"{rec}: joint map (green = within healthy range,\nred = warmer than range, grey = unreliable)", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
