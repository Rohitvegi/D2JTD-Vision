"""Stages 5-8 runner: landmarks -> masks -> registration -> joint ROIs, for every recording.

Outputs (config pipeline.processed_dir):
  frames/<rec_id>.csv     one row per analysed frame: flags, 14 joint temps, dorsum temp, landmarks
  registration.csv        one row per recording: transform + held-out Dice
  qc/<rec_id>.png         visual check: RGB with landmarks | thermal with mapped ROIs and RGB outline

Usage (Python 3.12 env, from the project root):
  .venv312\\Scripts\\python.exe -m src.run_pipeline                 # all 68 recordings
  .venv312\\Scripts\\python.exe -m src.run_pipeline --only S1_left_static S2_right_open_close
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import landmarks as L
from src import registration as R
from src import rois
from src import landmark_correction as LC
from src.io_utils import RecInfo, crop_to_npy_scale, iter_frames, load_config, load_recordings
from src.segmentation import hand_size, rgb_mask, thermal_mask

log = logging.getLogger("pipeline")
CORE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))   # ~1 native sensor pixel (4 px)


def annotation_frames(n: int, stride: int) -> list[int]:
    """Two frames per recording reserved for manual RQ2 annotation (~10 % and ~90 % through)."""
    return sorted({int(round(f * (n - 1) / stride)) * stride for f in (0.1, 0.9)})


def register_recording(info: RecInfo, pc: dict, box, k: float) -> tuple[dict, np.ndarray]:
    rc = pc["registration"]
    idx = np.linspace(0, info.n_frames - 1, rc["n_frames"]).round().astype(int)
    lm = L.make_landmarker(pc["landmark_model"], video=False, min_conf=pc["min_detection_confidence"])
    pairs, sizes = [], []
    for i, rgb, T in iter_frames(info, idx, box):
        pts, _ = L.detect(lm, rgb)
        if pts is None:
            continue
        pts, fl = L.check_and_fix(pts, rgb.shape[:2])
        if not fl["ok"]:
            continue
        th_m, _ = thermal_mask(T)
        pairs.append((rgb_mask(rgb, pts), th_m))
        sizes.append(hand_size(pts))
    lm.close()
    if len(pairs) < 3:
        raise RuntimeError(f"{info.rec_id}: only {len(pairs)} usable frames for registration")
    if rc.get("fixed_scale"):
        scales = np.array([float(rc["fixed_scale"])])          # rigid rig: only the shift varies
    else:
        scales = np.arange(rc["scale_min"], rc["scale_max"] + 1e-9, rc["scale_step"])
    rig = rc.get("rig_default")                                # (s, tx, ty) median over all recordings
    reg = R.fit_recording(pairs, k, scales, tuple(rig) if rig else None)
    reg["hand_size_median"] = float(np.median(sizes))
    return reg, R.matrix(k, reg["scale"], reg["tx"], reg["ty"])


def load_calibration_model(pc: dict) -> dict | None:
    """Learned per-frame calibration (src.calibration_model), used when registration.mode is learned/hybrid."""
    mode = pc["registration"].get("mode", "per_recording")
    if mode == "per_recording":
        return None
    from src import calibration_model as CM
    cal = CM.load_model(Path(pc["registration"].get("calibration_model", CM.MODEL_PATH)))
    if cal is None:
        raise FileNotFoundError("registration.mode needs a trained calibration model: run python -m src.calibration_model train")
    cal = dict(cal, mode=mode)
    return cal


def process_recording(info: RecInfo, cfg: dict) -> dict:
    pc = cfg["pipeline"]
    box = tuple(pc["video_box"])
    k = crop_to_npy_scale(box)
    out_dir = Path(pc["processed_dir"])
    t0 = time.time()
    reg, M = register_recording(info, pc, box, k)

    stride = int(pc["frame_stride"])
    idx = sorted(set(range(0, info.n_frames, stride)) | set(annotation_frames(info.n_frames, stride)))
    lm = L.make_landmarker(pc["landmark_model"], video=True, min_conf=pc["min_detection_confidence"])
    corr = LC.load() if pc.get("use_landmark_correction", True) else None
    cal = load_calibration_model(pc)
    if cal is not None and corr is not None and cal.get("along_frac"):
        corr = dict(corr, along_frac=cal["along_frac"])          # correction re-learned for this calibration mode
    reg["calibration_mode"] = cal["mode"] if cal else "per_recording"
    rows, qc = [], None
    for i, rgb, T in iter_frames(info, idx, box):
        row = dict(rec_id=info.rec_id, frame=i, t_s=i / info.fps)
        pts, score, zz = L.detect(lm, rgb, ts_ms=int(1000 * i / info.fps), with_z=True)
        row["detected"] = pts is not None
        if pts is None:
            rows.append(row)
            continue
        pts, fl = L.check_and_fix(pts, rgb.shape[:2])
        row.update(fl)
        row["hand_score"] = score
        hand, thr = thermal_mask(T)
        core = cv2.erode(hand.astype(np.uint8), CORE_KERNEL) > 0
        row["mask_thr_C"] = thr
        ys, xs = np.nonzero(hand)
        row["th_area"] = int(ys.size)
        row["th_cx"], row["th_cy"] = (float(xs.mean()), float(ys.mean())) if ys.size else (np.nan, np.nan)
        row["openness"] = L.openness(pts)
        Mf = M
        if cal is not None:                               # per-frame calibration follows the hand's distance
            from src import calibration_model as CM
            Mf = CM.frame_matrix(cal, k, pts, (reg["tx"], reg["ty"]), reg["hand_size_median"])
        # transform actually used for this frame (joint view / inverse mapping reuse it exactly)
        row["cal_scale"], row["cal_tx"], row["cal_ty"], row["cal_angle"] = R.decompose(Mf, k)
        p_raw = R.map_points(pts, Mf)                     # stored as lm*_x/y (used to evaluate RQ2)
        p_npy = LC.apply(p_raw, corr)                     # joint centres used for ROIs
        row["lm_corrected"] = corr is not None
        jp = p_npy[[j[1] for j in L.JOINTS]].round().astype(int)
        inside = [(0 <= x < T.shape[1] and 0 <= y < T.shape[0] and hand[y, x]) for x, y in jp]
        row["joint_centres_on_hand"] = float(np.mean(inside))
        row["frame_valid"] = bool(fl["ok"] and row["joint_centres_on_hand"] >= pc["frame_valid_joint_frac"])
        if fl["ok"]:
            row.update(rois.extract(T, hand, p_npy, pc["roi"]["beta"], pc["roi"]["usable_frac"],
                                    core=core, min_px=pc["roi"]["min_px"],
                                    radius_mode=pc["roi"].get("radius_mode", "qiao"),
                                    width_frac=pc["roi"].get("width_frac", 0.9)))
        # joints not visible in the RGB image (folded under / covered) get NO temperature:
        # MediaPipe only guesses their position, so the ROI would measure some other skin.
        vis = L.joint_visibility(pts, zz)
        for name, reason in vis.items():
            row[f"{name}_vis"] = int(not reason)
            if reason and fl["ok"]:
                for s in ("_T", "_sd", "_mean", "_p90"):
                    if f"{name}{s}" in row:
                        row[f"{name}{s}"] = np.nan
        row["n_hidden_joints"] = int(sum(bool(r) for r in vis.values()))
        for j in range(21):
            row[f"lm{j}_x"], row[f"lm{j}_y"] = float(p_raw[j, 0]), float(p_raw[j, 1])
            row[f"lm{j}_z"] = float(zz[j]) if zz is not None else np.nan
        rows.append(row)
        if qc is None and row["frame_valid"] and i >= info.n_frames // 2:
            qc = (i, rgb, T, pts, p_npy, hand)
    lm.close()
    df = pd.DataFrame(rows)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "frames" / f"{info.rec_id}.csv", index=False)
    if qc is not None:
        save_qc(qc, M, pc, out_dir / "qc" / f"{info.rec_id}.png", info.rec_id)
    reg.update(rec_id=info.rec_id, subject=info.rec.subject, hand=info.rec.hand, task=info.rec.task,
               frames_analysed=len(df), seconds=round(time.time() - t0, 1),
               swapped_files=info.rec_id in set(pc.get("swapped_video_files", [])))
    return reg


def save_qc(qc, M, pc, path: Path, rec_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i, rgb, T, pts, p_npy, hand = qc
    path.parent.mkdir(parents=True, exist_ok=True)
    rm = R.warp_mask(rgb_mask(rgb, pts), M)
    fig, ax = plt.subplots(1, 2, figsize=(9, 6))
    ax[0].imshow(rgb)
    ax[0].plot(pts[:, 0], pts[:, 1], ".", color="#2a9d8f", ms=6)
    ax[0].set_title(f"RGB + landmarks (frame {i})")
    ax[1].imshow(T, cmap="inferno")
    ax[1].contour(rm, levels=[0.5], colors="#2a9d8f", linewidths=1)
    rj = rois.joint_radii(p_npy, hand, pc["roi"]["beta"], pc["roi"].get("radius_mode", "qiao"),
                          pc["roi"].get("width_frac", 0.9))
    for name, lm, finger, _ in L.JOINTS:
        ax[1].add_patch(plt.Circle(p_npy[lm], rj[name], fill=False, color="white", lw=1))
    ax[1].set_title("Thermal (.npy) + mapped joint ROIs\n(teal = RGB hand outline after registration)")
    for a in ax:
        a.axis("off")
    fig.suptitle(rec_id)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _worker(args):
    info, cfg = args
    os.environ.setdefault("GLOG_minloglevel", "2")
    cv2.setRNGSeed(int(cfg.get("seed", 42)))                 # GrabCut / k-means determinism
    try:
        return process_recording(info, cfg)
    except Exception as e:                                    # keep going; report at the end
        return dict(rec_id=info.rec_id, error=repr(e))


def write_manifest(out: Path, cfg: dict, infos, args) -> None:
    """Record exactly what produced this output folder (reproducibility)."""
    import hashlib
    import json
    import platform
    from importlib.metadata import version
    src = Path(__file__).parent
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(src.glob("*.py")))).hexdigest()
    pkgs = {}
    for p in ("mediapipe", "opencv-python", "numpy", "pandas", "scikit-learn"):
        try:
            pkgs[p] = version(p)
        except Exception:
            pkgs[p] = "unknown"
    man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), command=" ".join(map(str, [__name__] + (args or []))),
               config=cfg, src_sha256=code_hash, python=platform.python_version(), packages=pkgs,
               recordings=[i.rec_id for i in infos], seed=int(cfg.get("seed", 42)))
    json.dump(man, open(out / "run_manifest.json", "w"), indent=2, default=str)


def run(cfg: dict, infos, workers: int, merge: bool = False, progress=None) -> pd.DataFrame:
    """Process recordings in parallel; returns the registration table (also written to disk)."""
    out = Path(cfg["pipeline"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, (inf, cfg)): inf.rec_id for inf in infos}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            if progress:
                progress(n, len(infos), r)
    reg = pd.DataFrame(results).sort_values("rec_id")
    reg_path = out / "registration.csv"
    if merge and reg_path.is_file():
        old = pd.read_csv(reg_path)
        reg = pd.concat([old[~old["rec_id"].isin(reg["rec_id"])], reg]).sort_values("rec_id")
    reg.to_csv(reg_path, index=False)
    return reg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stages 5-8: landmarks, masks, registration, ROIs")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", nargs="*", help="rec_ids to process")
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--processed-dir", default=None,
                    help="write to this folder instead of config pipeline.processed_dir (versioned runs)")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config))
    if args.processed_dir:
        cfg["pipeline"]["processed_dir"] = args.processed_dir
    out = Path(cfg["pipeline"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "pipeline.log", mode="a")])
    infos = load_recordings(cfg, only=args.only)
    write_manifest(out, cfg, infos, argv if argv is not None else __import__("sys").argv[1:])
    log.info("Processing %d recordings with %d workers", len(infos), args.workers)

    def progress(n, total, r):
        if "error" in r:
            log.error("[%2d/%d] %-26s FAILED %s", n, total, r["rec_id"], r["error"])
        else:
            log.info("[%2d/%d] %-26s %5.1fs  heldout Dice %.3f  scale %.3f  t=(%.0f,%.0f)",
                     n, total, r["rec_id"], r["seconds"], r["dice_heldout_median"], r["scale"], r["tx"], r["ty"])
    reg = run(cfg, infos, args.workers, merge=bool(args.only), progress=progress)
    log.info("Wrote %s", out / "registration.csv")
    return 0 if "error" not in reg.columns or reg["error"].isna().all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
