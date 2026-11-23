"""v8 — Dataset management: import (ZIP / folder / single recording), file matching, validation,
calibration profiles and processing of NEW datasets without editing code.

Concepts
  * Profile   : regular expressions that read subject / hand / task / stream from file paths.
                Built-in: "flirone" (original layout) and "generic" (e.g. P01_L_static_rgb.mp4).
                Custom profiles can be given as JSON {"name":..., "pattern": "<regex with named groups>"}.
  * Manifest  : data/datasets/<dataset_id>/manifest.json — one entry per recording with its three
                files (rgb, thermal_video, radiometric), metadata and match status. Editable (manual
                matching) and versioned by creation time. Source files are referenced, never copied,
                except when a ZIP is extracted (into the dataset folder; sources are never deleted).
  * Calibration profile : models/calibration/<name>.json — camera geometry and RGB->thermal mapping.
                Recordings whose geometry differs are reported "Requires calibration".
  * Validation statuses : Valid | Valid with warnings | Invalid | Requires calibration |
                Requires manual file matching.  Invalid recordings are not processed unless the user
                explicitly includes them.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from src.data_audit import Recording, content_box

DATASETS = Path("data/datasets")
CALIB_DIR = Path("models/calibration")
STREAMS = ("rgb", "thermal_video", "radiometric")
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

PROFILES = {
    "flirone": r"(?P<subject>S\d+)/(?P<hand>left|right) hand/(?P<task>static|open_close)_"
               r"(?P<stream>Thermal_Radiometric_Sequence|Thermal|RGB)\.(?P<ext>npy|mp4)$",
    "generic": r"(?P<subject>[A-Za-z]{0,4}\d+)[_\- ](?P<hand>left|right|l|r)[_\- ](?P<task>static|open_close|openclose|oc|st)"
               r"[_\- ](?P<stream>rgb|visible|thermal|ir|radiometric|temperature|temp)\.(?P<ext>mp4|avi|mov|mkv|npy)$",
}
STREAM_MAP = {"rgb": "rgb", "visible": "rgb", "thermal": "thermal_video", "ir": "thermal_video",
              "thermal_radiometric_sequence": "radiometric", "radiometric": "radiometric",
              "temperature": "radiometric", "temp": "radiometric"}
HAND_MAP = {"l": "left", "r": "right", "left": "left", "right": "right"}
TASK_MAP = {"static": "static", "st": "static", "open_close": "open_close", "openclose": "open_close", "oc": "open_close"}


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #
def safe_extract(zip_path: Path, dest: Path, max_ratio: float = 200.0) -> list[Path]:
    """Extract a ZIP refusing path traversal, absolute paths, links, zip bombs and full disks."""
    zip_path, dest = Path(zip_path), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        total = sum(i.file_size for i in infos)
        packed = max(sum(i.compress_size for i in infos), 1)
        if total / packed > max_ratio:
            raise ValueError(f"refusing archive: expands {total / packed:.0f}x (possible zip bomb)")
        free = shutil.disk_usage(root).free
        if total > 0.9 * free:
            raise ValueError(f"not enough disk space: needs {total / 2**30:.1f} GB, {free / 2**30:.1f} GB free")
        out = []
        for i in infos:
            name = PurePosixPath(i.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or (i.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"refusing unsafe archive member: {i.filename}")
            target = (root / Path(*name.parts)).resolve()
            if root not in target.parents:
                raise ValueError(f"refusing member outside destination: {i.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            out.append(target)
    return out


def classify(path: Path, root: Path, profiles: dict) -> dict | None:
    rel = path.relative_to(root).as_posix()
    for pname, pat in profiles.items():
        m = re.search(pat, rel, flags=re.IGNORECASE)
        if m:
            g = m.groupdict()
            stream = STREAM_MAP.get(g["stream"].lower())
            if stream is None:
                continue
            ext = "." + g["ext"].lower()
            if stream == "radiometric" and ext != ".npy":
                continue
            if stream != "radiometric" and ext not in VIDEO_EXT:
                continue
            return dict(subject=g["subject"].upper(), hand=HAND_MAP[g["hand"].lower()],
                        task=TASK_MAP[g["task"].lower()], stream=stream, profile=pname, path=str(path))
    return None


def build_manifest(root: Path, name: str | None = None, profiles: dict | None = None,
                   dataset_id: str | None = None, source: str = "folder") -> dict:
    """Scan a folder (no copying), match files into recordings."""
    root = Path(root).resolve()
    profiles = profiles or PROFILES
    recs, unmatched = {}, []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        c = classify(p, root, profiles)
        if c is None:
            if p.suffix.lower() in VIDEO_EXT | {".npy"}:
                unmatched.append(str(p))
            continue
        rid = f"{c['subject']}_{c['hand']}_{c['task']}"
        r = recs.setdefault(rid, dict(rec_id=rid, subject=c["subject"], hand=c["hand"], task=c["task"],
                                      files={}, duplicates={}, profile=c["profile"]))
        if c["stream"] in r["files"]:
            r["duplicates"].setdefault(c["stream"], []).append(c["path"])
        else:
            r["files"][c["stream"]] = c["path"]
    for r in recs.values():
        r["missing"] = [s for s in STREAMS if s not in r["files"]]
        r["match_status"] = "requires manual matching" if (r["missing"] and "radiometric" in r["missing"]) or r["duplicates"] \
            else ("complete" if not r["missing"] else "complete for analysis (thermal video optional)")
        r["label"] = None
    did = dataset_id or f"{(name or root.name).replace(' ', '_')}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    return dict(dataset_id=did, name=name or root.name, created=dt.datetime.now().isoformat(timespec="seconds"),
                root=str(root), source=source, profiles=list(profiles), recordings=sorted(recs.values(), key=lambda r: r["rec_id"]),
                unmatched_files=unmatched)


def import_zip(zip_path: Path, name: str | None = None, profiles: dict | None = None) -> dict:
    did = f"{(name or Path(zip_path).stem).replace(' ', '_')}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    raw = DATASETS / did / "raw"
    safe_extract(zip_path, raw)
    m = build_manifest(raw, name or Path(zip_path).stem, profiles, did, source=f"zip:{Path(zip_path).name}")
    save_manifest(m)
    return m


def import_folder(folder: Path, name: str | None = None, profiles: dict | None = None) -> dict:
    m = build_manifest(folder, name, profiles, source="folder (referenced in place, not copied)")
    save_manifest(m)
    return m


def import_single(files: dict, subject: str, hand: str, task: str, name: str = "single_recording") -> dict:
    """Individual upload: files = {"rgb": path, "thermal_video": path (optional), "radiometric": path}."""
    rid = f"{subject.upper()}_{HAND_MAP[hand.lower()]}_{TASK_MAP[task.lower()]}"
    r = dict(rec_id=rid, subject=subject.upper(), hand=HAND_MAP[hand.lower()], task=TASK_MAP[task.lower()],
             files={k: str(v) for k, v in files.items() if v}, duplicates={}, profile="manual", label=None)
    r["missing"] = [s for s in STREAMS if s not in r["files"]]
    r["match_status"] = "requires manual matching" if "radiometric" in r["missing"] or "rgb" in r["missing"] else "complete"
    m = dict(dataset_id=f"{name}_{dt.datetime.now():%Y%m%d_%H%M%S}", name=name, created=dt.datetime.now().isoformat(timespec="seconds"),
             root=str(Path(files["rgb"]).parent), source="individual upload", profiles=["manual"], recordings=[r], unmatched_files=[])
    save_manifest(m)
    return m


def set_file(manifest: dict, rec_id: str, stream: str, path: str) -> dict:
    """Manual matching: assign a file to a recording's stream and re-derive its status."""
    r = next(x for x in manifest["recordings"] if x["rec_id"] == rec_id)
    r["files"][stream] = str(path)
    r["duplicates"].pop(stream, None)
    r["missing"] = [s for s in STREAMS if s not in r["files"]]
    r["match_status"] = "complete" if not r["missing"] else (
        "requires manual matching" if "radiometric" in r["missing"] or "rgb" in r["missing"] else "complete for analysis (thermal video optional)")
    r["manually_matched"] = True
    return manifest


def save_manifest(m: dict) -> Path:
    d = DATASETS / m["dataset_id"]
    d.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(d / "manifest.json", "w"), indent=2)
    return d / "manifest.json"


def load_manifest(dataset_id_or_path: str) -> dict:
    p = Path(dataset_id_or_path)
    p = p if p.suffix == ".json" else DATASETS / dataset_id_or_path / "manifest.json"
    return json.load(open(p))


def list_datasets() -> list[dict]:
    out = []
    for p in sorted(DATASETS.glob("*/manifest.json")):
        m = json.load(open(p))
        out.append(dict(dataset_id=m["dataset_id"], name=m["name"], created=m["created"], n=len(m["recordings"])))
    return out


def fingerprint(path: str, nbytes: int = 1 << 20) -> str:
    h = hashlib.md5()
    p = Path(path)
    size = p.stat().st_size
    h.update(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(nbytes))
        if size > nbytes:
            f.seek(size - nbytes)
            h.update(f.read(nbytes))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Calibration profiles
# --------------------------------------------------------------------------- #
def calibration_from_config(cfg: dict, name: str = "flirone_one_pro_v3") -> dict:
    pc = cfg["pipeline"]
    rc = pc["registration"]
    return dict(name=name, created=dt.datetime.now().isoformat(timespec="seconds"),
                camera="FLIR ONE Pro (smartphone add-on)", method="similarity: fixed scale + per-recording shift (v3)",
                note="NOT a stereo calibration; derived from hand-mask overlap on the FLIROne dataset",
                video_hw=[480, 640], video_box=list(pc["video_box"]), npy_hw=[640, 480],
                fixed_scale=rc.get("fixed_scale"), rig_default=rc.get("rig_default"),
                landmark_correction="models/landmark_correction.json")


def save_calibration(cal: dict) -> Path:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    p = CALIB_DIR / f"{cal['name']}.json"
    if p.exists():
        raise FileExistsError(f"{p} exists: calibration profiles are versioned, choose a new name")
    json.dump(cal, open(p, "w"), indent=2)
    return p


def load_calibration(name: str) -> dict:
    return json.load(open(CALIB_DIR / f"{name}.json"))


def list_calibrations() -> list[str]:
    return sorted(p.stem for p in CALIB_DIR.glob("*.json"))


def apply_calibration(cfg: dict, cal: dict) -> dict:
    import copy
    c = copy.deepcopy(cfg)
    c["pipeline"]["video_box"] = cal["video_box"]
    c["pipeline"]["registration"]["fixed_scale"] = cal.get("fixed_scale", cal.get("scale"))
    c["pipeline"]["registration"]["rig_default"] = cal["rig_default"]
    c["pipeline"]["swapped_video_files"] = []
    if cal.get("shift_model"):                       # learned (v9) profile: calibration applied per frame
        c["pipeline"]["registration"]["mode"] = cal.get("mode", "learned")
        c["pipeline"]["registration"]["calibration_model"] = str(CALIB_DIR / f"{cal['name']}.json")
    else:                                            # v3 profile: one fitted shift per recording
        c["pipeline"]["registration"]["mode"] = "per_recording"
    return c


def auto_calibrate_dataset(m: dict, cfg: dict, name: str | None = None, progress=None) -> dict:
    """Find the calibration of a new dataset/camera from its own recordings and save it as a profile."""
    from src.calibration import find_calibration
    infos = infos_from_manifest(m)
    if not infos:
        raise ValueError("no recording with both an RGB video and a radiometric file")
    name = name or f"auto_{m['dataset_id']}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    cal = find_calibration([(i.rgb_path, i.npy_path) for i in infos[:12]], name, landmark_model=cfg["pipeline"]["landmark_model"],
                           progress=progress)
    prof = cal.to_profile()
    prof["dataset"] = m["dataset_id"]
    prof["self_check"] = dict(cal.quality, dice_heldout_median=cal.quality["dice_heldout_median"])
    save_calibration(prof)
    return prof


# --------------------------------------------------------------------------- #
# Recordings for the pipeline
# --------------------------------------------------------------------------- #
@dataclass
class _Info:                                   # duck-type of io_utils.RecInfo
    rec: Recording
    rgb_path: Path
    thermal_mp4_path: Path | None
    npy_path: Path
    fps: float
    n_frames: int

    @property
    def rec_id(self):
        return self.rec.rec_id


def video_props(path: str) -> dict:
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    d = dict(ok=ok, n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0, fps=float(cap.get(cv2.CAP_PROP_FPS)) if ok else 0.0,
             w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if ok else 0, h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if ok else 0)
    frame = None
    if ok:
        r, frame = cap.read()
        d["first_frame_ok"] = bool(r)
    cap.release()
    d["frame"] = frame
    return d


def infos_from_manifest(m: dict, rec_ids: list[str] | None = None) -> list[_Info]:
    out = []
    for r in m["recordings"]:
        if rec_ids is not None and r["rec_id"] not in rec_ids:
            continue
        if "rgb" not in r["files"] or "radiometric" not in r["files"]:
            continue
        vp = video_props(r["files"]["rgb"])
        seq = np.load(r["files"]["radiometric"], mmap_mode="r")
        fps = vp["fps"] if vp["fps"] > 0 else 8.7
        rec = Recording(r["subject"], r["hand"], r["task"], Path(r["files"]["rgb"]).parent,
                        paths={"rgb": Path(r["files"]["rgb"]), "npy": Path(r["files"]["radiometric"])})
        out.append(_Info(rec, Path(r["files"]["rgb"]), Path(r["files"]["thermal_video"]) if "thermal_video" in r["files"] else None,
                         Path(r["files"]["radiometric"]), fps, int(min(seq.shape[0], vp["n"] or seq.shape[0]))))
    return out


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_recording(r: dict, cal: dict, cfg: dict, n_probe: int = 5) -> dict:
    """Checks -> list of (check, level, message); level in ok / warn / fail / calib / match."""
    checks = []
    add = lambda c, lvl, msg: checks.append(dict(check=c, level=lvl, message=msg))  # noqa: E731
    if r.get("duplicates"):
        add("file matching", "match", f"several files for {list(r['duplicates'])}: choose one")
    for s in ("rgb", "radiometric"):
        if s not in r["files"]:
            add("required files", "match", f"missing {s} file")
    if "thermal_video" not in r["files"]:
        add("required files", "warn", "no thermal video (only used for checks; radiometric data is the measurement)")
    if any(c["level"] == "match" for c in checks):
        return _status(r, checks)
    vp = video_props(r["files"]["rgb"])
    if not vp["ok"] or not vp.get("first_frame_ok"):
        add("RGB readable", "fail", "RGB video cannot be opened/decoded")
        return _status(r, checks)
    add("RGB readable", "ok", f"{vp['w']}x{vp['h']}, {vp['n']} frames (header), {vp['fps']:.1f} fps (header)")
    try:
        seq = np.load(r["files"]["radiometric"], mmap_mode="r")
    except Exception as e:
        add("radiometric readable", "fail", f"cannot read .npy: {e}")
        return _status(r, checks)
    add("radiometric readable", "ok", f"shape {tuple(seq.shape)}, dtype {seq.dtype}")
    if seq.ndim != 3 or not np.issubdtype(seq.dtype, np.floating):
        add("radiometric format", "fail", "expected a float array (frames, H, W)")
        return _status(r, checks)
    if [vp["h"], vp["w"]] != cal["video_hw"] or list(seq.shape[1:]) != cal["npy_hw"]:
        add("geometry vs calibration", "calib",
            f"video {vp['h']}x{vp['w']} / radiometric {seq.shape[1]}x{seq.shape[2]} differ from calibration "
            f"'{cal['name']}' ({cal['video_hw']} / {cal['npy_hw']})")
        return _status(r, checks)
    b = content_box(vp["frame"])
    if abs(b[2] - cal["video_box"][2]) > 6 or abs(b[3] - cal["video_box"][3]) > 6:
        add("image area vs calibration", "calib", f"letterbox {b} differs from calibration {cal['video_box']}")
    idx = np.linspace(0, seq.shape[0] - 1, num=min(8, seq.shape[0])).round().astype(int)
    sample = np.stack([np.asarray(seq[i], np.float32) for i in idx])
    fin = np.isfinite(sample)
    nan_frac = 1 - fin.mean()
    med = float(np.median(sample[fin])) if fin.any() else np.nan
    if 10 <= med <= 45:
        add("temperature units", "ok", f"median {med:.1f} (deg C plausible)")
    elif 280 <= med <= 320:
        add("temperature units", "fail", f"median {med:.1f}: looks like Kelvin - convert to deg C first")
    else:
        add("temperature units", "fail", f"median {med:.1f}: not a plausible skin/room temperature in deg C")
    add("invalid pixels", "warn" if nan_frac > 0.01 else "ok", f"{nan_frac * 100:.2f} % non-finite in sampled frames")
    n_v, n_t = vp["n"], seq.shape[0]
    if "thermal_video" in r["files"]:
        tv = video_props(r["files"]["thermal_video"])
        same = n_v == n_t == tv["n"]
        add("frame counts", "ok" if same else "warn", f"RGB {n_v}, thermal video {tv['n']}, radiometric {n_t}")
    else:
        add("frame counts", "ok" if n_v == n_t else "warn", f"RGB {n_v}, radiometric {n_t}")
    if abs(n_v - n_t) > max(3, 0.02 * n_t):
        add("synchronisation", "fail", "RGB and radiometric frame counts differ: frame i is not the same moment")
    else:
        add("synchronisation", "ok", "frame counts agree (index pairing valid; timestamps not available)")
    # probe frames: hand detection, thermal segmentation, registration overlap, ROI availability
    try:
        probe = _probe(r, cal, cfg, n_probe, n_t)
        add("hand detection", "ok" if probe["detected"] >= max(1, n_probe - 1) else ("warn" if probe["detected"] else "fail"),
            f"hand found in {probe['detected']}/{n_probe} probe frames")
        add("thermal hand segmentation", "ok" if probe["area_ok"] else "warn",
            f"thermal hand area {probe['area_frac'] * 100:.0f} % of frame")
        if np.isfinite(probe["dice"]):
            add("registration (calibrated mapping)", "ok" if probe["dice"] >= 0.8 else ("warn" if probe["dice"] >= 0.6 else "calib"),
                f"hand-outline Dice {probe['dice']:.2f} with calibration '{cal['name']}'")
        add("joint ROI availability", "ok" if probe["roi_usable"] >= 0.8 else "warn", f"{probe['roi_usable'] * 100:.0f} % of joint ROIs usable")
    except Exception as e:                                  # never crash the validator
        add("probe", "warn", f"could not run probe frames: {e}")
    return _status(r, checks)


def _probe(r, cal, cfg, n_probe, n_t):
    from src import landmarks as L
    from src import registration as R
    from src import rois
    from src.io_utils import crop_to_npy_scale
    from src.segmentation import dice, rgb_mask, thermal_mask
    box = tuple(cal["video_box"])
    k = crop_to_npy_scale(box)
    s, tx, ty = cal["rig_default"]
    M = R.matrix(k, s, tx, ty, float(cal.get("angle_deg", 0.0)))
    lm = L.make_landmarker(cfg["pipeline"]["landmark_model"], video=False)
    seq = np.load(r["files"]["radiometric"], mmap_mode="r")
    want = set(np.linspace(0, n_t - 1, n_probe).round().astype(int).tolist())
    cap = cv2.VideoCapture(r["files"]["rgb"])
    det, dices, areas, usable = 0, [], [], []
    i = 0
    while want and i <= max(want):
        if i in want:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = np.ascontiguousarray(bgr[box[0]:box[1], box[2]:box[3], ::-1])
            T = np.asarray(seq[i], np.float32)
            hand, _ = thermal_mask(np.nan_to_num(T, nan=float(np.nanmedian(T))))
            areas.append(hand.mean())
            pts, _ = L.detect(lm, rgb)
            if pts is not None:
                det += 1
                pts, _ = L.check_and_fix(pts, rgb.shape[:2])
                dices.append(dice(R.warp_mask(rgb_mask(rgb, pts), M), hand))
                ro = rois.extract(T, hand, R.map_points(pts, M), cfg["pipeline"]["roi"]["beta"], cfg["pipeline"]["roi"]["usable_frac"],
                                  radius_mode=cfg["pipeline"]["roi"].get("radius_mode", "qiao"))
                usable.append(np.mean([np.isfinite(ro[f"{j}_T"]) for j in L.JOINT_NAMES]))
        elif not cap.grab():
            break
        i += 1
    cap.release()
    lm.close()
    af = float(np.median(areas)) if areas else 0.0
    return dict(detected=det, dice=float(np.median(dices)) if dices else np.nan, area_frac=af,
                area_ok=0.05 <= af <= 0.8, roi_usable=float(np.mean(usable)) if usable else 0.0)


def _status(r, checks):
    lv = {c["level"] for c in checks}
    status = ("Requires manual file matching" if "match" in lv else "Invalid" if "fail" in lv
              else "Requires calibration" if "calib" in lv else "Valid with warnings" if "warn" in lv else "Valid")
    return dict(rec_id=r["rec_id"], status=status, checks=checks)


def validate_manifest(m: dict, cal: dict, cfg: dict, progress=None) -> list[dict]:
    out = []
    fps_seen = {}
    for n, r in enumerate(m["recordings"], 1):
        v = validate_recording(r, cal, cfg)
        if "radiometric" in r["files"] and Path(r["files"]["radiometric"]).is_file():
            fpr = fingerprint(r["files"]["radiometric"])
            if fpr in fps_seen:
                v["checks"].append(dict(check="duplicate", level="warn", message=f"radiometric file identical to {fps_seen[fpr]}"))
                if v["status"] == "Valid":
                    v["status"] = "Valid with warnings"
            fps_seen[fpr] = r["rec_id"]
        out.append(v)
        if progress:
            progress(n, len(m["recordings"]), v)
    d = DATASETS / m["dataset_id"]
    d.mkdir(parents=True, exist_ok=True)
    json.dump(dict(dataset_id=m["dataset_id"], calibration=cal["name"], validated=dt.datetime.now().isoformat(timespec="seconds"),
                   results=out), open(d / "validation.json", "w"), indent=2)
    return out


def process_dataset(m: dict, cfg: dict, cal: dict, rec_ids: list[str], workers: int = 4, progress=None):
    """Run the measurement pipeline (Stages 5-8) + feature extraction on chosen recordings."""
    import pandas as pd
    from src import run_pipeline
    from src.feature_extraction import extract_dataset, load_frames
    c = apply_calibration(cfg, cal)
    out = DATASETS / m["dataset_id"] / "processed"
    c["pipeline"]["processed_dir"] = str(out)
    infos = infos_from_manifest(m, rec_ids)
    reg = run_pipeline.run(c, infos, workers, merge=True, progress=progress)
    F = extract_dataset(load_frames(out / "frames"), pd.read_csv(out / "registration.csv"))
    F.to_csv(out / "features.csv", index=False)
    return reg, F
