"""Installation health check: is this copy ready to run?

  rgbthermal doctor            (or: python -m src.doctor)

Each check prints OK / WARN / FAIL with a reason. Exit code 0 when nothing FAILs.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
REQUIRED_FILES = ["config.yaml", "models/hand_landmarker.task", "models/landmark_correction.json"]
LOCK = HOME / "requirements.lock.txt"
CHECKSUMS = HOME / "models" / "MODEL_SHA256.json"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checks() -> list[tuple[str, str, str]]:
    out = []
    add = lambda name, level, msg: out.append((name, level, msg))  # noqa: E731
    v = sys.version_info
    add("Python", "OK" if (v.major, v.minor) == (3, 12) else "FAIL",
        f"{platform.python_version()} ({'required 3.12 (MediaPipe wheels)' if (v.major, v.minor) != (3, 12) else 'supported'})")
    # pinned packages
    if LOCK.is_file():
        bad = []
        for line in LOCK.read_text().splitlines():
            line = line.split("#")[0].strip()
            if "==" not in line:
                continue
            name, want = line.split("==")
            try:
                have = md.version(name)
                if have != want:
                    bad.append(f"{name} {have} (expected {want})")
            except md.PackageNotFoundError:
                bad.append(f"{name} missing")
        add("Packages", "FAIL" if any("missing" in b for b in bad) else "WARN" if bad else "OK",
            "; ".join(bad) if bad else "all pinned versions installed")
    else:
        add("Packages", "WARN", "requirements.lock.txt not found")
    # files and model integrity
    miss = [f for f in REQUIRED_FILES if not (HOME / f).is_file()]
    add("Program files", "FAIL" if miss else "OK", f"missing: {miss}" if miss else "config and models present")
    if CHECKSUMS.is_file():
        exp = json.load(open(CHECKSUMS))
        wrong = [k for k, h in exp.items() if not (HOME / k).is_file() or _sha(HOME / k) != h]
        add("Model integrity", "FAIL" if wrong else "OK", f"changed or missing: {wrong}" if wrong else f"{len(exp)} files match SHA-256")
    cals = sorted((HOME / "models" / "calibration").glob("*.json"))
    add("Calibration profiles", "OK" if cals else "FAIL", ", ".join(p.stem for p in cals) or "none")
    reg = [p for p in (HOME / "models" / "registry").glob("*") if (p / "bundle.joblib").is_file()]
    add("Screening models", "OK" if reg else "WARN", f"{len(reg)} saved model bundle(s)" if reg else "none: train one in the app (Models page)")
    # libraries actually work
    try:
        import numpy as np

        from src import landmarks as L
        lm = L.make_landmarker(str(HOME / "models" / "hand_landmarker.task"), video=False)
        pts, _ = L.detect(lm, np.zeros((480, 360, 3), np.uint8))
        lm.close()
        add("MediaPipe hand model", "OK", "loads and runs (no hand on a blank image, as expected)" if pts is None else "loads and runs")
    except Exception as e:
        add("MediaPipe hand model", "FAIL", f"{type(e).__name__}: {e}")
    try:
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "t.mp4")
            w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 48))
            for _ in range(3):
                w.write(np.full((48, 64, 3), 100, np.uint8))
            w.release()
            cap = cv2.VideoCapture(p)
            ok, _ = cap.read()
            cap.release()
        add("Video decoding (OpenCV)", "OK" if ok else "FAIL", f"OpenCV {cv2.__version__}")
    except Exception as e:
        add("Video decoding (OpenCV)", "FAIL", str(e))
    try:
        from PySide6 import QtWidgets  # noqa: F401
        from PySide6.QtPdf import QPdfDocument  # noqa: F401
        add("Desktop GUI (Qt)", "OK", f"PySide6 {md.version('PySide6')}")
    except Exception as e:
        add("Desktop GUI (Qt)", "WARN", f"GUI unavailable ({e}); command-line tools still work")
    # writable folders and disk
    for d in ("data", "results", "logs", "models/registry", "models/calibration"):
        p = HOME / d
        try:
            p.mkdir(parents=True, exist_ok=True)
            t = p / ".write_test"
            t.write_text("x")
            t.unlink()
        except Exception as e:
            add(f"Write access {d}/", "FAIL", str(e))
    add("Write access", "OK", "data/, results/, logs/, models/ writable")
    free = shutil.disk_usage(HOME).free / 2**30
    add("Disk space", "OK" if free >= 5 else "WARN", f"{free:.0f} GB free (processing writes ~20 MB per recording)")
    # optional original dataset
    try:
        from src.io_utils import load_config
        raw = (load_config(HOME / "config.yaml").get("dataset_root") or "").strip()
        root = Path(raw) if raw else None
        if root is not None and root.is_dir():
            add("Original FLIROne dataset", "OK", str(root))
        else:
            add("Original FLIROne dataset", "WARN", ("not set" if root is None else f"{root} not found") +
                " — optional, only needed to re-run the original study (set RGBT_DATASET_ROOT or dataset_root in config.yaml); "
                "new datasets are imported in the app")
    except Exception as e:
        add("Configuration", "FAIL", f"config.yaml unreadable: {e}")
    return out


def main(argv=None) -> int:
    os.chdir(HOME)
    from src.__version__ import __version__
    print(f"RGB–Thermal Hand-Joint Analysis {__version__} — installation check ({HOME})\n")
    res = checks()
    w = max(len(n) for n, _, _ in res)
    for name, level, msg in res:
        print(f"  [{level:4s}] {name.ljust(w)}  {msg}")
    fails = sum(lv == "FAIL" for _, lv, _ in res)
    warns = sum(lv == "WARN" for _, lv, _ in res)
    print(f"\n{'READY' if not fails else 'NOT READY'}: {fails} failed, {warns} warning(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
