"""Build a small 'new dataset' ZIP for end-to-end tests (software test only).

Takes the first N frames of three FLIROne recordings and renames them with a different naming
convention (P16_L_static_rgb.mp4 ...). The people are the same as in the training data under new
IDs, so results from this fixture test the SOFTWARE, not generalisation. Also adds an unrelated
video and a text file (must be reported as unmatched / ignored) and a DUMMY label file.

Usage:  .venv312\\Scripts\\python.exe -m tests.make_fixture
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

def _dataset_root() -> Path:
    from src.io_utils import load_config        # config.yaml dataset_root, or RGBT_DATASET_ROOT
    return Path(load_config(Path("config.yaml"))["dataset_root"])


SRC = _dataset_root()
OUT = Path("data/test_fixtures")
N = 45
PICK = [("S16", "left", "static", "P16", "L"), ("S16", "right", "static", "P16", "R"), ("S17", "left", "static", "P17", "L")]


def cut_video(src: Path, dst: Path, n: int):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        vw.write(f)
    vw.release()
    cap.release()


def main():
    work = OUT / "mini_dataset"
    work.mkdir(parents=True, exist_ok=True)
    for s, hand, task, new, h in PICK:
        d = SRC / s / f"{hand} hand"
        stem = f"{new}_{h}_{task}"
        cut_video(d / f"{task}_RGB.mp4", work / f"{stem}_rgb.mp4", N)
        cut_video(d / f"{task}_Thermal.mp4", work / f"{stem}_thermal.mp4", N)
        np.save(work / f"{stem}_radiometric.npy", np.asarray(np.load(d / f"{task}_Thermal_Radiometric_Sequence.npy", mmap_mode="r")[:N]))
    cut_video(SRC / "S16" / "left hand" / "static_RGB.mp4", work / "random_video.mp4", 5)
    (work / "notes.txt").write_text("not a recording")
    with zipfile.ZipFile(OUT / "mini_dataset.zip", "w", zipfile.ZIP_STORED) as z:
        for p in sorted(work.iterdir()):
            z.write(p, f"mini_dataset/{p.name}")
    pd.DataFrame([dict(subject="P16", label="healthy", source="DUMMY - software test only"),
                  dict(subject="P17", label="arthritis", source="DUMMY - software test only"),
                  dict(subject="P99", label="healthy", source="DUMMY - unmatched on purpose")]).to_csv(
        OUT / "labels_DUMMY_software_test.csv", index=False)
    print("fixture:", OUT / "mini_dataset.zip", round((OUT / "mini_dataset.zip").stat().st_size / 2**20, 1), "MB")


if __name__ == "__main__":
    main()
