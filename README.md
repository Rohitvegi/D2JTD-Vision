# RGB–Thermal Hand Joint Analysis — project guide

**Deploying on another PC:** see [DEPLOYMENT.md](DEPLOYMENT.md). Build the release zip with
`.venv312\Scripts\python.exe tools\build_release.py`, then on the target PC unzip it and run `install.ps1`.

**On this development machine:**

| Task | Command |
| --- | --- |
| Desktop application | `.venv312\Scripts\python.exe -m src.cli app` |
| Installation check | `.venv312\Scripts\python.exe -m src.cli doctor` |
| Find / verify / apply the RGB→thermal calibration | `.venv312\Scripts\python.exe -m src.cli calibrate find …` (see `src/calibration.py`) |
| All tests | `.venv312\Scripts\python.exe -m src.cli test` |

| Version 1.0.0 additions | |
| --- | --- |
| v9 calibration | `src/calibration.py`: finds the image area, scale, rotation and per-frame shift from the recordings, applies them to points and images (and in reverse), and verifies the result. Evidence is in [reports/v9_calibration](reports/v9_calibration/README.md) |
| Deployment | `pyproject.toml`, `requirements.lock.txt` (exact versions), `install.ps1`, `RGB-Thermal App.bat`, `rgbthermal.bat`, health check `src/doctor.py`, log file `logs/rgbthermal.log`, model checksums `models/MODEL_SHA256.json` |

This project grows one stage at a time. Do not start a stage until the previous one ran and its checks passed.

| Stage | Status |
| --- | --- |
| 1. Dataset audit | Done — 68/68 complete; S11_left_static has RGB/thermal file names swapped (handled) |
| 2. Literature review | Done — [docs/stage2_literature_review.md](docs/stage2_literature_review.md) |
| 3. Learning objective | Done — [docs/stage3_learning_objective.md](docs/stage3_learning_objective.md) |
| 4–8. Environment, landmarks, masks, registration, ROIs | Done — `src/run_pipeline.py` |
| 9–11. Features, statistics, one-class test | Done — `src/analysis.py` |
| 12. Edge demo + benchmark | Done on laptop CPU — `src/edge_demo.py` |
| RQ2 annotation + joint correction | Done — 777 clicked joints; `src/landmark_correction.py` |
| **Results** | **[reports/final_report.md](reports/final_report.md)** |
| v4 audit | Done — [docs/v4_audit_and_plan.md](docs/v4_audit_and_plan.md); v3 frozen in `baselines/v3_baseline/` |
| v5 annotation QC + v7 unseen-subject models | Done — [reports/v5_v7_update.md](reports/v5_v7_update.md); tests in `tests/` |
| v6 features + v8 application (upload, validation, labels, independent test, reports, GUI, live simulation) | Done — [reports/v6_v8_update.md](reports/v6_v8_update.md) |

---

## Stage 1 — Dataset audit

### What it does, in plain words
Before building any model we must know exactly what is in the data. This script opens every
recording, checks that nothing is missing, broken or duplicated, measures the real frame rate,
reads the temperature arrays **without loading them fully into RAM**, checks whether the RGB video,
thermal video and temperature array line up frame by frame, and saves pictures you can inspect.

### Why this method
* `numpy.load(..., mmap_mode="r")` reads only the frames we ask for, so 1–2 GB files cost almost no RAM.
* The `.npy` header is read separately, so a truncated (corrupted) file is detected before it crashes anything.
* Reading every video timestamp catches variable frame rate and dropped frames, which the file header can hide.
* Correlating thermal-MP4 frame *i* with `.npy` frames *i−3 … i+3* tells us whether the streams are index-aligned.
* Files are fingerprinted (size + first/last 16 MB), so a recording saved twice under two names is found.

### 1. Create the project in VS Code (once)
1. Unzip this folder somewhere short, e.g. `C:\Projects\rgb_thermal_hand_screening`.
2. VS Code → **File › Open Folder…** → choose that folder.
3. **Terminal › New Terminal** (PowerShell opens at the project root). Then:

```powershell
python --version                       # needs 3.10, 3.11 or 3.12 (MediaPipe later needs <= 3.12)
python -m venv .venv
.\.venv\Scripts\Activate.ps1           # prompt now starts with (.venv)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. In VS Code press `Ctrl+Shift+P` → **Python: Select Interpreter** → pick `.venv`.

### 2. Point the project at your dataset
Open `config.yaml` and set `dataset_root` to the folder that contains S1 … S17:

```yaml
dataset_root: "C:/path/to/FLIROne"      # or set the environment variable RGBT_DATASET_ROOT
```

### 3. Run
Smoke test first (2 subjects, no timestamp scan, ~1 minute):

```powershell
python -m src.data_audit --limit 2 --quick
```

If that finishes without errors, run the full audit (reads every video frame's timestamp;
expect roughly 10–30 minutes depending on your disk):

```powershell
python -m src.data_audit
```

### 4. Expected output
```
reports/audit/
├── dataset_audit.csv                 # one row per recording (68 rows), ~60 columns
├── audit_report.md                   # summary tables, duplicates, issues
├── audit.log                         # everything printed in the terminal
├── dataset_temperature_histogram.png
└── figures/
    ├── S1_left_static_samples.png    # RGB | thermal MP4 | radiometric °C | outline overlay
    ├── S1_left_static_timestamps.png # frame intervals + RGB−thermal timestamp difference
    └── ...                           # 2 figures × 68 recordings
data/metadata/dataset_audit.csv       # copy used by later stages
```

The terminal prints one line per recording, e.g.
`[ 7/68] S10_left_static   14.2s  OK`.
Open `audit_report.md` in VS Code and press `Ctrl+Shift+V` for the formatted preview.

### 5. Checks — confirm each before Stage 2
- [ ] Report says **17 subjects** and **68 / 68** complete recordings.
- [ ] `Radiometric shape` line: expected `(N, H=640, W=480)` — an upright portrait image.
      Never transpose it: both MP4s show this same portrait image letterboxed (black bars) in a
      480×640 frame, inside the `content box` listed in the report (x = 140–500).
- [ ] `Frame counts equal` should be 68 of 68. Any mismatch is listed under issues.
- [ ] `Thermal MP4 ↔ .npy spatial match r` should be above 0.9 everywhere.
- [ ] `Thermal MP4 ↔ .npy frame alignment` should be all `aligned` (lag 0). `undetermined`
      only means the hand barely moved, so the motion signal could not pin the lag down.
- [ ] `Frames where thermal MP4 ≠ .npy` should be 0 (every frame is checked).
- [ ] `cv2 seek error` is expected to be non-zero: these videos have a variable frame rate, so
      `cap.set(CAP_PROP_POS_FRAMES, i)` lands on the wrong frame. Later stages must decode
      frames in order instead of seeking.
- [ ] Temperatures roughly 18–38 °C; NaN/inf count 0.
- [ ] Open 3–4 `*_samples.png`. Column 3 must look like a hand. Column 4 shows how far the
      thermal outline sits from the RGB hand (this offset is what Stage 5 corrects).
- [ ] **Duplicates section.** File sizes already suggest `S12 right hand` static and open-close
      may be the same recording. The audit confirms or rules this out.

Send me `audit_report.md`, `dataset_audit.csv` and two sample figures (one static, one open-close).
Stage 3 depends on what they show.

### 6. Common errors
| Error | Cause | Fix |
| --- | --- | --- |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| `ModuleNotFoundError: No module named 'src'` | Run from the wrong folder | `cd` to the project root (where `config.yaml` is) and use `python -m src.data_audit` |
| `FileNotFoundError: Dataset root not found` | Wrong path in `config.yaml` | Use forward slashes and end at the `FLIROne` folder |
| `ModuleNotFoundError: cv2` / `yaml` | venv not active | Activate `.venv`, rerun `pip install -r requirements.txt` |
| A recording shows `cannot open` | Corrupted or locked MP4 | Close any player using it; re-download that file |
| Very slow full run | Dataset on a slow / external disk | Run `--quick` first; the timestamp scan is the slow part |
| `MemoryError` | Should not happen (mmap) | Tell me the recording name from `audit.log` |

---

## Stages 4–12 — measurement pipeline (run after Stage 1)

Objective and pass criteria: [docs/stage3_learning_objective.md](docs/stage3_learning_objective.md).
Final write-up: [reports/final_report.md](reports/final_report.md).

### 4. Environment (once)
MediaPipe needs Python ≤ 3.12, so the pipeline uses its own environment:

```powershell
uv venv --python 3.12 .venv312
uv pip install --python .venv312\Scripts\python.exe -r requirements-pipeline.txt
```

The hand-landmark model (`models/hand_landmarker.task`, 7.8 MB) comes from Google's official
MediaPipe model storage:
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`

### 5–8. Landmarks → masks → registration → joint ROIs (~4 min with 6 workers)
```powershell
.venv312\Scripts\python.exe -m src.run_pipeline --workers 6
```
Writes `data/processed/frames/<rec>.csv` (one row per analysed frame), `registration.csv` and
`qc/<rec>.png` (look at a few: white circles must sit on the joints).

### 9–11. Features, reliability, reference ranges, open–close, one-class test
```powershell
.venv312\Scripts\python.exe -m src.analysis
```
Writes `reports/results/` (tables, figures, `summary.json`, `results.md`) and
`models/healthy_reference.json`.

### RQ2 manual check + joint correction
Click 6 joints on 136 reserved thermal frames (~35–45 min, guide: [docs/annotation_guide.md](docs/annotation_guide.md)), learn the along-finger correction, rerun the pipeline and analysis:
```powershell
.venv312\Scripts\python.exe -m src.annotate
```

```powershell
.venv312\Scripts\python.exe -m src.landmark_correction
```

Then run `src.run_pipeline` and `src.analysis` again (commands above).

### 12. Edge-style demo and benchmark
```powershell
.venv312\Scripts\python.exe -m src.edge_demo --rec S2_right_static --benchmark
.venv312\Scripts\python.exe -m src.edge_demo --rec S2_right_static --warm-joint middle_PIP
```

| Module | Stage | Job |
| --- | --- | --- |
| `src/io_utils.py` | 4 | recordings, in-order frame reader, letterbox geometry, file-swap fix |
| `src/landmarks.py` | 5 | MediaPipe landmarks + Qiao-style checks, 14 joints |
| `src/segmentation.py` | 6 | thermal (Otsu) and RGB (landmark-seeded GrabCut) hand masks |
| `src/registration.py` | 7 | per-recording RGB→thermal similarity transform, held-out Dice |
| `src/rois.py` | 8 | joint/dorsum ROIs, on-skin fraction, core-mask sampling |
| `src/run_pipeline.py` | 5–8 | runs all of the above per recording |
| `src/analysis.py` | 9–11 | features, RQ1–RQ7 statistics, figures |
| `src/annotate.py` | RQ2 | click tool for the registration ground truth |
| `src/landmark_correction.py` | RQ2 | learns/evaluates the along-finger joint correction (leave-one-subject-out) |
| `src/edge_demo.py` | 12 | fixed-transform live path, healthy-range flags, timing/RAM |
| `src/annotation_qc.py` | v5 | checks the hand clicks: consistency, three-cornered hat, contact sheet |
| `src/anomaly_models.py` | v7 | 6 anomaly models, nested leave-one-subject-out thresholds, learning curves, model registry |

### v6–v8 modules
| Module | Version | Job |
| --- | --- | --- |
| `src/feature_extraction.py` | v6 | 129 features per recording in 5 groups, stability, ablation |
| `src/dataset_manager.py` | v8 | ZIP/folder/single import, matching, calibration profiles, validation, processing new data |
| `src/evaluation.py` | v8 | labels, metrics, training wrappers, locked independent tests |
| `src/report_generator.py` | v8 | PDF / Excel / CSV reports |
| `src/joint_view.py` | v8 | RGB photo with the VISIBLE joints and their temperatures + thermal ROIs (also the app's *Joint view* page) |
| `src/landmarks.py` `joint_visibility` | v8 | marks joints hidden in the RGB image (folded under the hand, covered by another finger, end joint tucked under); hidden joints get no temperature in the pipeline (`data/processed_v8`) |
| `src/live_acquisition.py` | v8 | timestamp pairing, webcam source, simulation mode |
| `ui/app.py` | v8 | desktop application (PySide6) |
| `tests/` | — | 32 tests incl. an end-to-end run (`tests/make_fixture.py` builds its ZIP) |
