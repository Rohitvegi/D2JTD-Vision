# RGB–Thermal Hand-Joint Analysis — deployment guide (v1.0.0)

Research and screening-support software. It measures hand-joint temperatures from paired RGB and radiometric
thermal recordings and compares hands with a healthy reference. **It does not diagnose arthritis.**

## 1. Requirements
| Item | Requirement |
|---|---|
| OS | Windows 10/11, 64-bit |
| Python | 3.12 (installed automatically by `uv`, or the Python 3.12 'py' launcher) |
| Disk | ~2 GB for the installation; ~20 MB per processed recording |
| RAM | 4 GB minimum (radiometric files are memory-mapped, not loaded) |
| GPU | not needed |

## 2. Install
1. Unzip `rgb_thermal_hand-1.0.0.zip` to a short path, e.g. `C:\RGBThermal`.
2. Open PowerShell in that folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Options: `-Offline` installs only from the local `uv` cache (no download), and `-Shortcut` adds a desktop
shortcut. The installer creates `.venv`, installs the exact versions in `requirements.lock.txt` and runs
the health check. It must end with **READY**.

## 3. Run
| What | How |
|---|---|
| Desktop application | double-click **`RGB-Thermal App.bat`** |
| Health check | `rgbthermal.bat doctor` |
| Find the camera calibration from new recordings | `rgbthermal.bat calibrate find --rgb a.mp4 --npy a.npy [--rgb ... --npy ...] --name my_camera` |
| Check a calibration on another recording | `rgbthermal.bat calibrate verify --profile models\calibration\my_camera.json --rgb x.mp4 --npy x.npy` |
| Map one frame's joints into the thermal image | `rgbthermal.bat calibrate apply --profile ... --rgb x.mp4 --npy x.npy --frame 300` |
| Automated tests | `rgbthermal.bat test` |
| Version | `rgbthermal.bat version` |

Typical workflow in the application:
1. **Dataset upload**: import a ZIP or folder (or a single recording) → **Validate**.
2. **Calibration**: if validation says *Requires calibration* (a different camera or export), use
   **Auto-calibrate from this dataset**. A new profile is saved and its self-check is shown.
3. **Processing**, then **Independent test** with a saved model, then **Joint view** and **Reports**
   (PDF/Excel).

## 4. Data and configuration
* Input per recording: RGB video (`.mp4/.avi/.mov`), radiometric temperatures as `.npy` (frames × H × W, float, °C),
  optional thermal display video. File-name patterns are set in the upload page.
* The original FLIROne study dataset is **not shipped**. To re-run the original study, set `dataset_root` in
  `config.yaml` or the environment variable `RGBT_DATASET_ROOT`.
* Results are written inside the installation folder: `data/datasets/` (imported datasets and processing),
  `results/` (test runs), `models/` (calibrations, model bundles) and `logs/rgbthermal.log`.
* Calibration and model files are versioned and never overwritten. `models/MODEL_SHA256.json` lets the health
  check detect changed model files.

## 5. Calibration (how a new camera is handled)
`src/calibration.py` finds the RGB→thermal mapping from the recordings themselves, with no calibration
target. It determines:
* the image area (letterbox removed),
* the scale,
* the in-plane rotation,
* a per-frame shift learned from the hand,

and checks the result on frames not used for fitting (grade good / acceptable / poor). A radiometric export
that is rotated or transposed relative to the video is reported, not silently mis-mapped. Evidence:
`reports/v9_calibration/`.

## 6. Updating and uninstalling
* **Update:** unzip the new version into a new folder, run `install.ps1`, then copy over `data/`, `results/`,
  `models/calibration/` and `models/registry/` from the old folder.
* **Uninstall:** delete the folder (and the desktop shortcut if you created one). Nothing is written elsewhere.

## 7. Troubleshooting
| Symptom | Fix |
|---|---|
| `install.ps1` cannot be run | use `powershell -ExecutionPolicy Bypass -File .\install.ps1` |
| doctor: Python FAIL | MediaPipe needs Python 3.12; install `uv` and rerun the installer |
| doctor: Model integrity FAIL | a model file was changed or lost; re-copy `models/` from the release zip |
| validation: *Requires calibration* | Calibration page → Auto-calibrate from this dataset |
| calibration grade *poor* | the RGB and radiometric files probably do not belong together or are not frame-synchronised |
| anything else | see `logs/rgbthermal.log` |

## 8. Scope (read before use)
The models were built from 17 healthy volunteers. Outputs ("within healthy range", "potential anomaly") are
distances from that healthy reference, not diagnoses. Sensitivity to real arthritis has not been measured.
