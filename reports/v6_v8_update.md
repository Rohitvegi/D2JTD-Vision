# Update — v6 features and v8 application (dataset upload, independent testing, reports, GUI)

**Date:** 21 Nov 2026 · **Baseline:** `baselines/v3_baseline/` (unchanged) · **Tests:** 32 passing (`tests/`)

---

## v6 — features and which ones help

New per-ROI statistics (pipeline run `data/processed_v6/`, v3 data untouched): mean, median, 90th
percentile, SD, valid-pixel share, whole-hand temperature, invalid-pixel count, and a **cold-band
contamination check** (share of ROI pixels > 1.5 °C below the ROI median — a metal ring has low
emissivity and reads cold). Hot pixels are never removed or flagged.

`src/feature_extraction.py` computes 129 features per recording in five groups. Stability (split-half,
static recordings, median over the group):

| Group | Features | ICC(3,1) | MDC95 |
| --- | --- | --- | --- |
| relative — D2JTD | 14 | 0.98 | 0.42 °C |
| relative — vs whole hand | 28 | 0.96 | 0.48 °C |
| spatial (MCP−PIP, PIP−DIP, neighbouring knuckles, finger gradient, palm-to-finger) | 18 | 0.97 | 0.29 °C |
| absolute (median, mean, P90, spread, dorsum, hand) | 58 | 0.97 | 0.90 °C |
| temporal (trends, frame variation) | 4 | **0.40** | — |

Ablation (Ledoit–Wolf model, nested leave-one-subject-out, simulated warming):

| Feature set | n | Specificity | Single joint +2 °C | All 10 joints +1 °C |
| --- | --- | --- | --- | --- |
| D2JTD only (v3) | 14 | 0.94 | 0.42 | 0.35 |
| **D2JTD + spatial (selected)** | 32 | 0.94 | **0.94** | **0.56** |
| D2JTD + temporal | 18 | 0.94 | 0.50 | 0.41 |
| D2JTD + absolute | 72 | 0.94 | 0.41 | 0.29 |
| all groups | 122 | 0.94 | 0.45 | 0.12 |

* Spatial contrast is the most useful addition and is as stable as D2JTD.
* **More features is worse:** 122 features on 34 hands collapses sensitivity — the overfitting effect.
* Temporal features are unreliable (ICC 0.40) and were not adopted.
* Caveat: the simulation warms one ROI uniformly — exactly the local contrast spatial features
  measure — so this gain is an upper estimate until patients are measured.

Saved model: `models/registry/v7_mahalanobis_ledoit_wolf_20261121_172656` (D2JTD + spatial).

## v8 — application layer

| Module | What it does | Tested |
| --- | --- | --- |
| `src/dataset_manager.py` | ZIP import with safe extraction (path traversal, links, zip bombs, disk space refused), folder import in place (no copies), single-recording upload; filename profiles (FLIROne + generic `P01_L_static_rgb.mp4`, custom regex); manual matching; duplicate detection; versioned calibration profiles; validation → Valid / Valid with warnings / Invalid / Requires calibration / Requires manual file matching, each check explained | yes |
| `src/evaluation.py` | labels from CSV/Excel at subject / hand / recording / joint level with source; unmatched labels and recordings listed; classification metrics with single-class and small-sample guards; measurement metrics (MAE, RMSE, median, P95, signed); healthy-reference training; supervised LOSO only when both classes exist; **locked independent test** (inputs SHA-256-fingerprinted, model applied as saved, own folder per run, training-subject overlap check) | yes |
| `src/report_generator.py` | PDF (with sample counts and limitations), Excel and CSV for datasets and test runs | yes |
| `src/live_acquisition.py` | timestamp pairing buffer with thermal-frame age; webcam RGB source; FLIR ONE stub with explanation (no desktop SDK); **clearly labelled simulation** | yes |
| `ui/app.py` | PySide6 desktop app: Dashboard, Dataset upload, Calibration, Processing, Models, Independent test, Explainability, Live (simulation), Reports | builds and switches all pages |
| `src/run_pipeline.py` | `--processed-dir` for versioned runs, run manifest (config, code hash, package versions), fixed OpenCV seed | **bit-identical reruns verified** |

### End-to-end check (software test)
A 3-recording ZIP with a different naming convention, an unrelated video and a dummy label file
went through import → validation → processing → independent test → PDF/Excel reports in 27 s:
3/3 matched and Valid, the stray video listed as unmatched, registration Dice 0.83–0.88, all three
"Normal measurement", metrics computed with explicit counts, the deliberately unmatched label row
reported. The fixture re-uses training people under new IDs (P16 = S16), which the ID-based overlap
check cannot detect — it tests the software, not generalisation.

### Live mode (simulation)
Replaying S2_right_static on two clocks (RGB at the recording rate, thermal at 8.7 Hz): 173/173 RGB
frames paired, 66 % with exactly the right thermal frame and 100 % within one frame, median thermal
age 8 ms (max 127 ms), ~17 frames/s measured. Live FLIR ONE streaming was not possible (mobile SDK only).

## Still not validated (needs inputs from outside this project)
Clinical accuracy (patients), stereo calibration (calibration recording), inter-annotator error
(second annotation pass), between-day repeatability (repeat sessions), edge-device performance
(edge board), real accessories (recordings with rings), live hardware (camera + SDK).

## How to use

```powershell
.venv312\Scripts\python.exe -m ui.app
```

```powershell
.venv312\Scripts\python.exe -m pytest -q
```

## Later addition — joint visibility (data/processed_v8)

Joints that are not visible in the RGB image are no longer measured or displayed. A joint is hidden
when it lies inside the back-of-hand outline (folded under the hand), when another finger covers it
and is closer to the camera (MediaPipe relative depth), or when its middle joint is bent > 100°
(end joint tucked under). MediaPipe still outputs positions for such joints, but they are guesses, so
their ROI would measure other skin. Hidden joints get no temperature in the per-frame data
(`{joint}_vis` = 0) and the joint view draws only visible joints.

Over all 68 recordings (`data/processed_v8`): static (open hand) 0 % of frames with any hidden joint;
open–close 66 % of frames, end joints hidden 40 %, middle joints 5 %, thumb IP 7 %, knuckles 0 %.
Hidden-joint count correlates 0.93 with hand closing. Not validated against hand-labelled visibility.

The v6 feature/ablation results and the saved models used static recordings only, where nothing is
hidden, so they are unaffected.
