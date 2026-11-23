# Update — annotation quality (v5) and unseen-subject model evaluation (v7)

**Date:** 21 Nov 2026 · **Baseline:** `baselines/v3_baseline/` (frozen, unchanged) ·
**Code:** `src/annotation_qc.py`, `src/anomaly_models.py`, `tests/test_v7.py` (19 tests, all passing)

---

## 1. Are the hand-clicked annotations wrong?

**Short answer: mostly no.** They are internally consistent and carry no systematic offset, but they
are noisy, and two frames contain labelling mistakes. The low joint-position score is mostly the
pipeline's along-finger offset (already corrected) plus the limits of clicking blurry thermal knuckles.

| Check | Result |
| --- | --- |
| Internal consistency (order along finger, bone length, finger identity) | 133 of 136 frames pass |
| Confirmed labelling mistakes (checked against the RGB photo) | **S5_left_static f675** (MCP/PIP swapped), **S7_left_static f75** (index ↔ ring reversed) |
| Borderline | S11_right_open_close f921 (MCP–PIP length 1.54× on a bent finger; may be correct) |
| RQ2 without the 3 flagged frames | 10.34 → **10.07 px** (corrected, leave-one-subject-out): the mistakes are not the main cause |
| Systematic offset (median vector between estimators) | clicks vs thermal MediaPipe ≈ (−0.6, −1.6) px; RGB-mapped vs both ≈ 9–11 px → **the offset is the pipeline's** |
| Three-cornered hat, random scatter (bias removed) | clicks ≈ **17.6 px RMS**, RGB-mapped ≈ **6.7 px**, thermal MediaPipe ≈ 16.9 px |

The clicking estimate is an **upper bound**: the RGB-mapped and thermal-MediaPipe estimates share the
same MediaPipe model, which shrinks their mutual gap and inflates the share attributed to the clicks.
A second annotation pass on ~20–40 frames is the only direct measurement of clicking error.

**What to do:** re-click the two confirmed frames (≈1 minute), then rerun the correction and analysis:

```powershell
.venv312\Scripts\python.exe -m src.annotate --redo S5_left_static:675 S7_left_static:75
```

Visual check of flagged and worst frames: `reports/v5_annotation_qc/contact_sheet.png`.

---

## 2. Performing well on unseen data (v7)

v3 chose the anomaly threshold on the same hands it was evaluated on. v7 fixes this: every learned
quantity — imputation, scaling, covariance, model **and threshold** — comes from training subjects only
(outer leave-one-subject-out; threshold from an inner leave-one-subject-out at the 95th percentile).
Anomalies are simulated (healthy volunteers only).

| Model (nested LOSO) | Specificity (95 % CI) | All 10 MCP/PIP +1 °C | Single joint +2 °C | Scoring cost |
| --- | --- | --- | --- | --- |
| Robust covariance (MCD) — **selected by the pre-declared rule** | 0.94 (0.85–1.00) | 0.29 | **0.80** | 0.2 ms/hand |
| One-class SVM | 0.94 (0.82–1.00) | **0.65** | 0.26 | 0.2 ms |
| Mahalanobis, Ledoit–Wolf shrinkage — **stable alternative** | 0.94 (0.82–1.00) | 0.38 | 0.41 | 0.2 ms |
| Robust z-score (median/MAD) | 0.94 (0.82–1.00) | 0.09 | 0.19 | 0.008 ms |
| Mahalanobis, diagonal | 0.94 (0.82–1.00) | 0.09 | 0.06 | 0.008 ms |
| Isolation Forest | 0.94 (0.82–1.00) | 0.03 | 0.06 | 6.9 ms |

**Compared with v3 (in-sample thresholds):** v3 reported Mahalanobis specificity 0.88 (target 0.95) and
sensitivities of 0.71–1.00 for "all 10 joints +1 °C". With honest thresholds specificity reaches its
target, but sensitivity at 1 °C is **0.03–0.65**: the v3 sensitivities were optimistic.

**All 10 joints +2 °C:** 1.00 for MCD, SVM and Ledoit–Wolf. **Diffuse whole-hand warming** is never
detected (0.06 = false-positive rate): D2JTD is relative to the back of the hand, so it ignores warming
of the whole hand by design (the same property that makes it robust to room temperature).

### Learning curves — would more subjects help?
| Training subjects | 4 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- |
| MCD specificity / sens. (all 10 +1 °C) | 0.74 / 0.77 | 0.83 / 0.57 | 0.99 / 0.03 | 0.94 / 0.29 |
| Ledoit–Wolf | 0.89 / 0.47 | 0.88 / 0.54 | 0.92 / 0.39 | 0.94 / 0.38 |
| One-class SVM | 0.81 / 0.76 | 0.88 / 0.63 | 0.92 / 0.47 | 0.94 / 0.65 |

MCD is **unstable below ~12 subjects** (more training hands than parameters are needed); Ledoit–Wolf
keeps its specificity at every size. Recommendation: use the Ledoit–Wolf bundle whenever a model is
retrained on few subjects; collect more subjects before trusting fine sensitivity differences.

### Out-of-range check
Each saved model stores the training range of every feature (+25 % margin). On held-out real hands
6 % were flagged "outside model experience" — an estimate of how often a new healthy hand would be
reported as unfamiliar rather than scored.

### Saved models (versioned, never overwritten)
`models/registry/v7_robust_covariance_mcd_<time>/` and `models/registry/v7_mahalanobis_ledoit_wolf_<time>/`,
each with `bundle.joblib` (imputer + model + threshold + feature order) and `metadata.json` (training
subjects, features, threshold method, validation results, software versions, config hash, limitations).

### Not done yet (from the plan)
Pixel-level warming with different hotspot sizes (needs the per-frame pipeline; the experiments above
inject at feature level), feature ablation (v6), dataset upload, independent-test workflow and GUI (v8).

## 3. Reproduce

```powershell
.venv312\Scripts\python.exe -m src.annotation_qc
```

```powershell
.venv312\Scripts\python.exe -m src.anomaly_models
```

```powershell
.venv312\Scripts\python.exe -m pytest -q
```
