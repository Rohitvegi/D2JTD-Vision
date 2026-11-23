# Stage 3 — Learning objective

**Status:** agreed 21 Nov 2026 (proposed in Stage 2 review §5, accepted).
**Depends on:** Stage 1 audit (data are complete and index-aligned), Stage 2 review (evidence and gaps).

---

## 1. Objective

> Build and validate an automated pipeline that extracts per-joint **relative** temperature features
> from smartphone RGB–thermal hand video, quantify its **geometric accuracy** and **measurement
> repeatability**, and establish **healthy reference ranges** for those features.

In plain words: before anyone can screen for a "warm joint", we must show that the system finds the
right joint in the thermal image, measures it consistently, and knows what a normal hand looks like.
This dataset (17 volunteers, no clinical labels) can answer exactly those questions.

### What this project will NOT claim
* No arthritis or inflammation classification performance — there are no labelled patients.
* No comparison of absolute °C between people — room conditions were not recorded (Stage 2 §3.2).
* No diagnosis wording anywhere. The tool is described as a measurement / screening-aid pipeline.

---

## 2. Research questions, outcomes and pass criteria

Units: **joint** = one of 14 regions (thumb MCP, IP; MCP/PIP/DIP of fingers 2–5), reported also for
Qiao et al.'s 10 (MCP + PIP) for comparability. **Dorsum** = back-of-hand region bounded by landmarks
0, 1, 2, 5, 9, 13, 17. Temperatures always come from the radiometric `.npy`.

| # | Question | Primary outcome | Pass criterion (set now, before looking) |
| --- | --- | --- | --- |
| RQ1 | Can joints be located reliably in the RGB video? | % frames with 21 landmarks that pass the anatomical checks (finger order, MCP collapse, inside hand mask), per task | ≥ 90 % of *static* frames; *open–close* reported, no threshold (fists hide joints) |
| RQ2 | How accurately are RGB joint positions mapped onto the thermal frame? | Landmark error (px in the 640×480 `.npy` frame and in native sensor pixels) on hand-annotated validation frames; hand-mask Dice after alignment | Median error ≤ 8 px (≈ 2 native pixels); Dice ≥ 0.90 (Qiao et al. benchmark) |
| RQ3 | Do joint ROIs measure skin, not background? | % ROIs fully inside the thermal hand mask; native pixels per ROI | ≥ 95 % of ROIs on valid frames; report pixel counts |
| RQ4 | How repeatable are the features within a recording? | Within-recording SD; ICC(3,1) between first and second half of each static recording; SEM and MDC95 = 1.96·√2·SEM | MDC95 of joint D2JTD < 0.4 °C (the healthy-vs-RA gap reported by Qiao et al.) |
| RQ5 | What is the healthy reference range? | Per-joint D2JTD mean ± SD and 95 % bootstrap CI across subjects; L−R asymmetry \|ΔT\| | Reported; compared with Qiao healthy 1.5 ± 0.4 °C and the < 0.5 °C L−R norm |
| RQ6 | Does the open–close task change joint temperatures? | Slope of joint and dorsum temperature (°C/min) and of D2JTD; static vs open–close, paired per hand | Reported with effect size and Wilcoxon signed-rank p (n = 34 hands) |
| RQ7 *(optional)* | Could a healthy-only model flag an abnormal joint? | One-class model trained leave-one-subject-out; sensitivity at 95 % specificity for **synthetic** +0.5 / 1 / 2 °C added to one joint ROI | Reported; explicitly labelled as a method demonstration, not clinical performance |

**Hypotheses** (from the literature, tested in RQ5): H1 — in healthy hands D2JTD > 0 for MCP and PIP
joints (joints cooler than dorsum). H2 — for most joints the healthy L−R difference is below 0.5 °C.

**Engineering outcomes** (edge requirement, measured at the end): per-frame latency, RAM, model size
on a CPU-only laptop, then on the target edge board if available.

---

## 3. Data use rules

* All 68 recordings are used. Unit of independence = **subject** (17).
* Any tuned component (thresholds, ROI scale, one-class model) is tuned with **leave-one-subject-out**
  cross-validation; nothing from the held-out subject is used for tuning.
* Validation annotations for RQ2: 2 frames per recording (one early, one late), 6 joints each
  (index/middle/ring MCP and PIP) clicked on the thermal frame — 136 frames. Annotated before
  the registration method is tuned, and stored in `data/annotations/`.
* Video frames are read **in order** (Stage 1 finding); RGB frame *i* ↔ `.npy` frame *i*.
* Frames failing a quality check are **dropped and counted**, never repaired.
* Temporal features use real time (frame index ÷ measured fps per recording), not frame counts.

---

## 4. Stage plan (4–12)

| Stage | Module | Output | Feeds |
| --- | --- | --- | --- |
| 4 | Environment + shared I/O | Python 3.12 venv with MediaPipe; `src/io.py` in-order reader, letterbox mapping | all |
| 5 | Hand landmarks (RGB) + checks | per-frame 21 landmarks, check flags | RQ1 |
| 6 | Hand segmentation (RGB and thermal masks) | per-frame masks | RQ2, RQ3 |
| 7 | RGB → thermal registration + annotation tool | per-recording transform, error report | RQ2 |
| 8 | Joint and dorsum ROIs + quality flags | per-frame ROI temperatures, validity | RQ3 |
| 9 | Feature extraction | D2JTD, L−R, HDI, temporal slopes per joint | RQ4–6 |
| 10 | Reliability and reference statistics | ICC, SEM, MDC, reference table | RQ4, RQ5, RQ6 |
| 11 | Optional one-class + synthetic test | sensitivity curves | RQ7 |
| 12 | Edge benchmark + final report | latency/RAM/size, write-up | engineering |

Each stage keeps the Stage 1 rule: it runs, its checks pass, then the next one starts.

---

## 5. Risks and how they are handled

| Risk | Effect | Mitigation |
| --- | --- | --- |
| MediaPipe fails on fists / dark backgrounds | RQ1 low for open–close | Report per task; use static recordings for the reference ranges |
| Thermal hand larger and offset vs RGB, varying with distance | Joint ROIs land off-joint | Per-recording affine from masks (Stage 7), checked against annotations |
| Low native thermal resolution (160 × 120 sensor) | ROIs contain few real pixels | Report native pixel count; average across frames |
| Camera random error ~0.3 °C | Noisy features | Frame averaging; RQ4 quantifies what remains |
| Only 17 subjects | Wide reference intervals | Bootstrap CIs; state that ranges are preliminary |
| No environment data | Absolute temperature not comparable | Relative features only; absolute as covariate |
