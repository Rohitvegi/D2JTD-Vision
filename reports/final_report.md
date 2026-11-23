# Final report — RGB–thermal hand-joint measurement pipeline

**Data:** FLIROne dataset, 17 healthy volunteers × 2 hands × 2 tasks = 68 recordings (FLIR ONE Pro)
**Objective (Stage 3):** build and validate an automated pipeline that extracts per-joint *relative*
temperature features from smartphone RGB–thermal video, quantify its geometric accuracy and
repeatability, and establish healthy reference ranges.
**Not claimed:** any arthritis/inflammation detection performance — the dataset has no patients and no labels.

Numbers come from `reports/results/` (final version v3: finger-width ROIs + along-finger joint
correction learned from the hand-clicked joints). Earlier versions are kept for comparison:
`reports/results_v1_qiao_radius/` (v1) and `reports/results_v2_before_correction/` (v2).
Pass criteria were fixed in `docs/stage3_learning_objective.md` before any analysis.

> **Later updates:** annotation quality check and leakage-free model evaluation are in [v5_v7_update.md](v5_v7_update.md); features, the desktop application and the independent-test workflow in [v6_v8_update.md](v6_v8_update.md). Its v7 numbers supersede the RQ7 sensitivities below, which used in-sample thresholds and are optimistic. Erratum: open–close landmark validity is 99.57 %, not 99.7 % (see docs/v4_audit_and_plan.md).

---

## 1. Scorecard (v3)

| # | Question | Result | Criterion | Verdict |
| --- | --- | --- | --- | --- |
| RQ1 | Joints located in RGB | 100 % of static frames valid (open–close 99.7 %) | ≥ 90 % static | **Pass** |
| RQ2a | RGB → thermal outline overlap | held-out hand-mask Dice median **0.911** | Dice ≥ 0.90 | **Pass** |
| RQ2b | Joint position vs hand-clicked joints (777 points, 136 frames) | raw 14.3 px → **10.3 px** after correction (leave-one-subject-out); static 8.8 px, open–close 13.3 px | median ≤ 8 px | **Fail** (static near) |
| RQ2c | *Added:* temperature error caused by joint misplacement | median **0.048 °C** (P90 0.23 °C) | — | small vs 0.26 °C MDC |
| RQ3 | ROIs on skin | **98 %** of ROIs ≥ 95 % on the hand (v1: 48 %) | ≥ 95 % | **Pass** (partly by construction) |
| RQ4 | Repeatability (10-joint mean D2JTD) | ICC 0.99, **MDC95 0.26 °C**; 6 of 14 single joints < 0.4 °C | MDC95 < 0.4 °C | **Pass** (mean); MCPs per joint |
| RQ5 | Healthy reference | mean D2JTD **0.39 °C** (95 % CI 0.10–0.76); H1 holds for 3/10 joints; H2 not supported | reported | see §3 |
| RQ6 | Open–close effect | warmer absolute temps (+0.18 °C dorsum, +0.31 °C joints), **no change in D2JTD or trends** | reported | see §3 |
| RQ7 | Healthy-only model (synthetic) | +1 °C on all 10 MCP/PIP: sensitivity 0.71 (max\|z\|, spec 0.94) / 1.00 (Mahalanobis, spec 0.88) | reported | method demo only |
| — | RGB ↔ `.npy` sync (open–close) | 32/34 aligned at lag 0 (median r 0.84) | — | OK |
| — | Edge speed (laptop CPU) | 29 ms/frame ≈ 34 fps vs 8.7 fps sensor; 7.5 MB model | real-time | OK (laptop, not edge board) |

---

## 2. Pipeline as built

```
RGB MP4 ──decode in order──► crop letterbox ──► MediaPipe Hand Landmarker (VIDEO mode, 21 pts)
                                                    │ Qiao-style checks (finger order, MCP collapse, in-image)
.npy (°C, 640×480) ──► thermal hand mask (Otsu seed + hysteresis) ─┐
                                                    ▼              │
             RGB→thermal similarity: fixed scale 1.25, per-recording shift (fallback: rig median)
                                                    ▼
             along-finger joint correction (MCP +9.3 %, PIP/IP/DIP +10.7 % of bone length)
                                                    ▼
             14 joint ROIs: r = min(β·|MCP−PIP|, 0.9 × distance to nearer finger edge)
             + dorsum polygon; temperatures sampled on the eroded hand mask
                                                    ▼
             per-frame joint/dorsum °C ──► D2JTD = dorsum − joint ──► per-recording medians, trends
```

Stage 1 findings built in: `.npy` is upright portrait (no transpose); MP4s letterboxed at x 140–500;
frames decoded in order (OpenCV seeking was up to ±30 frames off); `S11_left_static` has its RGB and
thermal MP4 file names swapped on disk (corrected; the audit now detects this automatically).

### Deviations from the Stage 3 plan (made after seeing intermediate output; all in `config.yaml`)
1. **ROI sampling** from the hand mask eroded by ~1 sensor pixel; joint used if ≥ 60 % of its circle is on the hand.
2. **Thermal mask hysteresis** — subject S17's cold fingers were cut off by plain Otsu.
3. **Fixed registration scale = 1.25** (free fit over 68 recordings: median 1.25, IQR 1.24–1.26);
   per-recording shift, falling back to the rig median when that fits held-out frames better.
4. **Open-hand filter** (openness ≥ 1.88) only for open–close recordings.
5. **ROI radius capped at the finger width** (v2) — after RQ3 failed with Qiao et al.'s rule (v1).
6. **Along-finger joint correction** (v3) — learned from the hand-clicked joints after RQ2b failed;
   evaluated leave-one-subject-out. Raw (uncorrected) positions stay in the frame tables so RQ2 is
   never scored on the data the correction was learned from.

---

## 3. Results in detail

### RQ1 — landmarks
MediaPipe found a hand in 99.9 % of frames. The failure modes reported by Qiao et al. on FLIR ONE
images (finger-order reversal, MCP collapse) never occurred in static recordings and in 0.09 % of
open–close frames. The checks do not prove landmarks sit on the true joints — that is RQ2b.

### RQ2 — geometric accuracy
**Outline overlap (RQ2a, pass).** Held-out hand-mask Dice median 0.911; the RGB→thermal transform is
nearly constant across the dataset (shift IQR ≈ 0.5 px horizontal, 1 px vertical).

**Joint positions (RQ2b, fail).** 777 hand-clicked joints (index/middle/ring MCP and PIP) on 136
reserved thermal frames.

| Step | Median error | P90 |
| --- | --- | --- |
| Raw mapped RGB joints | 14.3 px (3.6 sensor px) | 26.4 px |
| + along-finger correction (leave-one-subject-out) | **10.3 px (2.6 sensor px)** | 23.7 px |
| ... static recordings only | 8.8 px | — |
| ... open–close recordings only | 13.3 px | — |

How the error was diagnosed (scratch experiments, summarised):
* The error is a **shift along the finger**: clicked joints lie ~10.6 px (≈ 10 % of a bone length)
  towards the fingertip of the mapped joints; across the finger only 2 px. Within a frame all six
  joints share most of it (common offset 11.8 px; residual 6.6 px).
* It is **not fixed by better outline registration**: excluding cold fingertips, the forearm, or using
  a looser thermal threshold left the bias at 12–16 px, and per-frame refits were worse than one fit
  per recording — sliding along a finger barely changes outline overlap.
* It is **not a timing offset** (neighbouring RGB frames did not match better) and only weakly related
  to hand size / camera distance (r = 0.23).
* MediaPipe run directly on the thermal image found the hand in 90 % of frames but was not more
  accurate (13 px), and adding its per-frame shift did not help (10.1 px).
* Remaining error is largest when fingers flex (open–close). Part of it is clicking error, which
  could not be measured (one annotator, one pass).

**What the error costs (RQ2c, added).** Reading each joint ROI at the pipeline position vs at the
clicked position (same radius, same pixels) changes the joint temperature by a median of
**0.048 °C** after correction (0.067 °C before; P90 0.23 °C). Because ROIs (radius ≈ 26 px) are larger
than the error and finger temperature varies slowly, the miss matters little for the measured
values — it is far below the 0.26 °C repeatability limit (RQ4).

### RQ3 — ROIs on skin
| | v1 Qiao radius | v3 final |
| --- | --- | --- |
| ROIs ≥ 95 % on hand | 48 % | **98 %** |
| ROIs yielding a value | 99.3 % | 99.7 % |
| sensor pixels per ROI (median) | 142 | 76 |

Caveat: the radius is derived from the same hand mask that RQ3 checks, so the pass is partly by
construction; v3 shows ROIs can stay on the finger while keeping ≥ ~40 sensor pixels.

### RQ4 — repeatability (static recordings, first half vs second half)
| Feature | ICC(3,1) | MDC95 (°C) |
| --- | --- | --- |
| 10-joint mean D2JTD | 0.99 | **0.26** |
| MCP joints (5) | 0.97–0.98 | 0.20–0.27 |
| Thumb IP | 0.98 | 0.38 |
| PIP joints (4) | 0.98 | 0.41–0.47 |
| DIP joints (4) | 0.98–0.99 | 0.47–0.57 |

The 10-joint mean resolves the 0.4 °C healthy-vs-RA gap reported by Qiao et al.; MCPs individually do
too; PIPs are borderline, DIPs are not. Within-session only — between-day test–retest is not
measurable with this dataset.

### RQ5 — healthy reference (static, n = 17 subjects; `fig_reference_d2jtd.png`)
* 10-joint mean D2JTD **0.39 ± 0.73 °C** (95 % CI 0.10–0.76). Qiao et al.'s healthy 1.5 ± 0.4 °C lies
  outside this CI: joints are cooler than the back of the hand, but by much less than reported there.
  **Reference ranges are device-and-protocol specific.**
* **H1** (D2JTD > 0, Holm-corrected): supported for 3 of 10 joints — index MCP, little MCP, little PIP.
  Spread is large at PIP/DIP (SD ≈ 1.0–1.4 °C); MCPs are tight (SD 0.39–0.52 °C).
* **H2** (left − right absolute joint temperature < 0.5 °C): **not supported** — median |ΔT| 0.95–1.74 °C,
  14 % of pairs < 0.5 °C; left and right were recorded separately. The relative feature is far more
  symmetric (median |ΔD2JTD| 0.19–0.34 °C at thumb/index/middle/ring MCPs).

### RQ6 — open–close vs static (34 hands; `fig_open_close.png`)
* Absolute temperatures higher in open–close: dorsum +0.18 °C (p = 0.011; subject-level 0.013), joint
  mean +0.31 °C (p = 0.002; 0.003).
* **D2JTD level and all temperature trends did not differ** (all p > 0.10).
* Hands were warming during both tasks (~+0.15–0.20 °C/min): not in thermal equilibrium.
* Task order is not recorded, so the absolute difference may be a time effect, not an exercise effect.

### RQ7 — healthy-only model with synthetic warming (`fig_one_class.png`)
| Synthetic change | max \|z\| (spec 0.94) | Mahalanobis (spec 0.88) |
| --- | --- | --- |
| all 10 MCP/PIP +0.5 °C | 0.32 | 0.47 |
| all 10 MCP/PIP +1.0 °C | 0.71 | 1.00 |
| one joint +1.0 °C | 0.23 | 0.49 |
| one joint +2.0 °C | 0.52 | 1.00 |

Polyarticular warming ≥ 1 °C is detectable against this healthy reference; a single joint < 1 °C
mostly is not. Synthetic changes on healthy hands only.

### Stage 12 — edge-style run (`src.edge_demo`, `edge_benchmark.json`)
| Item | Value |
| --- | --- |
| Per-frame time (median) | **29 ms (≈ 34 fps)** — landmarks 9 ms, thermal mask 12 ms, ROIs 8 ms |
| Sensor rate | 8.7 Hz → ~3.9× headroom |
| Memory | ~160 MB after model load; peak ~455 MB incl. memory-mapped `.npy` pages |
| Model sizes | landmarker 7.5 MB; healthy reference 1.5 KB; joint correction < 1 KB |
| Hardware | AMD laptop CPU, 16 logical cores, single process — **not** an edge board |

Demo: +2 °C on the middle PIP ROI of S2_right_static flags exactly that joint (z = −2.1).

---

## 4. Conclusions

1. **The measurement chain works on smartphone RGB–thermal video**: landmarks in essentially every
   frame, a near-constant RGB→thermal transform for this camera, ~4× real-time on a laptop CPU.
2. **Joint placement misses the 8 px target** (10.3 px after correction; 8.8 px on static hands), but
   **the miss changes joint temperatures by only ~0.05 °C** — small next to the measurement's own
   0.26 °C repeatability limit.
3. **Relative features are the right choice**: the 10-joint D2JTD is highly repeatable and robust to
   session drift, task order and left/right timing; absolute temperatures shifted 0.2–1.7 °C.
4. **Healthy reference values differ from the literature** (0.39 vs 1.5 °C): thresholds must be
   calibrated per device and protocol.

## 5. What is needed next

1. **Calibration recording** with the same camera (heated checkerboard at hand distance) — an
   independent geometric reference that would also test the joint correction.
2. **A second annotation pass** on ~20 frames (or a second annotator) to measure clicking error and
   so know how much of the 10 px is the pipeline.
3. **New data collection** with the Phase 0 protocol (room temperature, acclimatisation, fixed task
   order, repeated sessions); a "positive control" (warm compress on one joint) would test detection
   with real heat.
4. **Labelled patients** (clinical / ultrasound reference) — the only way to evaluate screening.
5. Benchmark on the chosen edge board.

## 6. Reproduce

```powershell
..\.venv\Scripts\python.exe -m src.data_audit                    # Stage 1 (Python 3.13 env)
.venv312\Scripts\python.exe -m src.run_pipeline --workers 6      # Stages 5–8, ~6 min
.venv312\Scripts\python.exe -m src.annotate                      # RQ2 clicks (manual, done)
.venv312\Scripts\python.exe -m src.landmark_correction           # learn + evaluate joint correction
.venv312\Scripts\python.exe -m src.run_pipeline --workers 6      # rerun with the correction
.venv312\Scripts\python.exe -m src.analysis                      # Stages 9–11
.venv312\Scripts\python.exe -m src.edge_demo --rec S2_right_static --benchmark
```
