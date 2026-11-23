# v9: RGB→thermal calibration

## Final module: `src/calibration.py` (find → apply → verify) — the default since v1.0.0
`find_calibration()` determines everything from paired recordings, with no calibration target:
- the image area (median over frames),
- an orientation check (a rotated or transposed export is refused, with an explanation),
- the scale and rotation, using a joint search scored by the distance between hand **outlines** (capped at 10 px, so
  a warm forearm cannot pull the scale),
- a per-frame shift learned from the hand,
- a self-check on frames not used for fitting.

`Calibration` then maps points RGB→thermal and back, resamples whole images and reads temperatures at points.
`verify()` grades the calibration on any recording.

**Proof 1: recovering KNOWN transformations** (`python -m src.calibration selftest`;
`selftest_known_distortions.csv`). Three real recordings were deliberately distorted: the thermal side was
scaled ±8–10 %, rotated ±4–6° and shifted by up to 25 px, and the RGB picture was moved 40 px inside the video
frame. In all 12 cases the module recovered the transformation:
- scale within 0.01, rotation within 0.75°;
- mapping error 1.4 px median across cases (worst case median 4.4 px).

**Proof 2: generalisation to unseen people** (`python -m src.calibration benchmark`;
`benchmark_find_calibration.json`). Calibrated from subjects S1–S5 only, then applied unchanged to S6–S17 and
scored against 551 hand-clicked joints:

| | Joint error, median | Static | Open–close | Dice |
|---|---|---|---|---|
| v3: one shift fitted per recording | 11.04 px | 9.53 | 13.23 | 0.903 |
| **Found calibration (this module)** | **9.97 px** | **8.29** | **12.77** | **0.912** |

**Production profile:** `models/calibration/flirone_one_pro_v9_found.json`, built from all 68 recordings, with the
joint correction re-learned for this mapping. It found box 0–480 × 140–500, scale 1.27, angle 0°; held-out Dice
0.917, outline error 3.8 px (grade good). `config.yaml` uses it (`registration.mode: learned`).

**Grades:** good ≤ 5 px, acceptable ≤ 8 px, otherwise poor (median held-out outline distance). On the FLIROne
reference data a correct calibration leaves 3.8–4.2 px between the RGB and thermal hand outlines, because of
edge blur and segmentation differences.

---

# Earlier step: learned per-frame shift (src/calibration_model.py)

**Code:** `src/calibration_model.py` · **Model:** `models/calibration/flirone_one_pro_v9_learned.json` ·
**Tests:** `tests/test_v9.py` (all 41 project tests pass)

## What it does
- **Train once.** 1,729 frames from all 68 recordings (24 per recording, plus every annotated frame). For each
  frame it records the RGB landmarks, both hand masks and the best RGB→thermal shift for that single frame.
  A robust (Huber) model learns the shift from the hand seen in RGB: its size (a proxy for distance) and position.
- **Apply to any video.** The shift is predicted for every frame from that frame's landmarks, so a new video
  of the same camera needs no fitting (`registration.mode: learned`, now the default in `config.yaml`).
- **New camera or dataset: auto-calibrate.** It finds the image area (letterbox), the scale and the shift
  model from the dataset's own recordings. It then checks the result on frames not used for fitting and saves
  a calibration profile. In the app: Calibration page → *Auto-calibrate from this dataset*. On the command
  line: `python -m src.calibration_model auto --dataset <id>`.

## Results (leave-one-subject-out; 759 clicked joints that have landmarks)

| Method | Joint error, median | Static | Open–close | Dice |
|---|---|---|---|---|
| One shift for everything | 10.16 px | 8.55 | 12.97 | 0.905 |
| v3: fitted per recording | 10.35 px | 8.83 | 12.53 | 0.907 |
| **v9 learned (chosen)** | **10.13 px** | **8.48** | 12.60 | 0.905 |
| Hybrid | 10.31 px | 8.81 | 12.63 | 0.907 |
| Per-frame fit (upper bound) | 10.47 px | 9.22 | 13.48 | 0.906 |

**Auto-calibration from scratch.** It was run on S1–S5 only, with nothing preset. It found:
- the image box exactly (0–480, 140–500),
- scale 1.28 (the hand-tuned value is 1.25),
- shift (−12.3, −32.4) px (hand-tuned: −12.3, −31.5),
- held-out Dice 0.921, in 85 s.

Applied unchanged to the 12 unseen subjects S6–S17, it gave a joint error of 10.87 px against 11.04 px when
each recording is fitted separately (static 8.53 vs 9.53 px), and Dice 0.913 vs 0.903.

## What this shows
- **The FLIR ONE Pro mapping is stable.** The best shift barely depends on hand distance (correlation 0.09)
  and varies only 1.4 / 2.6 px within a recording. So one learned calibration transfers to every video, and
  fitting each video separately is unnecessary.
- **Calibration is not what limits joint accuracy.** Even a perfect per-frame outline fit does not reduce the
  ~10 px joint error. The remaining error comes from where MediaPipe places joints compared with the thermal
  joint centre, and from click noise. It is not a mapping error, so better outline-based calibration cannot
  remove it.
- **The learned-vs-v3 difference is small.** p = 0.06 (Wilcoxon), so treat the two as equal. The benefit is
  that the calibration is found automatically and works on any video, not that it is more accurate.
