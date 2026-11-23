# Limitations and future work

Each entry gives the problem, why it matters, its effect on the reported results, the proposed improvement,
and whether that improvement is implemented. Numbers come from `metrics_verified.json`.

Status key: **Implemented** = in the code and tested · **Implemented, not validated** = in the code, no data to
test it · **Partial** = part of the remedy exists · **Future** = not started (usually needs new data).

| # | Limitation | Why it matters | Effect on results | Proposed improvement | Status |
|---|---|---|---|---|---|
| 1 | No arthritis labels (all 17 subjects healthy) | The system's purpose is screening support | Sensitivity, AUC, accuracy and PPV cannot be measured; only simulated sensitivity reported | Recruit patients with a reference diagnosis (rheumatologist + ultrasound); run the locked independent test | Future (code ready and guarded) |
| 2 | 17 subjects | Small reference sample | Wide CIs (specificity CI 0.82–1.00); MCD model unstable below ~12 training subjects | ≥ 50 healthy subjects across ages and skin tones | Future |
| 3 | Joint localisation error 10.3 px median (target 8 px) | ROIs may sit partly off the joint | Fails RQ2b; DIP joints least repeatable. Temperature cost small (median 0.048 °C) | Learned along-finger correction (done); stereo / depth-aware mapping; thermal-domain refinement | Partial |
| 4 | No geometric calibration recording | Intrinsic/stereo parameters unknown | Reprojection error cannot be computed; 2-D transform only | Record a heated or emissivity-contrast checkerboard; stereo calibration and rectification | Future |
| 5 | Camera parallax with depth | One 2-D transform is exact at one depth only | Open–close error 13.2 px vs static 8.8 px | Depth-aware mapping using MediaPipe z or stereo depth | Future |
| 6 | Different frame rates in live use | RGB and thermal cameras run on separate clocks | Not an issue in the stored data (equal frame counts) | Timestamp pairing buffer with thermal-frame age (simulated: 100 % within one frame) | Partial (simulation only) |
| 7 | Environment not recorded | Room temperature and acclimatisation change skin temperature | Absolute temperatures not comparable between sessions; relative features used instead | Log room temperature/humidity; 15 min acclimatisation (TISEM) | Future |
| 8 | No between-day repeatability | Screening compares visits | MDC95 0.26 °C is within-session only | Repeat sessions on different days | Future |
| 9 | Accessory contamination (rings, bracelets) | Metal reads cold and biases ROIs | Cold-band flag exists but untested on real accessories | Cold-band flag (done); recordings with rings to validate | Implemented, not validated |
| 10 | Occlusion / folded joints | Landmarks for hidden joints are guesses | Hidden joints excluded (66 % of open–close frames have ≥ 1 hidden joint) | Visibility rule (done); validate against hand-labelled visibility | Implemented, not validated |
| 11 | Skin tone and emissivity | RGB landmarking and GrabCut may vary with skin tone | Not assessed; cohort not described | Diverse cohort; report performance per group | Future |
| 12 | Hand position variation | Distance changes scale and parallax | Per-recording shift absorbs part of it | Positioning jig or distance check at capture | Future |
| 13 | Model generalisation | One camera, one setting, healthy volunteers | Unknown performance on other devices or sites | Locked independent tests (done); external datasets | Partial |
| 14 | Future class imbalance | Patients will be fewer than healthy controls | — (no patients yet) | Per-class counts, balanced accuracy, small-sample warnings (done) | Implemented |
| 15 | No independent external dataset | Generalisation unproven | Software test only (fixture re-uses training subjects) | Upload module ready; acquire new recordings | Partial |
| 16 | Edge deployment | Target is a phone / embedded device | Only laptop timing (29.4 ms/frame) | Benchmark on the target board: time, memory, power | Future |
| 17 | Clinical validation | Needed before any clinical use | No evidence of clinical value | Prospective study with reference standard and ethics approval | Future |
| 18 | Single annotator, one pass | Ground truth for joint error is noisy | Click scatter only bounded (≤ 17.6 px RMS) | Second annotation pass or second annotator | Future |

## Recommended order of future work

1. **Labelled patient cohort.** This is the only way to measure sensitivity, and the decisive step for the screening question.
2. **Calibration recording and depth-aware mapping.** Directly addresses the failed joint-position criterion.
3. **Second annotation pass and between-day sessions.** These measure the two error sources currently only bounded.
4. **Edge-device benchmark.** Needed before any claim about real-time portable use.
5. **External dataset through the locked independent-test workflow.** Generalisation to new devices and sites.

## What must not be claimed

- Arthritis diagnosis, clinical accuracy, or "detection" of arthritis.
- Universal temperature thresholds. The reference intervals come from 17 healthy volunteers in one setting.
- Perfect camera alignment. The held-out Dice median is 0.911, and joints are 10.3 px off on the median.
- Proven edge deployment. Only laptop timings exist.
