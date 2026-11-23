# Stage 2 — Literature review

**Project:** Lightweight RGB–thermal hand-joint screening for arthritis-related inflammation (edge AI)
**Written:** 21 Nov 2026 · **Inputs:** project architecture PDF, pipeline diagram, Stage 1 audit, 2 local papers, web search

---

## 0. Why this review, and how to read it

The architecture document fixes a pipeline: RGB landmarks → map joints to thermal → per-joint
temperature features → small classifier. This review asks one question per pipeline phase:
**what does the published evidence say, and what does that mean for *our* dataset?**
Each section ends with a **→ Decision** line. Section 5 turns these into the choices Stage 3 must make.

Evidence strength is marked as **[strong]** (systematic review / large cohort / repeated finding),
**[moderate]** (one solid study) or **[weak]** (small study, abstract only, or conflicting).

---

## 1. What our dataset is (from Stage 1 + dataset README)

| Property | Value | Consequence |
| --- | --- | --- |
| Camera | FLIR ONE Pro (smartphone add-on) | Low-cost sensor; see §3.3 on accuracy |
| Subjects | 17, both hands, tasks *static* and *open–close* (68 recordings) | Very small *n* for any classifier |
| Streams | RGB MP4, thermal MP4 (display only), radiometric `.npy` (°C) | Use `.npy` for every number |
| Frame geometry | `.npy` = upright portrait H 640 × W 480; MP4s letterbox it at x 140–500 | Verified in Stage 1 (r ≈ 0.996) |
| Sync | All three streams have equal frame counts and are index-aligned **when decoded in order** | OpenCV frame seeking is unreliable on these files |
| Frame rate | 5.9–8.8 fps, variable | Use time, not frame counts, for temporal features |
| **Labels** | **None.** No diagnosis, no inflammation score, no clinical exam | **Cannot train or test an arthritis classifier on this data alone** |
| Metadata missing | Room temperature, humidity, acclimatisation, distance, emissivity, age/sex | Absolute temperatures cannot be standardised across subjects |

The last two rows decide most of what follows.

---

## 2. Closest prior work: Qiao et al. (2025), D2JTD from RGB-T images

*Biomedical Signal Processing and Control* 112 (2026) 108482 — local copy in Downloads.
This is almost exactly our pipeline, built on the **same camera (FLIR ONE Pro)**.

| Step | What they did | Result |
| --- | --- | --- |
| Data | 82 RA patients + 30 healthy; 661 image pairs, patients self-captured at home weekly | 23 % data loss from self-capture |
| RGB–thermal alignment | **Translation only** (rigid rig); edge maps cross-correlated within a search window | Simple, no checkerboard |
| Joints | Off-the-shelf keypoint models + their "physical-distance" re-ordering fix | OpenPose 89.0 % → 97.7 % images with all joints correct |
| Failure modes | (1) finger order reversed, (2) partial reversal, (3) index/little MCP collapse onto one point, (4) keypoints in background | 3–4 unfixable → ask for a retake |
| Segmentation | Fuse thermal + grey RGB (α = 0.5), marker-controlled watershed seeded by landmarks | Dice 0.90 ± 0.06 vs clinicians |
| ROIs | 10 joints (MCP + PIP; DIPs skipped), circle radius = βᵢ·‖MCP−PIP‖, β = 0.42/0.28/0.28/0.31/0.35 (thumb→little) | Scales with hand size and distance |
| Feature | **D2JTD** = dorsal-hand temperature − joint temperature | Healthy 1.5 ± 0.4 °C; RA 1.1 ± 0.7 °C; high activity 0.8 ± 0.3 °C |
| Classifier | SVM / small MLP on D2JTD only, 5×5 nested CV | Recall 76.8 %, precision 79.0 % |

Key lessons **[moderate]**:
1. **In healthy hands the joints are *cooler* than the back of the hand** (D2JTD ≈ +1.5 °C).
   Inflammation shows up as that gap *shrinking*, not as an obviously "hot joint".
2. Segmentation errors bias temperature **downwards** because background is cooler:
   Dice < 0.8 gave a 2.3 ± 0.8 °C underestimate. Joint ROIs must stay on skin.
3. Relative features (D2JTD) were chosen *because* absolute temperatures were unreliable on
   this camera in uncontrolled settings.
4. Landmark failures are frequent and patterned; they need explicit sanity checks.

→ **Decision:** treat Qiao et al. as the reference implementation. Re-use their ROI radius rule,
D2JTD feature and landmark checks; report our numbers in the same units so they are comparable.

---

## 3. Evidence by pipeline phase

### 3.1 Does joint inflammation show up in hand thermography?

* **Systematic review (Schiavon et al., 2021, *Cartilage*; local copy)** — 32 studies, 2 094 patients,
  1990–2021. Thermography correlated with disease presence, activity scores (DAS-28, HAQ),
  inflammatory markers (ESR, CRP) and ultrasound synovitis in most studies. The MCP joints of the
  2nd–3rd fingers and the 3rd PIP/DIP were the most informative hand joints. Caveats: small studies,
  different cameras and protocols, no meta-analysis possible. **[strong for "a signal exists", weak for effect size]**
* **ThermoJIS (RMD Open 2022; 8:e002458)** — ML on dorsal hand thermograms, trained on 449
  people, ultrasound as reference. AUC 0.78 for active synovitis in 146 RA patients; moderate
  correlation with ultrasound (ρ ≈ 0.5). Notably it was done **without acclimatisation or a controlled
  room**, and still worked. Single centre; authors hold a patent. **[moderate]**
* **Snekhalatha et al. (2015)** — RA hands 2.6 °C warmer at the MCPs, 0.8 °C warmer on the dorsum
  than controls (as summarised by Qiao et al.). **[moderate]**
* **Pauk et al. (2019, *Sensors*; *Med Biol Eng Comput*)** — resting finger temperature did *not*
  separate RA from healthy; the max–min temperature along the finger **after a cooling test** did.
  → dynamic (provocation) thermography can reveal what static images hide. **[moderate]**
* **Jones et al. (2018, *Clin Rheumatol*, "Hot joints: myth or reality?")** — 49 RA patients:
  **no** association between hand-joint temperature and disease activity. **[moderate, negative]**
* **Heat Distribution Index** (Salisbury 1983; Spalding et al. 2008) — spread of temperatures within
  a joint ROI; a 1.3 °C cut-off was *proposed* for active wrist/MCP arthritis. Not independently
  validated. **[weak]**
* Bilateral comparison is limited in RA because inflammation is usually symmetric; the review notes
  contralateral differences were small, so patients were separated by comparison with healthy
  controls, not by left–right asymmetry.

→ **Decision:** the signal is real but modest (AUC ≈ 0.75–0.80 in the best realistic studies) and
partly inconsistent. Claim "screening aid", never diagnosis (already in the architecture). Prefer
*relative* and *distributional* features over absolute temperature.

### 3.2 Measurement conditions and protocols

* **TISEM checklist** (Moreira et al., 2017, *J Therm Biol* 69:155–162) — 15-item Delphi consensus:
  participant details (3), camera/room/environment (8), image recording and analysis (4). Use it to
  report our methods and to score the bias risk of our own dataset.
* **IACT guidelines** (cited in Schiavon et al.) cover room, equipment and patient preparation.
* Skin temperature is affected by room temperature, humidity, air flow, circadian rhythm, recent
  exercise, food, drugs, alcohol, smoking, camera distance and angle (Schiavon et al.). **[strong]**
* **Skin emissivity ≈ 0.98** (range 0.97–0.99), not meaningfully dependent on pigmentation
  (Sanchez-Marin et al. 2009, *J Biomed Opt*; PLOS ONE 2020 pigmentation study). **[strong]**
  We cannot check what emissivity the FLIR app used for this dataset; it is not recorded.

→ **Decision:** our dataset fails the TISEM environment items (no room temperature, no acclimatisation
record). Therefore **absolute °C comparisons between subjects are not trustworthy**; within-hand
relative features are. Write this limitation into every later result.

### 3.3 The FLIR ONE Pro as a measuring instrument

* **Spec sheet:** thermal sensor 160 × 120 (upscaled for output), 8.7 Hz, sensitivity 70 mK,
  accuracy ±3 °C or ±5 % (only after 60 s warm-up, unit at 15–35 °C).
  Our `.npy` is 640 × 480, so it is resampled; a joint ROI covers far fewer *real* sensor pixels
  than it seems. (A quick check in this review could not confirm the exact native grid from the
  data; verify in Stage 3.)
* **Sensors 2026, 26(4):1295** — 40 adults, 2 400 hand-dorsum measurements vs an IR thermometer:
  large systematic bias, agreement outside clinical limits, repeatability SD ≈ 0.30–0.34 °C.
  ML calibration removed bias but **random error persisted**. **[moderate]**
* The frame rate in our data (5.9–8.8 fps) matches the 8.7 Hz sensor with dropped frames.

→ **Decision:** a fixed offset (bias) cancels in differences such as D2JTD or joint − palm; random
error does not. So: **average over many frames** (we have ~700–1 500 per recording) before computing
features, and report frame-to-frame repeatability as a quality metric.

### 3.4 Locating the joints

* **MediaPipe Hands** (Zhang et al., 2020, arXiv:2006.10214) — palm detector + landmark model,
  21 landmarks, runs real-time on phones; the detector reruns only when tracking is lost. Fits the
  edge requirement.
* Qiao et al. show generic keypoint models fail in **patterned** ways on these images (finger order
  reversal, MCP collapse, background keypoints) and that simple anatomical distance checks fix most.
* **Thermal-only** hand pose is possible (ThermoHands benchmark, arXiv:2403.09871; Li et al.
  arXiv:2303.01547) but needs specialised models; RGB landmarks are the practical route.
* In our open–close recordings the posture changes continuously (open hand ↔ fist, where finger
  joints are hidden), so landmark confidence must be checked **per frame**.

→ **Decision:** keep MediaPipe (Lite) as frozen, add Qiao-style checks (finger ordering via thumb
reference, MCP collapse test, background test) and drop frames that fail rather than repair them.
Note: MediaPipe currently needs Python ≤ 3.12; our `.venv` is 3.13 — Stage 3 needs a 3.11/3.12 env.

### 3.5 RGB → thermal registration

* The architecture assumes checkerboard **stereo calibration**. That needs images of a target visible
  to both cameras (heated or emissivity-contrast board: e.g. PST900, arXiv:1909.10980, flood-lamp
  heated paper; aluminium/vinyl or aluminium/carbon boards). **Our dataset contains no calibration
  target**, so this cannot be done for it.
* Qiao et al. used **translation-only** alignment for the same camera. Our Stage 1 overlays show the
  thermal hand is also **slightly larger** than the RGB hand, so translation alone will not be enough
  here — the RGB and thermal lenses have different fields of view.
* The offset between the two cameras changes with hand distance (parallax), and distance is not
  recorded.

→ **Decision:** for this dataset, estimate a **per-recording similarity/affine transform** by aligning
the RGB hand mask to the thermal hand mask (both are easy to segment here: warm hand vs cool
background, skin vs paper/table), and measure the residual error at landmarks. Keep checkerboard
calibration for the future own-rig data collection.

### 3.6 ROIs and features

| Feature | Evidence | Status for our data |
| --- | --- | --- |
| Joint − dorsum (D2JTD) | Qiao 2025 healthy 1.5 ± 0.4 °C **[moderate]** | ✅ Computable; we can produce a healthy reference |
| Joint − palm / joint − surrounding skin | architecture; similar idea to D2JTD | ✅ (we see the dorsum, not the palm — rename to "dorsum") |
| Left − right same joint | healthy ΔT usually < 0.5 °C; 905-adult reference study mean 0.06 ± 0.49 °C (Bioengineering 2023, 10(6):671) **[moderate]**; less useful in RA (symmetric) | ✅ Both hands recorded per subject, but different recordings → time gap |
| Heat Distribution Index | Spalding 2008 **[weak]** | ✅ Computable, exploratory |
| Absolute joint temperature | many studies, but camera/environment dependent | ⚠️ Not comparable across subjects here |
| Dynamic response (after cooling / exercise) | Pauk 2019; psoriatic-arthritis exercise study in Schiavon **[moderate]** | ✅ *Open–close* is a mild exercise task — we can measure the change over time |

ROI size: Qiao's βᵢ·‖MCP−PIP‖ rule, keeping each ROI on skin (shrink near edges), and reporting how
many native sensor pixels fall inside.

→ **Decision:** primary features = D2JTD per joint and its temporal trend; secondary = L−R
asymmetry, HDI. Absolute temperature only as a covariate.

### 3.7 Classifiers and evaluation

* Very high accuracies are reported on small thermogram datasets (e.g. RANet, *Sci Rep* 2023:
  95–97 %; RA-XTNet, *Diagnostics* 2024: 87.5 % on thermal). These are small single-source datasets
  without subject-independent external testing and are likely optimistic. **[weak]**
* Realistic, better-designed studies land at **AUC ≈ 0.78–0.87** (ThermoJIS; Winnipeg high-resolution
  thermography abstract, XGBoost AUC 0.869, 100 RA / 137 controls — conference abstract only).
* Qiao et al. used nested cross-validation; the architecture already insists on subject-wise splits.
  With 17 subjects, a 70/15/15 split leaves ~3 test subjects — too few; **leave-one-subject-out** is
  the appropriate scheme.

→ **Decision:** keep the lightweight models; replace the 70/15/15 split with grouped
(leave-one-subject-out) cross-validation; report confidence intervals, not single numbers.

---

## 4. Gap analysis: architecture vs evidence vs this dataset

| Architecture phase | Evidence says | This dataset | Gap / action |
| --- | --- | --- | --- |
| 0 Environment control | Required (TISEM, IACT) | Not recorded | Use relative features; state limitation |
| 1 Timestamped acquisition | Correct approach | Streams index-aligned; no seeking | Decode in order (done in Stage 1) |
| 2 Checkerboard calibration | Standard for own rig | No target recorded | Mask-based affine per recording + error report |
| 3 MediaPipe landmarks | Works; known failure modes | RGB available, varied posture | Add Qiao-style checks; Python 3.12 env |
| 4 Thermal preprocessing | Use radiometric data | `.npy` in °C, no NaN in sample | Average frames; flag edge/background pixels |
| 5 Joint ROIs | Qiao radius rule | Feasible | Implement + Dice-style check on a few hand-labelled frames |
| 6 Quality flags | Don't delete hot pixels | Feasible | Per-joint validity flags |
| 7 Features | Relative > absolute | Feasible | D2JTD, L−R, HDI, temporal |
| 8 Classifier | Needs labels | **No labels** | **Blocked** for arthritis; see §5 |
| 10 Validation | Needs clinical reference | None | Validate the *measurement*, not diagnosis |

---

## 5. What this means for Stage 3 (learning objective)

The dataset is 17 presumably healthy volunteers with no clinical labels, so **training or evaluating
an arthritis classifier on it would be scientifically invalid**. What it *can* support, in order:

1. **Validate the measurement pipeline** on real RGB–thermal data: landmark success rate (with the
   Qiao checks), RGB→thermal registration error, ROI-on-skin rate, per-joint validity.
2. **Build a healthy reference model** of per-joint D2JTD, L−R asymmetry and HDI, with
   between-subject spread — the "normal range" a future screening tool compares against
   (cf. healthy D2JTD 1.5 ± 0.4 °C in Qiao et al.).
3. **Measure reliability:** frame-to-frame and within-recording stability (repeatability SD,
   ICC), static vs open–close — needed because the camera's random error is ~0.3 °C.
4. **Characterise the open–close response** (how joint/dorsum temperature changes during mild
   exercise) as a candidate dynamic feature.
5. Optional, clearly labelled as such: a **one-class / anomaly** model trained on healthy hands,
   tested only on synthetic perturbations (e.g. +0.5–2 °C added to a joint ROI) to show detection
   sensitivity. This shows the *method*, not clinical performance.

Labelled patient data would be needed for any arthritis claim. Qiao et al.'s data are not shareable;
no public labelled RGB-T arthritis hand dataset was found in this review.

**Proposed Stage 3 objective (for discussion):**
*"Build and validate an automated pipeline that extracts per-joint relative temperature features from
smartphone RGB–thermal hand video, quantify its geometric accuracy and measurement repeatability, and
establish healthy reference ranges for those features."*

---

## 6. References

Local copies (Downloads):
1. Qiao Y, Hamann P, Robson C, Cosker D. Dorsal-to-Joint Temperature Differences (D2JTD) extraction
   from RGB-T images for remote rheumatoid arthritis monitoring. *Biomed Signal Process Control*
   2026;112:108482. https://doi.org/10.1016/j.bspc.2025.108482
2. Schiavon G, Capone G, Frize M, Zaffagnini S, Candrian C, Filardo G. Infrared thermography for the
   evaluation of inflammatory and degenerative joint diseases: a systematic review. *Cartilage*
   2021;13(Suppl 2):1790S–1801S. https://doi.org/10.1177/19476035211063862

Found in this review:

3. Moreira DG, Costello JT, Brito CJ, et al. Thermographic imaging in sports and exercise medicine:
   a Delphi study and consensus statement on the measurement of human skin temperature (TISEM).
   *J Therm Biol* 2017;69:155–162. https://doi.org/10.1016/j.jtherbio.2017.07.006
4. Zhang F, Bazarevsky V, Vakunov A, et al. MediaPipe Hands: on-device real-time hand tracking.
   arXiv:2006.10214, 2020. https://arxiv.org/abs/2006.10214
5. ThermoJIS — machine-learning thermal joint inflammation score from hand thermography, validated
   against ultrasound. *RMD Open* 2022;8(2):e002458. https://pmc.ncbi.nlm.nih.gov/articles/PMC9295660
   (companion ThermoDAI: *RMD Open* 2022;8(2):e002615)
6. Machine learning calibration of smartphone-based infrared thermal cameras: improved bias and
   persistent random error. *Sensors* 2026;26(4):1295.
   https://nchr.elsevierpure.com/en/publications/machine-learning-calibration-of-smartphone-based-infrared-thermal/
7. Teledyne FLIR. FLIR ONE Pro product specifications. https://www.flir.com/products/flir-one-pro/
8. Snekhalatha U, Anburajan M, Sowmiya V, Venkatraman B, Menaka M. Automated hand thermal image
   segmentation and feature extraction in the evaluation of rheumatoid arthritis. *Proc Inst Mech Eng H*
   2015;229(4):319–331. https://doi.org/10.1177/0954411915580809
9. Pauk J, Wasilewska A, Ihnatouski M. Infrared thermography sensor for disease activity detection in
   rheumatoid arthritis patients. *Sensors* 2019;19:3444. https://doi.org/10.3390/s19163444
10. Pauk J, Ihnatouski M, Wasilewska A. Detection of inflammation from finger temperature profile in
    rheumatoid arthritis. *Med Biol Eng Comput* 2019;57:2629–2639. https://doi.org/10.1007/s11517-019-02055-1
11. Jones B, Hassan I, Tsuyuki RT, Dos Santos MF, Russell AS, Yacyshyn E. Hot joints: myth or reality?
    A thermographic joint assessment of inflammatory arthritis patients. *Clin Rheumatol*
    2018;37(9):2567–2571. https://doi.org/10.1007/s10067-018-4108-0
12. Salisbury RS, Parr G, De Silva M, Hazleman BL, Page-Thomas DP. Heat distribution over normal and
    abnormal joints: thermal pattern and quantification. *Ann Rheum Dis* 1983;42(5):494–499.
    https://doi.org/10.1136/ard.42.5.494
13. Spalding SJ, Kwoh CK, Boudreau R, et al. Three-dimensional and thermal surface imaging produces
    reliable measures of joint shape and temperature: a potential tool for quantifying arthritis.
    *Arthritis Res Ther* 2008;10(1):R10. https://doi.org/10.1186/ar2360
14. Frize M, Ogungbemile A. Estimating rheumatoid arthritis activity with infrared image analysis.
    *Stud Health Technol Inform* 2012;180:594–598.
15. Reference standards for digital infrared thermography measuring surface temperature of the upper
    limbs (905 healthy adults). *Bioengineering* 2023;10(6):671. https://pmc.ncbi.nlm.nih.gov/articles/PMC10294988/
16. Freezing fingers syndrome, primary and secondary Raynaud's phenomenon: characteristic features with
    hand thermography (healthy L–R symmetry ΔT < 0.5 °C). *Acta Derm Venereol*.
    https://doi.org/10.2340/00015555-1508
17. Sanchez-Marin FJ, Calixto-Carrera S, Villaseñor-Mora C. Novel approach to assess the emissivity of
    the human skin. *J Biomed Opt* 2009;14(2):024006. https://doi.org/10.1117/1.3086612
18. The effect of constitutive pigmentation on the measured emissivity of human skin. *PLOS ONE* 2020.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC7688144/
19. ThermoHands: a benchmark for 3D hand pose estimation from egocentric thermal images.
    arXiv:2403.09871. https://arxiv.org/abs/2403.09871
20. Shivakumar SS, et al. PST900: RGB-thermal calibration, dataset and segmentation network.
    arXiv:1909.10980. https://arxiv.org/abs/1909.10980
21. RANet: a custom CNN model and quanvolutional neural network for the automated detection of
    rheumatoid arthritis in hand thermal images. *Sci Rep* 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10511741/
22. RA-XTNet: a novel CNN model to predict rheumatoid arthritis from hand radiographs and thermal
    images. *Diagnostics* 2024;14(17):1911. https://pmc.ncbi.nlm.nih.gov/articles/PMC11394616/

Citation details marked only by title were taken from search results and abstracts; full texts of
refs 5, 6, 15, 16, 18, 21 and 22 were not read. Check them before citing in a thesis or paper.
