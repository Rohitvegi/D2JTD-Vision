"""Project audit and viva defence guide (.docx).

Run from the project root:
  .venv312\\Scripts\\python.exe docs/deliverables_src/build_defence_guide.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from docx.shared import Pt  # noqa: E402

from defence_content import (AGG, ANNOTATION, B3, B9, CHECKS, PROD, SELF_MED, C, D, FIG, LIBS, LW, NAMES, NOT_USED, PR,  # noqa: E402
                             REPRESENT, REVIEW, ROBUST, SEQUENCE_WHY, SLIDES, STATUS, V3, V5, VIVA, WORKFLOW, f, pct)
from doc_helpers import Doc  # noqa: E402
from references import REFS, cite  # noqa: E402

import doc_helpers  # noqa: E402

doc_helpers.EQ_DIR = Path("deliverables/figures/equations_guide")      # keep the report's equation images intact
OUT = Path("deliverables/RGB_Thermal_Hand_Project_Audit_and_Viva_Guide.docx")
doc = Doc()
doc.d.sections[0].footer.paragraphs[0].runs[0].text = "RGB–Thermal Hand-Joint Project — Audit and Viva Defence Guide · page "

doc.title_page("Project Audit, Architecture and Viva Defence Guide",
               "RGB–Thermal Hand-Joint Temperature Analysis for Arthritis Screening Research",
               ["Every statement checked against the source code (src/, ui/), config.yaml and stored results",
                "17 healthy subjects · 68 recordings · FLIROne dataset",
                "Pipeline v3 → v9 (automatic calibration) · software release 1.0.0", "22 November 2026"])
doc.toc()

doc.h("How to use this guide")
doc.p("Parts 1–10 explain what was built and why, in the order the data flows. Parts 11–13 are the report summary, "
      "the slide-by-slide plan and the full speaking script for the accompanying viva deck "
      "(RGB_Thermal_Hand_Viva_Presentation.pptx, whose speaker notes contain the same script). Part 14 has "
      f"{len(VIVA)} likely viva questions with answers; Part 15 is a strict critical review. "
      "Status words used throughout: Completed, Partial, Planned, Missing, Questionable.")

# ============================================================ PART 1
doc.h("Part 1. Implementation audit")
doc.p("Each component was located in the code and traced through its inputs and outputs; documentation alone was "
      "never taken as proof that something exists.")
doc.table(["Component", "Where", "Purpose", "Input", "Output", "Status"], [list(r) for r in STATUS],
          "Project implementation status (from the source code).", widths=[3.2, 3.2, 3.0, 2.3, 2.6, 2.3], size=7.5)
groups = {}
for r in STATUS:
    key = r[5].split(" ")[0]
    groups.setdefault(key, []).append(r[0])
for k in ("Completed", "Partial", "Planned", "Missing", "Questionable"):
    if k in groups:
        doc.p(", ".join(groups[k]) + ".", bold_lead=f"{k}: ")

# ============================================================ PART 2
doc.h("Part 2. The dataset")
doc.h("2.1 Source, size and structure", 2)
doc.table(["Item", "Value"],
          [["Source", f"FLIROne hand dataset {cite('dataset')}: FLIR ONE Pro smartphone camera, healthy volunteers"],
           ["Subjects / hands / recordings", f"{D['subjects']} / 34 (17 left + 17 right) / {D['recordings']}"],
           ["Tasks", "static (open hand at rest) and open–close (repeated fist clenching)"],
           ["Frames", f"{D['frames_total']:,} RGB = {D['frames_total']:,} radiometric (equal in all 68 recordings)"],
           ["Frames per recording", f"{D['frames_min']}–{D['frames_max']} (median {D['frames_median']:.0f})"],
           ["Frame rate", f"measured {D['fps_min']:.2f}–{D['fps_max']:.2f} fps (header values 5.94–8.77 unreliable)"],
           ["RGB resolution", "640 × 480 container; picture at x 140–500 (360 × 480)"],
           ["Radiometric", "float32 °C, N × 640 × 480 (portrait); sensor 160 × 120"],
           ["File formats", ".mp4 (RGB), .mp4 (thermal display), .npy (radiometric)"],
           ["Ground-truth labels", "none for disease; 777 joint clicks created in this project"],
           ["Missing information", "diagnosis, age/sex, room temperature, acclimatisation time, distance, calibration recording, device timestamps"]],
          "Dataset summary (measured by src/data_audit.py).", widths=[4.5, 11.9])
doc.figure(FIG / "F03_dataset_tree.png", "Dataset folder structure and naming (S<subject>/<hand>/<task>_<stream>).", 13)
doc.h("2.2 Internal representation", 2)
doc.table(["Data", "Array / format", "Meaning"], [list(r) for r in REPRESENT], "How each data type is held in memory.", widths=[3.3, 6.5, 6.6], size=9)
doc.p("RGB values measure reflected visible light; thermal-video values are palette colours rescaled every frame, so the "
      "same colour can mean different temperatures in two frames; radiometric values are physical temperatures; a "
      "normalised image rescales them to 0–1 and loses the unit. They are not interchangeable: averaging palette "
      "colours or normalised values gives numbers with no physical meaning, and two frames could not be compared. All "
      "measurements therefore come from the radiometric .npy only.", bold_lead="Why these are not interchangeable. ")

# ============================================================ PART 3
doc.h("Part 3. Complete project workflow")
for (name, what, why, tech, lib, inp, out, skip, verify) in WORKFLOW:
    doc.h(name, 2)
    doc.bullets([("What happens: ", what), ("Why: ", why), ("Technique: ", tech), ("Library: ", lib),
                 ("Input → output: ", f"{inp} → {out}"), ("If skipped: ", skip), ("Verified by: ", verify)])
doc.h("Why the order matters", 2)
doc.table(["Ordering", "Reason"], [list(r) for r in SEQUENCE_WHY], "Why the sequence of operations is fixed.", widths=[5.0, 11.4], size=9.5)
doc.h("Annotation in this project", 2)
doc.table(["Question", "Answer"], [list(r) for r in ANNOTATION], "Annotation as actually implemented.", widths=[3.2, 13.2], size=9.5)
doc.figure(FIG / "R06_annotation_contact_sheet.png", "Annotation QC: flagged and worst frames with clicks and pipeline joints.", 15)
doc.h("From many frames to one temperature per joint", 2)
doc.p(f"A recording has hundreds of frames and each gives a slightly different temperature per joint. Example: "
      f"S2_right_static, {len(AGG)} usable frames. The final temperature of a joint is the median of its per-frame values "
      "(each per-frame value is the median of the ROI's skin pixels); frames failing the checks, or where the joint is hidden, "
      "are left out for that joint. Q1–Q3 (middle half of the frames) and the number of frames are reported next to it.")
from src.landmarks import JOINT_NAMES  # noqa: E402
_rows = []
for _j in JOINT_NAMES[:6] + ["little_MCP"]:
    _x = AGG[f"{_j}_T"].dropna()
    _rows.append([_j.replace("_", " "), len(_x), f(_x.min()), f(_x.max()), f"{f(_x.quantile(.25))}–{f(_x.quantile(.75))}", f(_x.median())])
doc.table(["Joint", "Frames", "Lowest", "Highest", "Q1–Q3", "Final (median)"], _rows,
          "S2_right_static: per-frame range and final value (°C, measured; first 6 joints + little MCP).",
          widths=[3.2, 1.6, 2.0, 2.0, 3.6, 3.0], size=9)
doc.figure(FIG / "N01_frame_aggregation.png", "From 267 frames to one temperature per joint.", 16.4)
doc.p(f"Why the median: the mean is pulled by single bad frames, the first/last frame depends on one moment and the maximum "
      f"picks the outlier. The final value repeats within {f(V3['mdc'], 2)} °C between two halves of a recording (MDC95). The model "
      "then scores one feature vector per hand, never individual frames.")

# ============================================================ PART 4
doc.h("Part 4. Architecture")
doc.p("Five diagrams describe the implemented system from different angles. Colours: blue = RGB, orange = thermal, "
      "purple = calibration/alignment, green = features and quality, pink = machine learning, gold = data/output.")
for fig, cap in (("D_A_high_level.png", "A. High-level system architecture: two branches combined in the fusion step."),
                 ("D_B_detailed.png", "B. Detailed technical architecture with algorithm and library in each block."),
                 ("F01_architecture.png", "Module-level architecture of all implemented components."),
                 ("D_C_training.png", "C. Training pipeline: healthy-reference model and joint correction, both subject-wise."),
                 ("D_D_inference.png", "D. New-data inference: validation, quality and range gates before scoring."),
                 ("D_E_evaluation.png", "E. Evaluation architecture: every metric paired with its own reference.")):
    doc.figure(FIG / fig, cap, 16.2)

# ============================================================ PART 5
doc.h("Part 5. Algorithms and libraries")
doc.table(["Library", "What it is", "Role in the project", "Why selected", "Alternatives", "Functions used"], [list(r) for r in LIBS],
          "Libraries actually imported by the code.", widths=[2.4, 2.2, 3.6, 2.6, 2.0, 3.6], size=7.5)
doc.note(NOT_USED)
doc.h("5.1 Mathematical concepts used", 2)
doc.p("Every coordinate from the 360 × 480 RGB picture is multiplied by k = 480/360 = 4/3 to reach the 480 × 640 .npy grid; "
      "masks are warped with nearest-neighbour interpolation so they stay binary.", bold_lead="Resizing and coordinate scaling. ")
doc.eq(r"\mathbf{p}_{th} = s\,(k\,\mathbf{p}_{rgb} - \mathbf{c}) + \mathbf{c} + \mathbf{t}",
       "Similarity transform used for alignment (scale s = 1.25 fixed, shift t per recording). A homography (8 parameters) "
       "and stereo calibration (intrinsics + rotation + translation) are NOT used: there is no calibration target, and a "
       "hand outline cannot constrain 8 parameters reliably.")
doc.eq(r"\tau^{*} = \arg\max_{\tau}\ \omega_0(\tau)\,\omega_1(\tau)\,[\mu_0(\tau)-\mu_1(\tau)]^2",
       f"Otsu threshold {cite('otsu1979')}: the temperature that maximises the between-class variance of hand vs background.")
doc.eq(r"\mathrm{D2JTD}_j = \tilde T_{dorsum} - \tilde T_j,\qquad \tilde T = \mathrm{median}\{T(x,y)\in \mathrm{ROI}\}",
       f"Temperature statistics and the main feature (Qiao et al. {cite('qiao2026')}).")
doc.eq(r"D_M(\mathbf{x}) = \sqrt{(\mathbf{x}-\mu)^{\top}\hat\Sigma^{-1}(\mathbf{x}-\mu)},\quad \hat\Sigma=(1-\delta)S+\delta\frac{\mathrm{tr}S}{p}I",
       f"Mahalanobis distance with Ledoit–Wolf shrinkage {cite('ledoit2004')}: the anomaly score. No loss function or "
       "optimiser is involved — the covariance is estimated in closed form.")
doc.p("Leave-one-subject-out: for each of 17 subjects, fit on 16 and test on the held-out one; the threshold uses a "
      "second, inner leave-one-subject-out on the 16. Regression and classification losses are not used.",
      bold_lead="Cross-validation. ")
doc.p(f"MediaPipe {cite('zhang2020')} detects the palm with a single-shot detector and regresses 21 landmarks with a "
      "convolutional network trained by Google; this project uses the trained model as-is.", bold_lead="Landmark detection. ")

# ============================================================ PART 6
doc.h("Part 6. RGB–thermal alignment")
doc.bullets([
    ("Different resolutions: ", "the thermal sensor has 160 × 120 pixels (exported as 640 × 480); the RGB camera has far more pixels and a wider field of view, and the dataset stores RGB letterboxed at 360 × 480."),
    ("Different FPS: ", "in this dataset the device exported equal frame counts, so the streams share one rate; in live use the two cameras run on separate clocks — handled by timestamp pairing in src/live_acquisition.py (tested in replay)."),
    ("Corresponding frames: ", f"frame i ↔ frame i; verified by cross-correlating RGB hand openness with thermal hand area ({V3['sync_aligned']}/34 aligned at lag 0)."),
    ("Spatial alignment (v9, default): ", f"automatic calibration (Section 6.1): image area, scale {PROD['scale']:.2f}, rotation {PROD['angle_deg']:.0f}° and a per-frame shift found from the 68 recordings. The v3 method (scale fixed at 1.25, one shift per recording) remains available as profile v3."),
    ("Calibrated once or repeatedly: ", "once per camera (a calibration profile); the shift is then predicted for every frame from the hand; a new camera or export is calibrated with Auto-calibrate."),
    ("Intrinsic / extrinsic parameters: ", "not used (no calibration target recorded); the v9 calibration uses the hand as the target. Homography and stereo calibration: not used."),
    ("Joint mapping: ", "MediaPipe joint (RGB) → equation above → along-finger correction (+9.3 % of MCP–PIP length for MCP, +10.7 % for PIP/DIP/IP) → ROI in the .npy."),
    ("Camera position errors: ", "absorbed by the per-recording shift; the residual is measured by held-out Dice and by joint error."),
    ("Hand movement and curvature: ", f"fingers bending towards the camera lie at a different depth, so a single 2-D transform misplaces them: open–close error {f(V3['joint_err_oc'], 1)} px vs static {f(V3['joint_err_static'], 1)} px."),
])
doc.h("Worked example", 2)
doc.p("Take the RGB point p = (180, 240), the centre of the 360 × 480 picture. Step 1, scale to the .npy grid: k·p = (240, 320). "
      "Step 2, subtract the centre c = (240, 320): (0, 0). Step 3, multiply by s = 1.25: (0, 0). Step 4, add c and the rig "
      "shift t = (−12.3, −31.5): (227.7, 288.5). Step 5, the along-finger correction moves a joint by about 10 % of its "
      "MCP–PIP length towards the fingertip. Step 6, the ROI disc is drawn there and the median of the °C values inside "
      "the eroded hand mask is the joint temperature.")
doc.p("A second point, p = (90, 120): k·p = (120, 160); minus c = (−120, −160); × 1.25 = (−150, −200); + c + t = (77.7, 88.5). "
      "Points far from the centre move more because of the scale — this is why the thermal hand looks larger.")
doc.figure(FIG / "F04_transform.png", "Coordinate chain RGB → .npy → thermal.", 14.5)
doc.p("A 2-D transform is exact only for points on one plane at one distance. Fingers that bend towards the camera, "
      "or a hand held closer, show parallax that no 2-D mapping can remove. A 3-D approach — stereo calibration with a "
      "heated checkerboard, then mapping each landmark using its depth (MediaPipe z or stereo disparity) — is necessary "
      "when joint-level accuracy below ~8 px is required. This is a proposed improvement, not implemented.",
      bold_lead="Limits of 2-D mapping. ")
doc.h("6.1 Automatic calibration (v9): find, apply, verify", 2)
doc.p("src/calibration.py finds the transformation from paired recordings, with the hand as calibration target:")
doc.numbered(["image area of the RGB video (median content box over frames);",
              "orientation check (a rotated or transposed radiometric export is refused, with an explanation);",
              "scale and rotation searched jointly, scored by the mean distance from the RGB hand outline to the thermal hand "
              "outline, capped at 10 px so that warm areas without an RGB counterpart (a forearm) cannot pull the scale;",
              "a per-frame shift learned from the hand's size and position (Huber regression);",
              "a self-check on frames not used for fitting: good ≤ 5 px, acceptable ≤ 8 px, otherwise poor."])
doc.eq(r"\mathbf{p}_{th} = s\,\mathrm{Rot}(\alpha)\,(k\,\mathbf{p}_{rgb} - \mathbf{c}) + \mathbf{c} + \mathbf{t}(\mathrm{frame})",
       "The calibrated transform applied to every frame (α = 0 reduces it to the v3 similarity transform).")
doc.table(["Test", "Result"],
          [["Known distortions (12: scale ±8–10 %, rotation ±4–6°, shift, picture moved 40 px, combined)",
            f"12/12 recovered; scale within 0.01, angle within 0.75°, {SELF_MED:.1f} px median"],
           ["Unseen people (calibrated on S1–S5, tested on S6–S17)",
            f"joint error {B9['joint_err_median']:.2f} px vs {B3['joint_err_median']:.2f} px (v3); Dice {B9['dice_median']:.3f} vs {B3['dice_median']:.3f}"],
           ["Production profile (all 68 recordings)",
            f"scale {PROD['scale']:.2f}, rotation {PROD['angle_deg']:.0f}°, Dice {PROD['quality']['dice_heldout_median']:.3f}, grade {PROD['quality']['grade']}"]],
          "Evidence that the calibration finds and applies the correct transformation.", widths=[8.0, 8.4], size=9)
doc.figure(FIG / "N02_calibration_selftest.png", "Known-distortion test: all 12 recovered.", 16)
doc.figure(FIG / "N03_calibration_benchmark.png", "Unseen-subject benchmark.", 15)
doc.figure(FIG / "N12_calibration_apply.png", "Applied: joints found in RGB mapped into the thermal image with their temperatures.", 14)

# ============================================================ PART 7
doc.h("Part 7. How the model is trained")
doc.table(["Item", "In this project", "Why"],
          [["Dataset preparation", "per-frame tables → per-recording features (median over valid frames); static recordings", "one vector per hand; static has no hidden joints"],
           ["Input features", "32: 14 D2JTD + 18 spatial contrasts", "most stable and most responsive in the ablation"],
           ["Target labels", "none (one-class); all training hands healthy", "no disease labels exist"],
           ["Splitting", "nested leave-one-subject-out (17 folds)", "subject-independent estimate with n = 17"],
           ["Architecture", "median imputer → Ledoit–Wolf mean and covariance → Mahalanobis distance", "few parameters, explainable"],
           ["Hyper-parameters", "threshold quantile 0.95; shrinkage δ analytic; OCSVM ν = 0.1 (comparison)", "fixed in advance, not tuned on 17 people"],
           ["Loss / optimiser / learning rate", "not applicable — closed-form estimate", "no gradient-based training"],
           ["Epochs / batch size / early stopping", "not applicable", "no iterative training"],
           ["Checkpoints", "versioned bundles in models/registry (bundle.joblib + metadata.json)", "never overwritten; reproducible"],
           ["Overfitting detection", "ablation (122 features worse), learning curve, LOSO only", "with 34 hands, more features overfit"],
           ["Underfitting detection", "simulated sensitivity of simple models (robust z, diagonal) is low", "they ignore feature correlations"],
           ["Model selection", "pre-declared score; default Ledoit–Wolf for stability", "MCD unstable below 12 subjects"]],
          "Model development as implemented.", widths=[3.6, 7.4, 5.4], size=9)
doc.p("Data leakage happens when information about the test person reaches training. With 55,239 frames from 34 hands, "
      "a random frame split would put near-identical frames of the same hand on both sides. Here every split is by "
      "subject; both hands of a person stay together; imputer, scaler, covariance, threshold and joint correction are "
      "all learned from training subjects only. The reported numbers are therefore subject-independent generalisation, "
      "not frame-level performance. (The earlier v3 analysis chose thresholds in-sample; those numbers are superseded.)",
      bold_lead="Leakage. ")

# ============================================================ PART 8
doc.h("Part 8. Evaluation and model comparison")
doc.table(["Task", "Type", "Metrics used", "Reference"],
          [["Joint localisation", "landmark localisation", "Euclidean error (median, P90), along/across bias", "777 clicks"],
           ["Registration", "segmentation overlap", "Dice, IoU", "thermal hand mask"],
           ["Temperature measurement", "measurement", "temperature cost of joint error (°C), ICC, MDC95", "clicked positions; split halves"],
           ["Healthy reference", "one-class detection", "specificity on real hands; sensitivity on simulated warming", "unseen subjects"],
           ["Disease classification", "classification", "accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC", "uploaded labels (not available)"]],
          "Tasks and their metrics (kept separate).", widths=[3.3, 3.2, 6.0, 3.9], size=9)
doc.eq(r"\mathrm{Sens} = \frac{TP}{TP+FN},\ \ \mathrm{Spec} = \frac{TN}{TN+FP},\ \ \mathrm{Prec} = \frac{TP}{TP+FP},\ \ F_1 = \frac{2PR}{P+R}",
       "Implemented in src/evaluation.py; sensitivity and AUC are refused when only one class is present, and a warning is shown below 10 samples per class.")
doc.eq(r"\mathrm{MAE} = \frac{1}{N}\sum_i |P_i - G_i|,\quad \mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_i (P_i-G_i)^2}",
       "Measurement metrics (implemented for uploaded reference values, e.g. temperatures).")
cmp = [[NAMES[k], "34 hands, nested LOSO", f(r["specificity"]), f(r["sens_all_10_MCP_PIP_1C"]), f(r["sens_single_joint_2C"]),
        f"{r['score_us_per_hand'] / 1000:.3f}"] for k, r in C.items()]
doc.table(["Model", "Split", "Specificity", "Sens. all 10 +1 °C*", "Sens. 1 joint +2 °C*", "ms / hand"], cmp,
          "Model comparison (identical splits and features). *Simulated warming. Accuracy, precision, F1 and MAE are not applicable without labels.",
          widths=[3.8, 3.4, 2.0, 2.4, 2.4, 2.0], size=9)
doc.p("All models share the same specificity (0.94, CI 0.82–1.00). Sensitivity differences reflect how each model uses "
      "correlated features: covariance-based models and the SVM respond to joint-specific warming; robust z and the "
      "diagonal model ignore correlations; Isolation Forest is insensitive with 34 samples. Bootstrap CIs overlap for most "
      "pairs, so with 17 subjects only large differences are meaningful. Inference time is ≤ 0.2 ms per hand except "
      "Isolation Forest (6.9 ms) — negligible next to the 29 ms per frame of image processing.", bold_lead="Interpretation. ")
doc.figure(FIG / "P03_prediction_overview.png", "Default model on every unseen hand: real healthy recording (green) and simulated warm joint (red).", 16)

# ============================================================ PART 9
doc.h("Part 9. New-data validation and error calculation")
doc.p("Implemented in src/dataset_manager.py (import, matching, validation) and src/evaluation.py (locked test, metrics). "
      "A recording's status is the worst level among its checks.")
doc.table(["Check", "Rule", "Failure result"], [list(r) for r in CHECKS], "Validation checks applied to every new recording.", widths=[4.2, 8.4, 3.8], size=8.5)
doc.table(["Concept", "Meaning in this project"],
          [["Data validation", "are the files usable? (checks above)"],
           ["Model validation", "how well does the model work on unseen subjects? (nested LOSO, Part 8)"],
           ["Prediction confidence", "anomaly score relative to θ; quality category — NOT a probability of being correct"],
           ["Prediction error", "|prediction − ground truth|; needs labels"],
           ["Calibration error", "mismatch to the stored calibration profile (geometry, letterbox, Dice)"],
           ["Alignment error", "Dice of warped masks; Euclidean joint error vs clicks"],
           ["Generalisation performance", "metrics on data from people never used in training"]],
          "Distinctions that are often confused.", widths=[4.2, 12.2], size=9.5)
doc.eq(r"e = \sqrt{(x_p-x_g)^2+(y_p-y_g)^2}",
       f"Applied to the 777 clicked joints: median {f(V3['joint_err_corr'], 2)} px after correction (leave-one-subject-out), "
       f"P90 {f(V3['joint_err_corr_p90'], 1)} px; the temperature at the pipeline vs clicked position differs by a median of {f(V3['dT_corr'], 3)} °C.")
doc.p("Without ground truth the system can still validate the data and give a score, a category and a quality level, but it "
      "cannot compute its error: a confident score on a mislabelled or unusual hand can still be wrong. Error and accuracy "
      "are computed only after labels are uploaded to a locked independent test.", bold_lead="Unlabelled data. ")
doc.p("New-data validation: implemented and exercised by an end-to-end software test (3 recordings, different naming, stray "
      "file, dummy labels). Error calculation with labels: implemented (classification and measurement metrics with guards); "
      "not yet exercised on real labelled patient data.", bold_lead="Status. ")
doc.figure(FIG / "A02_app_dataset_validation.png", "Validation page of the application (real screenshot).", 15)

# ============================================================ PART 10
doc.h("Part 10. Robustness, reliability and efficiency")
doc.table(["Challenge", "Existing solution", "Remaining problem", "Proposed improvement"], [list(r) for r in ROBUST],
          "Robustness against the listed challenges.", widths=[3.0, 5.0, 4.4, 4.0], size=8.5)
mp = V3["ms_parts"]
doc.table(["Measure", "Value (laptop CPU)"],
          [["Time per frame", f"{f(mp['total'], 1)} ms (landmarks {f(mp['landmarks'], 1)}, thermal mask {f(mp['thermal_mask'], 1)}, ROIs {f(mp['rois'], 1)})"],
           ["Throughput", f"≈ {V3['fps']:.0f} fps vs sensor ≈ 8.7 fps"],
           ["Memory", f"{V3['rss_model']:.0f} MB after model load; {V3['rss_peak']:.0f} MB peak"],
           ["Model size", f"hand landmarker {f(V3['model_mb'], 1)} MB; anomaly bundle a few kB"],
           ["Inference per hand", f"{LW['score_us_per_hand'] / 1000:.2f} ms (Ledoit–Wolf)"],
           ["GPU", "not used"],
           ["Edge device", "not measured — no real-time edge claim"]],
          "Computational measurements.", widths=[4.0, 12.4], size=9.5)
doc.h("10.1 Deployment and how to use the deployed system", 2)
doc.p("Release 1.0.0 (dist/rgb_thermal_hand-1.0.0.zip) contains the program, models with SHA-256 checksums, calibration "
      "profiles, configuration without personal paths, tests, installer and documentation; no datasets.")
doc.numbered(["Unzip on a Windows 10/11 PC and run: powershell -ExecutionPolicy Bypass -File .\\install.ps1 (creates Python 3.12 "
              "with the exact package versions; -Offline uses the local package cache).",
              "The installer runs the health check (rgbthermal doctor); it must end with READY.",
              "Start the application with 'RGB-Thermal App.bat'.",
              "Dataset upload → Validate; Calibration → Auto-calibrate (only for a new camera); Processing; Independent test with the "
              "saved model; Joint view; Reports (PDF/Excel with the final joint temperatures, Q1–Q3 and frames).",
              "Command line: rgbthermal.bat calibrate find | verify | apply, rgbthermal.bat test, rgbthermal.bat doctor."])
doc.figure(FIG / "N05_deployment_workflow.png", "From the release zip to results.", 16)
doc.figure(FIG / "N06_doctor_output.png", "Health check of an installation.", 15)
doc.p("A clean installation from the zip passed the health check, the automated tests and the complete new-data workflow "
      "(import → auto-calibration → validation → processing → independent test → reports → joint view → all 10 pages).",
      bold_lead="Verified. ")

# ============================================================ PART 11
doc.h("Part 11. Results, limitations and conclusion (report summary)")
doc.p("The full technical report (RGB_Thermal_Hand_Arthritis_Technical_Research_Report.docx/.pdf) contains every chapter in "
      "detail. The key results:")
doc.table(["RQ", "Result", "Verdict"],
          [["RQ1", f"valid frames static {pct(V3['static_valid'], 0)}, open–close {pct(V3['oc_valid'], 2)}", "Pass"],
           ["RQ2a", f"held-out Dice median {f(V3['dice_median'], 3)}", "Pass"],
           ["RQ2b", f"joint error {f(V3['joint_err_raw'], 1)} → {f(V3['joint_err_corr'], 1)} px (v3); {B9['joint_err_median']:.2f} px with v9 on unseen subjects (target 8)", "Fail (improved)"],
           ["RQ3", f"{pct(V3['roi_on_skin'], 1)} ROIs ≥ 95 % on hand", "Pass (partly by construction)"],
           ["RQ4", f"ICC {f(V3['icc'], 3)}, MDC95 {f(V3['mdc'], 2)} °C", "Pass"],
           ["RQ5", f"mean D2JTD {f(V3['mean10'], 2)} °C (CI {f(V3['mean10_ci'][0], 2)}–{f(V3['mean10_ci'][1], 2)})", "reported"],
           ["RQ7", f"specificity {f(LW['specificity'])}; {PR['real_correct']}/{PR['n_hands']} healthy correct, {PR['warmed_correct']}/{PR['n_hands']} simulated warm joints flagged", "simulation"]],
          "Results summary.", widths=[1.5, 11.0, 3.9], size=9.5)
doc.figure(FIG / "P01_correct_predictions_gallery.png", "Correct outputs on unseen healthy volunteers (measured).", 15.5)
doc.p("Limitations and future work are listed with severity in Part 15. In short: no patients, 17 subjects, 2-D "
      "alignment without a calibration target (joint error 9.97 px with v9, target 8 px), within-session repeatability, one "
      "annotator, laptop-only timing.")
doc.p("The project delivers an audited, reproducible RGB–thermal joint temperature measurement with honest, "
      "subject-independent evaluation and an application ready for new labelled data. It can say whether a hand lies "
      "within the range of these healthy volunteers and which joint makes it unusual; it cannot yet say whether a person "
      "has arthritis.", bold_lead="Conclusion. ")

# ============================================================ PART 12-13
doc.h("Part 12–13. Slide plan and presentation script")
doc.p(f"The viva deck has {len(SLIDES)} slides (about 1 minute each, 25 minutes in total). For every slide: what is on it, what "
      "to say, and the question most likely to follow. The same script is in the deck's speaker notes.")
for n, s in enumerate(SLIDES, 1):
    doc.h(f"Slide {n}. {s['t']}", 2)
    vis = s.get("fig") or ("table" if s.get("tab") else "title / closing layout")
    content = "; ".join(s.get("b", [])) if s.get("b") else (", ".join(s["tab"][0]) + " table" if s.get("tab") else "")
    doc.bullets([("Section: ", s["k"]), ("On the slide: ", content or "figure with one-line caption"), ("Visual: ", str(vis))])
    doc.p(s["s"], bold_lead="Say: ")
    for q, a in s.get("q", []):
        doc.p(a, bold_lead=f"Q: {q} — A: ")

# ============================================================ PART 14
doc.h(f"Part 14. Viva questions and answers ({len(VIVA)})")
cat = None
for n, (c, q, a) in enumerate(VIVA, 1):
    if c != cat:
        doc.h(c, 2)
        cat = c
    para = doc.d.add_paragraph()
    r = para.add_run(f"{n}. {q}")
    r.bold = True
    para.paragraph_format.space_after = Pt(1)
    para.paragraph_format.keep_with_next = True
    doc.p(a)

# ============================================================ PART 15
doc.h("Part 15. Critical review (strict examiner)")
doc.table(["Issue", "Severity", "Why it is a problem", "Effect on results", "Fix", "Before presentation?"], [list(r) for r in REVIEW],
          "Issues ranked by severity.", widths=[2.8, 1.5, 3.2, 3.2, 3.0, 2.7], size=7.5)
doc.h("Improvement roadmap", 2)
doc.numbered(["Now (before the viva): re-click the two mislabelled annotation frames; state the no-patient limitation on the first and last slide.",
              "Next data collection: patients with ultrasound-confirmed diagnosis plus matched healthy controls; record room temperature and a 15-minute acclimatisation.",
              "Same session: a heated-checkerboard recording for stereo calibration; recordings with rings; repeat session on a second day.",
              "Analysis: depth-aware joint mapping; second annotator; visibility labels; locked independent test of the frozen model.",
              "Deployment: benchmark on the target phone/board; then real two-camera streaming."])

doc.h("References")
for i, r in enumerate(REFS, 1):
    para = doc.d.add_paragraph()
    para.add_run(f"[{i}] {r['text']}").font.size = Pt(9.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
doc.d.core_properties.title = "Project Audit and Viva Defence Guide"
doc.d.save(OUT)
print("saved", OUT, "figures", doc.fig_n, "tables", doc.tab_n, "viva", len(VIVA), "slides", len(SLIDES))
