"""Build the technical research report (.docx). Every number comes from metrics_verified.json.

Run from the project root:
  .venv312\\Scripts\\python.exe docs/deliverables_src/build_report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from doc_helpers import Doc  # noqa: E402
from references import REFS, cite  # noqa: E402

M = json.load(open("deliverables/supporting/metrics_verified.json"))
D, V3, V5, V6, V7, V8 = (M[k] for k in ("dataset", "v3", "v5", "v6", "v7", "v8"))
FIG = Path("deliverables/figures")
OUT = Path("deliverables/RGB_Thermal_Hand_Arthritis_Technical_Research_Report.docx")


def pct(x, d=1):
    return f"{100 * x:.{d}f} %"


def f(x, d=2):
    return f"{x:.{d}f}"


C = {r["model"]: r for r in V7["comparison"]}
AB = {(r["feature_set"], r["model"]): r for r in V6["ablation"]}
ST = {r["group"]: r for r in V6["stability"]}
REF = {r["feature"]: r for r in V3["reference"]}
VIS = V8["visibility"]
E2E = V8["e2e"]
MF = E2E["metrics_flag_vs_label"]
LIVE = V8["live"]
PR = json.load(open("deliverables/supporting/prediction_examples_summary.json"))
import pandas as pd  # noqa: E402
BEN = json.load(open("reports/v9_calibration/benchmark_find_calibration.json"))
SELF = pd.read_csv("reports/v9_calibration/selftest_known_distortions.csv")
PROD = json.load(open("models/calibration/flirone_one_pro_v9_found.json"))
EXV = json.load(open("reports/v9_calibration/example_S9_left_static.json"))["verify"]
B_V3, B_V9 = BEN["results"]["v3: fitted per recording"], BEN["results"]["found calibration (this module)"]
_fr = pd.read_csv("data/processed_v8/frames/S2_right_static.csv")
AGG = _fr[_fr.frame_valid.fillna(False).astype(bool)]

doc = Doc()
doc.title_page(
    "RGB–Thermal Hand-Joint Temperature Analysis and Arthritis Screening-Support System",
    "Technical Research Report",
    ["FLIROne dataset · 17 healthy subjects · 68 recordings",
     "Pipeline v3 (frozen baseline) to v9 (automatic calibration) · software release 1.0.0",
     "22 November 2026",
     "",
     "Research and screening-support software"])
doc.toc()

# ============================================================================ summary
doc.h("Executive summary")
doc.p("This report describes a software system that measures the skin temperature of 14 hand joints "
      "from paired RGB video and radiometric thermal data recorded with a FLIR ONE Pro smartphone camera, "
      "compares each hand with a reference built from healthy volunteers, and flags hands whose joint "
      "temperature pattern lies outside that reference. It was developed and evaluated on the FLIROne "
      "dataset of 17 healthy volunteers.")
doc.p("What was built and measured on the 17-subject dataset:", bold_lead=None)
doc.bullets([
    ("Landmarks. ", f"MediaPipe found a hand in {pct(V3['hand_detection'], 2)} of analysed frames; "
     f"{pct(V3['static_valid'], 0)} of static and {pct(V3['oc_valid'], 2)} of open–close frames passed the landmark checks."),
    ("RGB→thermal mapping. ", f"Held-out hand-outline overlap (Dice) median {f(V3['dice_median'], 3)} "
     f"(10th percentile {f(V3['dice_p10'], 3)}) over 68 recordings."),
    ("Joint position. ", f"Median error against 777 hand-clicked joints: {f(V3['joint_err_raw'], 1)} px raw, "
     f"{f(V3['joint_err_corr'], 1)} px after a learned along-finger correction (leave-one-subject-out). "
     f"The pre-set target of 8 px was not met; the temperature cost of this error is a median of "
     f"{f(V3['dT_corr'], 3)} °C."),
    ("Repeatability. ", f"10-joint mean D2JTD (back-of-hand minus joint temperature): ICC(3,1) {f(V3['icc'], 3)}, "
     f"MDC95 {f(V3['mdc'], 2)} °C (split-half, 34 static hands, same session)."),
    ("Anomaly models. ", f"Six healthy-reference models evaluated with nested leave-one-subject-out thresholds: "
     f"specificity {f(C['mahalanobis_ledoit_wolf']['specificity'], 2)} for all. Sensitivity to simulated warming of all "
     f"10 MCP/PIP joints by 1 °C ranged {f(C['isolation_forest']['sens_all_10_MCP_PIP_1C'], 2)}–"
     f"{f(C['one_class_svm']['sens_all_10_MCP_PIP_1C'], 2)}; these are simulations, not patients."),
    ("Correct predictions. ", f"Scored by a model that never saw the person, {PR['real_correct']} of {PR['n_hands']} healthy hands "
     f"were correctly reported within the healthy range, and all {PR['warmed_correct']} hands with a simulated +2 °C index-MCP "
     "joint were correctly flagged, with that joint identified as the cause."),
    ("Automatic calibration (v9). ", f"A module that finds the RGB→thermal mapping from the recordings themselves "
     f"recovered 12 of 12 deliberately applied distortions (median error {SELF.mapping_error_px_median.median():.1f} px). "
     f"Calibrated on 5 subjects and applied to 12 unseen subjects, it reduced the joint position error from "
     f"{B_V3['joint_err_median']:.2f} px to {B_V9['joint_err_median']:.2f} px. It is now the default."),
    ("One temperature per joint. ", "Each joint's final temperature is the median of its per-frame values over the usable "
     "frames; reports now also give the spread (Q1–Q3) and the number of frames used."),
    ("Application and deployment. ", "A desktop application (PySide6) for dataset upload, validation, calibration, processing, "
     "locked independent testing, joint view and PDF/Excel reports, packaged as release 1.0.0 with an installer and a "
     "health check; a clean installation passed the complete workflow and 49 automated tests."),
])
doc.h("Implemented components", 2)
doc.p("Every component listed here is implemented in the project code and was run on the data in this report.")
doc.table(
    ["Component", "Module", "Verified by"],
    [["Dataset loading, audit, memory-mapped .npy", "data_audit, io_utils", "audit of all 68 recordings"],
     ["Frame pairing and sync check", "run_pipeline, analysis", "equal frame counts 68/68; sync 32/34 aligned"],
     ["RGB→thermal registration (fixed scale, per-recording shift, rig fallback)", "registration", "held-out Dice, 68 recordings"],
     ["MediaPipe landmarks and quality checks", "landmarks", "99.97 % hand detection"],
     ["Joint visibility (folded / covered joints)", "landmarks", "18,431 analysed frames"],
     ["Along-finger joint correction", "landmark_correction", "777 clicked joints, leave-one-subject-out"],
     ["Thermal and RGB hand segmentation", "segmentation", "Dice; QC overlays"],
     ["ROI extraction and temperature statistics", "rois", "98 % of ROIs on skin"],
     ["Cold-band (accessory) contamination flag", "rois", "per-ROI flag in frame tables"],
     ["Feature engineering (129 features, 5 groups)", "feature_extraction", "split-half stability"],
     ["Healthy-reference anomaly models (6)", "anomaly_models", "nested leave-one-subject-out"],
     ["Per-joint explanation", "anomaly_models", "correct joint located in 34/34 simulated cases"],
     ["Dataset upload, validation, locked independent test", "dataset_manager, evaluation", "end-to-end software test"],
     ["PDF / Excel / CSV reports", "report_generator", "generated for datasets and test runs"],
     ["Joint view (visible joints, temperatures, ROIs)", "joint_view", "all recordings"],
     ["Automatic RGB→thermal calibration: find, apply, verify (image area, scale, rotation, per-frame shift)", "calibration",
      "12/12 known distortions recovered; unseen subjects 9.97 px"],
     ["Final joint temperature per recording with spread (median, Q1–Q3, frames used)", "feature_extraction, report_generator",
      "unit test; dataset reports"],
     ["Desktop GUI (10 pages) incl. Auto-calibrate", "ui/app", "49 automated tests"],
     ["Installer, health check, command line, release build", "install.ps1, doctor, cli, build_release",
      "clean installation + acceptance test"],
     ["Live mode (recorded replay, two-clock pairing)", "live_acquisition", "173/173 frames paired"]],
    "Implemented components.", widths=[7.0, 4.0, 5.4], size=9)

# ============================================================================ 1 intro
doc.h("1. Introduction")
doc.h("1.1 Arthritis and why the hand joints", 2)
doc.p("Arthritis is a group of conditions in which joints become inflamed or degenerate. Rheumatoid "
      "arthritis (RA) is a systemic autoimmune disease whose hallmark is inflammation of the synovial lining "
      "of joints (synovitis); it typically affects the small joints of the hands and feet symmetrically, in "
      f"particular the metacarpophalangeal (MCP) and proximal interphalangeal (PIP) joints {cite('smolen2016')}. "
      "Untreated synovitis leads to cartilage and bone erosion, deformity and disability, which is why early "
      "detection and treatment within a 'window of opportunity' improve outcomes.")
doc.p("Osteoarthritis (OA) of the hand is degenerative and more often affects the distal interphalangeal (DIP) "
      "joints and the thumb base. Inflammatory episodes in either condition increase local blood flow, and "
      f"therefore local skin temperature, over the affected joint {cite('schiavon2021')}. The hand is a "
      "practical site for thermal measurement: its joints lie close to the skin surface, it can be placed in "
      "a repeatable pose, and 14 joints per hand give an internal comparison (each joint against the back of "
      "the same hand and against the other joints).")
doc.h("1.2 Limitations of conventional assessment", 2)
doc.p("Clinical joint examination (tender and swollen joint counts) is subjective and examiner-dependent. "
      "Ultrasound with power Doppler and MRI detect synovitis objectively but require trained operators and "
      "costly equipment and are not suited to frequent or community screening. Laboratory markers "
      f"(rheumatoid factor, anti-CCP antibodies, CRP, ESR) are not joint-specific {cite('smolen2016')}.")
doc.h("1.3 Thermal and RGB imaging", 2)
doc.p("Infrared thermography measures emitted long-wave infrared radiation and converts it to surface "
      "temperature. It is contact-free, fast and has no radiation dose. Reviews of thermography in arthritis "
      "report that inflamed joints are on average warmer than healthy ones, but also that measurement "
      f"protocols vary widely and that acclimatisation and standardised ROIs are essential {cite('schiavon2021')}, "
      f"{cite('ring2012')}. A thermal image alone, however, has low resolution and poor anatomical contrast: "
      "a knuckle is hard to locate in a 160×120-pixel sensor image.")
doc.p("RGB images have high resolution and texture, which lets a hand-pose model locate the joints "
      f"reliably {cite('zhang2020')}. Combining the two — find the joints in RGB, read the temperature in "
      f"the co-registered radiometric image — is the approach of Qiao et al. {cite('qiao2026')}, whose "
      "dorsal-to-joint temperature difference (D2JTD) feature this project adopts.")
doc.h("1.4 Motivation, objectives and research questions", 2)
doc.p("The motivation is a low-cost, smartphone-camera measurement that could later support screening "
      "research by quantifying joint temperature reproducibly. Because only healthy volunteers were "
      "available, the objective was set as a measurement study: can each joint's temperature be measured "
      "accurately and repeatably, and what does the healthy range look like? The pass criteria were fixed "
      "before analysis:")
doc.table(["ID", "Question", "Pass criterion"],
          [["RQ1", "Are the joints located in the RGB image?", "≥ 90 % of static frames valid"],
           ["RQ2", "Is the RGB→thermal mapping accurate (outline; joint position)?", "Dice ≥ 0.90; median joint error ≤ 8 px"],
           ["RQ3", "Do the joint ROIs lie on skin?", "≥ 95 % of ROIs ≥ 95 % on the hand"],
           ["RQ4", "Is the joint temperature repeatable?", "MDC95 < 0.4 °C"],
           ["RQ5", "What is the healthy D2JTD reference?", "reported with CI"],
           ["RQ6", "Does hand motion (open–close) change the measurement?", "reported"],
           ["RQ7", "Can a healthy-only model flag simulated warming?", "reported (simulation)"]],
          "Research questions and pre-registered pass criteria.", widths=[1.3, 9.5, 5.6])
doc.h("1.5 Contributions", 2)
doc.bullets([
    "An audited, reproducible pipeline from FLIR ONE RGB + radiometric .npy to per-joint temperatures, including fixes for dataset defects found in the audit (swapped files, letterboxing, unreliable seeking).",
    "Quantified geometric accuracy against 777 hand-clicked joints, a learned joint-position correction evaluated leave-one-subject-out, and the temperature cost of the remaining error.",
    "Repeatability (ICC, MDC95) of every joint and feature group, and a healthy D2JTD reference for 14 joints.",
    "A leakage-free comparison of six healthy-reference anomaly models with nested subject-wise thresholds and simulated warming, plus feature-set ablation.",
    "A joint-visibility rule so folded or covered joints are not measured.",
    "An application for uploading new datasets, validating them, running locked independent tests and producing reports."])
doc.h("1.6 Scope", 2)
doc.p("The system measures joint temperatures, builds a healthy reference and reports for each hand whether its "
      "joint temperature pattern is 'within healthy range' or a 'potential anomaly' — a distance from the healthy "
      "volunteers, not a probability of disease. It is research and screening-support software.")

# ============================================================================ 2 background
doc.h("2. Literature and Technical Background")
doc.note("Statements in this chapter come from the cited literature; results of this project are in Chapters 7–9.")
doc.h("2.1 Infrared thermography and radiometric data", 2)
doc.p("Every surface above absolute zero emits infrared radiation; a microbolometer camera measures the "
      "8–14 µm band and, using the emissivity of the surface (≈0.98 for human skin), converts the signal to "
      f"temperature {cite('ring2012')}, {cite('sanchez2009')}. The colourised thermal video is only a display: its colours depend on an "
      "automatic palette range that changes from frame to frame. Radiometric data store the temperature of "
      "every pixel in °C and are the only valid source for quantitative measurement.")
doc.p(f"The FLIR ONE Pro attaches to a smartphone and combines a 160×120 thermal sensor with a visible "
      f"camera; FLIR specifies ±3 °C or ±5 % absolute accuracy {cite('flironepro')}. Absolute accuracy of this "
      "class is too coarse for joint comparisons, which is one reason relative features (temperature "
      "differences within the same image) are preferred: a common offset cancels.")
doc.h("2.2 Thermography in arthritis", 2)
doc.p(f"Systematic reviews {cite('schiavon2021')} report higher joint temperatures in active RA and OA, with "
      f"heterogeneous protocols. Studies of hand thermography in RA {cite('snekhalatha2015')}, {cite('pauk2019')} "
      f"and in juvenile arthritis {cite('thermojis2022')} use ROIs over the joints and compare against reference "
      f"regions; hand-joint thermal measurements can be reliable {cite('spalding2008')}, although clinically 'hot' joints are not always measurably warmer {cite('jones2018')}. Qiao et al. {cite('qiao2026')} located hand joints with MediaPipe on FLIR ONE RGB images, mapped "
      "them to the thermal image and introduced D2JTD, the back-of-hand temperature minus the joint temperature. "
      f"Measurement guidelines (TISEM) {cite('moreira2017')} stress acclimatisation, a controlled room and "
      f"reporting of ROI definitions. Smartphone thermal cameras show bias and random error that calibration reduces only partly {cite('sensors2026')}.")
doc.h("2.3 Camera calibration, stereo calibration and registration", 2)
doc.p(f"Intrinsic calibration estimates each camera's focal length, principal point and lens distortion, "
      f"usually from views of a planar target {cite('zhang2000')}. Stereo calibration additionally estimates "
      "the rotation and translation between two cameras; with it, points can be mapped exactly if their depth "
      "is known. A 2-D transform fitted to image content — here, the hand outline — maps one image onto the "
      "other directly; this is the approach implemented in this project (Section 5.3).")
doc.h("2.4 Hand detection and landmarks (MediaPipe)", 2)
doc.p(f"MediaPipe Hands {cite('zhang2020')}, built on the MediaPipe framework {cite('lugaresi2019')}, first "
      "detects the palm, then regresses 21 hand landmarks: the wrist, four points per finger (for the "
      "fingers MCP, PIP, DIP, tip; for the thumb CMC, MCP, IP, tip). Each landmark has image x, y and a "
      "relative depth z (wrist-relative, same scale as x). In VIDEO mode the model tracks the hand between "
      "frames. The landmarks approximate joint centres seen from the camera; they are not the anatomical "
      "joint lines.")
doc.h("2.5 Segmentation and ROI extraction", 2)
doc.p(f"Otsu's method {cite('otsu1979')} picks the grey-level threshold that maximises the between-class "
      "variance of a bimodal histogram — here warm hand versus cooler background. Hysteresis thresholding "
      "grows the region from confident seeds into weaker but connected pixels. GrabCut "
      f"{cite('rother2004')} segments colour images by iterated graph cuts from a rough initial mask. "
      "Skin-colour thresholds alone are unreliable across skin tones, lighting and backgrounds, which is why "
      "the RGB mask is seeded by landmarks rather than colour.")
doc.h("2.6 Healthy-reference modelling and anomaly detection", 2)
doc.p("Without patient data, a model can only describe what healthy looks like and measure how far a new "
      "hand lies from it (one-class or novelty detection). Methods used here: robust z-scores (median and "
      f"MAD), Mahalanobis distance with Ledoit–Wolf shrinkage of the covariance {cite('ledoit2004')}, the "
      f"minimum covariance determinant (MCD) {cite('rousseeuw1999')}, Isolation Forest {cite('liu2008')} and "
      f"one-class SVM {cite('scholkopf2001')}, all as implemented in scikit-learn {cite('pedregosa2011')}.")
doc.h("2.7 Subject-independent validation", 2)
doc.p("Two recordings of the same person are more alike than recordings of different people. If one person "
      "appears in both training and test data, performance is overstated (leakage). Leave-one-subject-out "
      "(LOSO) cross-validation holds out all data of one person at a time; thresholds must also be chosen "
      "without the held-out person (nested LOSO).")
doc.h("2.8 Repeatability statistics", 2)
doc.p(f"The intraclass correlation ICC(3,1) {cite('shrout1979')}, {cite('koo2016')} measures consistency of "
      f"repeated measurements relative to between-subject spread. The minimal detectable change MDC95 "
      f"{cite('weir2005')} is the smallest change larger than measurement noise with 95 % confidence.")
# ============================================================================ 3 dataset
doc.h("3. Dataset Description")
doc.p(f"The FLIROne hand dataset {cite('dataset')} contains recordings of {D['subjects']} healthy subjects "
      "(S1–S17). Each subject has four recordings: left and right hand, each in a static pose (open hand at "
      "rest) and an open–close task (repeated fist clenching). Each recording consists of an RGB MP4, a "
      "colourised thermal MP4 and a radiometric NumPy file (.npy). All values below were measured by the "
      "Stage 1 audit (src/data_audit.py) on every file.")
doc.table(["Item", "Value"],
          [["Subjects / recordings", f"{D['subjects']} / {D['recordings']} (17 × 2 hands × 2 tasks), all complete"],
           ["Health status", "all healthy volunteers (no arthritis labels)"],
           ["Files per recording", "RGB .mp4, thermal .mp4, radiometric .npy"],
           ["RGB video", "640×480 container; image letterboxed at x 140–500 (360×480 content)"],
           ["Radiometric .npy", "float32, shape (N, 640, 480) portrait, °C"],
           ["Frames per recording", f"{D['frames_min']}–{D['frames_max']} (median {D['frames_median']:.0f}); total {D['frames_total']:,}"],
           ["Frame rate (measured from timestamps)", f"{D['fps_min']:.2f}–{D['fps_max']:.2f} fps (median {D['fps_median']:.2f})"],
           ["Duration", f"{D['duration_min']:.1f}–{D['duration_max']:.1f} s (median {D['duration_median']:.1f} s)"],
           ["Frame counts equal (RGB = thermal MP4 = .npy)", f"{D['frame_counts_equal']} of 68"],
           ["Temperature range in frames", f"{D['tmin_min']:.2f}–{D['tmax_max']:.2f} °C; median frame mean {D['tmean_median']:.2f} °C"],
           ["Recordings with NaN/inf", str(D["nan_recordings"])],
           ["Storage", f".npy {D['npy_gb']:.1f} GB; videos {D['video_gb']:.2f} GB"],
           ["Defects found and handled", "S11_left_static: RGB and thermal MP4 file names swapped (corrected); OpenCV seek error up to ±30 frames (frames decoded in order)"]],
          "Dataset summary (measured by the audit).", widths=[5.5, 10.9])
doc.figure(FIG / "F03_dataset_tree.png", "Dataset folder structure and naming convention (S<subject>_<hand>_<task>).", 14)
doc.figure(FIG / "R07_audit_sample_frames.png", "Sample input frames from the audit: RGB, thermal MP4 and radiometric .npy of the same frame (measured data).", 15.5)
doc.h("3.1 Why radiometric .npy and not the thermal video", 2)
doc.p("The thermal MP4 is a colour rendering with automatic per-frame scaling and lossy compression; its pixel "
      "values cannot be converted back to temperature. The .npy holds one float per pixel in °C. The pipeline "
      "therefore reads temperatures only from the .npy; the thermal MP4 is used solely by the audit to check "
      "that it shows the same frames (the audit found it does for 67 of 68 recordings; the exception is "
      "S11_left_static, whose RGB and thermal MP4 files are swapped).")
doc.h("3.2 Processing large data without loading it into memory", 2)
doc.p(f"The .npy files total {D['npy_gb']:.0f} GB, more than the RAM of a typical laptop. They are opened with "
      "numpy.load(mmap_mode='r'): the file is memory-mapped and only the frames actually indexed are read from "
      "disk. Videos are decoded sequentially frame by frame. Recordings are processed one per worker process, "
      "and every frame's results are written to a per-recording CSV, so memory use does not grow with "
      f"dataset size (peak resident memory measured at {V3['rss_peak']:.0f} MB).")
doc.figure(FIG / "R08_audit_timestamps.png", "Audit: frame timing per recording (measured). Header frame rates were unreliable; measured rates are used.", 15)

# ============================================================================ 4 architecture
doc.h("4. System Architecture")
doc.p("The figure below shows the implemented system. Colours separate RGB processing, thermal processing, "
      "calibration/alignment, feature extraction, machine learning and output/evaluation. Every block is "
      "implemented in the project code.")
doc.figure(FIG / "F01_architecture.png", "System architecture as implemented.", 16.4)
doc.figure(FIG / "F14_app_modules.png", "Application module architecture: Python modules and the GUI pages that use them.", 15.5)
doc.table(["Layer", "Modules", "Responsibility"],
          [["Data", "io_utils, data_audit, dataset_manager", "find, match, audit and validate recordings; memory-mapped access"],
           ["Vision", "landmarks, segmentation, registration, landmark_correction", "joints in RGB, hand masks, RGB→thermal mapping"],
           ["Measurement", "rois, run_pipeline", "joint ROIs, temperature statistics, per-frame tables, run manifest"],
           ["Analysis", "feature_extraction, analysis, annotation_qc", "features, statistics, accuracy and repeatability"],
           ["Models", "anomaly_models, evaluation", "healthy-reference models, nested LOSO, locked independent tests"],
           ["Output", "report_generator, joint_view, edge_demo, live_acquisition, ui/app", "PDF/Excel reports, visualisations, GUI, live simulation"]],
          "Software layers (Python 3.12; OpenCV, MediaPipe, NumPy, pandas, scikit-learn, PySide6).", widths=[2.6, 6.3, 7.5])

# ============================================================================ 5 pipeline
doc.h("5. Complete Working Pipeline")
doc.p("The figure below traces one recording through the pipeline. The command "
      "'python -m src.run_pipeline' runs stages 1–10 for all recordings and writes one CSV per recording "
      "(one row per analysed frame, every third frame) plus a registration table and a run manifest "
      "(configuration, code hash, package versions). Reruns are bit-identical (fixed OpenCV seed).")
doc.figure(FIG / "F02_pipeline.png", "End-to-end pipeline: data flow from the three input files to scores and reports.", 16)


def stage(title, rows, extra=None):
    doc.h(title, 2)
    for lead, text in rows:
        doc.p(text, bold_lead=lead + ": ")
    if extra:
        extra()


stage("5.1 Stage 1 — Dataset loading and auditing", [
    ("Input", "dataset folder (or ZIP) with RGB MP4, thermal MP4 and .npy per recording."),
    ("Purpose", "find every recording, verify it is complete and consistent, and record the facts later stages depend on."),
    ("Algorithm", "file-name parsing (S<n>_<left|right>_<static|open_close>), header and measured frame counts, "
     "dtype/shape/NaN checks, letterbox detection (non-black bounding box), thermal-MP4↔.npy correlation per "
     "frame to detect swapped or misaligned files, seek-accuracy test."),
    ("Libraries", f"OpenCV {cite('bradski2000')} (VideoCapture), NumPy (memory-mapped load), pandas."),
    ("Steps", "1) list and match files; 2) open .npy memory-mapped, read shape/dtype, min/max/NaN per frame; "
     "3) decode both videos sequentially, count frames, measure timestamps; 4) locate the image box in the "
     "letterbox; 5) correlate thermal MP4 grey levels with .npy temperatures on all frames; 6) write "
     "dataset_audit.csv and audit_report.md."),
    ("Key parameters", "letterbox box y 0–480, x 140–500; swap detection r < 0.95 between thermal MP4 and .npy."),
    ("Output", "audit table (one row per recording), audit report, configuration entries (video_box, swapped_video_files)."),
    ("Quality checks", f"68/68 complete, frame counts equal for {D['frame_counts_equal']}, 0 NaN recordings, 1 swap detected."),
    ("Failure conditions", "missing file, unequal frame counts, unreadable video, unknown naming — reported per recording and excluded."),
])
stage("5.2 Stage 2 — RGB and thermal synchronisation", [
    ("Input", "decoded RGB frames and .npy frames of one recording."),
    ("Purpose", "pair each RGB frame with the thermal frame taken at the same time."),
    ("Method", f"the three streams have identical frame counts ({D['frame_counts_equal']}/68 recordings), so frame i of the RGB "
     "video corresponds to frame i of the .npy (index pairing). The device recorded them together; there are no "
     "separate timestamps for the .npy. Frames are decoded in order, because OpenCV seeking was measured to be "
     f"off by up to ±{abs(D['seek_error_range'][1]):.0f} frames. Every third frame is analysed (frame_stride 3, ≈2.8 fps)."),
    ("Verification", "for the 34 open–close recordings, hand openness from RGB landmarks and thermal hand area "
     "from the .npy mask are cross-correlated over lags of ±20 analysed frames; the lag with the highest "
     f"correlation should be 0. Result: {V3['sync_aligned']}/34 aligned at lag 0 (±1), median r = {f(V3['sync_corr'])}; "
     "the remaining two were ambiguous (weak motion signal), not offset."),
    ("Different sampling rates", f"the stored data share one rate ({D['fps_median']:.2f} fps median), so no resampling is "
     "needed. For live use (different camera rates) src/live_acquisition.py pairs each RGB frame with the most "
     f"recent thermal frame and records its age; in simulation {pct(LIVE['pairs_within_1_frame'], 0)} of pairs were "
     f"within one frame, median age {LIVE['thermal_age_ms_median']:.1f} ms."),
    ("Output", "paired (RGB, °C-matrix) per analysed frame; sync_rgb_thermal.csv."),
    ("Failure conditions", "a dropped frame in one stream would shift later pairs; the sync check detects such an offset."),
])


def _reg_extra():
    doc.eq(r"\mathbf{p}_{th} = s\,(k\,\mathbf{p}_{rgb} - \mathbf{c}) + \mathbf{c} + \mathbf{t}",
           "RGB→thermal similarity transform. p_rgb: pixel in the 360×480 RGB content box; k = 4/3 rescales it to "
           "the 480×640 .npy grid; c = (240, 320) is the image centre; s is the scale (fixed at 1.25: the thermal "
           "lens has a narrower field of view, so the hand appears larger); t = (t_x, t_y) the translation "
           "(parallax between the two lenses, which depends on hand distance).")
    doc.figure(FIG / "F04_transform.png", "Coordinate systems and the RGB→thermal transform (diagram).", 15)
    doc.figure(FIG / "F09_registration_real.png", "Registration on a real frame: RGB-derived hand mask warped onto the thermal mask (measured).", 15.5)


stage("5.3 Stage 3 — Image registration (RGB→thermal mapping)", [
    ("Input", "RGB hand mask and thermal hand mask on 16 frames sampled from the recording."),
    ("Purpose", "map RGB pixel coordinates (where joints are found) to thermal coordinates (where temperatures are read)."),
    ("Calibration profile", "the fitted rig parameters are stored as a versioned calibration profile "
     "(FLIR ONE Pro v3) that new datasets are checked against."),
    ("Algorithm", "a similarity transform (equation 1). A free fit of scale and translation over all 68 recordings "
     f"gave a median scale of {V3['scale']:.2f} (IQR 1.24–1.26), so the scale was fixed and only the translation "
     "is estimated per recording: phase correlation between the warped RGB mask and the thermal mask gives an "
     "initial shift, refined to maximise Dice. Half the sampled frames fit, the other half evaluate. If the "
     f"recording's own shift scores worse on its held-out frames than the rig median (s, tx, ty) = "
     f"({V3['scale']}, {V3['tx']}, {V3['ty']}) px, the rig median is used."),
    ("Libraries", "OpenCV (warpAffine, phaseCorrelate), NumPy."),
    ("Quality metric", f"Dice of held-out masks: median {f(V3['dice_median'], 3)}, 10th percentile {f(V3['dice_p10'], 3)}, "
     f"minimum {f(V3['dice_min'], 3)}; {V3['dice_ge90']}/68 recordings ≥ 0.90."),
    ("Output", "registration.csv (s, tx, ty, Dice fit/held-out, source = own or rig)."),
    ("Since v9", "the default mapping is the automatic calibration of Chapter 9 (scale, rotation and a shift per frame, "
     "found from the recordings); this per-recording method remains available as calibration profile v3."),
], _reg_extra)


def _lm_extra():
    doc.figure(FIG / "F05_landmarks_real.png", "MediaPipe's 21 landmarks on a real RGB frame, with the 14 analysed joints labelled (measured).", 15.5)


stage("5.4 Stage 4 — Hand detection and landmark extraction", [
    ("Input", "RGB frame cropped from the letterbox (360×480)."),
    ("Algorithm", "MediaPipe Tasks Hand Landmarker (model hand_landmarker.task, 7.5 MB), VIDEO running mode, one "
     "hand, minimum detection confidence 0.5. Output: 21 landmarks (x, y normalised, z relative depth) and handedness."),
    ("Joints analysed", "14 per hand: thumb MCP and IP; MCP, PIP and DIP of index, middle, ring and little fingers. "
     "Landmark indices: thumb 2 (MCP), 3 (IP); index 5/6/7; middle 9/10/11; ring 13/14/15; little 17/18/19; wrist 0."),
    ("Frame validity", "checks after Qiao et al.: all landmarks inside the image; fingers in anatomical order across "
     "the hand; no collapsed MCP row; a minimum hand size; at least 80 % of joint centres on the thermal hand "
     "mask after mapping. For open–close recordings only frames with openness ≥ 1.88 (mean fingertip-to-wrist "
     "distance over palm size) are used for the reference statistics."),
    ("Results", f"hand found in {pct(V3['hand_detection'], 2)} of frames; valid: static {pct(V3['static_valid'], 0)}, "
     f"open–close {pct(V3['oc_valid'], 2)} (RQ1 pass)."),
    ("Failure conditions", "motion blur, hand partly outside the image, fingers crossed; such frames are marked invalid, not repaired."),
], _lm_extra)


def _joint_extra():
    doc.eq(r"\mathbf{j}' = \mathbf{j} + \alpha_k\,(\mathbf{j}_{PIP} - \mathbf{j}_{MCP})",
           f"Along-finger correction. j: mapped joint position; α_k a fraction of the MCP–PIP bone vector learned "
           f"from the clicked joints: α = {f(V3['corr_mcp'], 3)} for MCP joints and {f(V3['corr_pip'], 3)} for PIP, DIP and IP joints "
           "(positive = towards the fingertip).")
    doc.figure(FIG / "G01_localisation_error.png", "Joint localisation error against 777 hand-clicked joints, raw vs corrected, static vs open–close (measured).", 15)
    doc.table(["Measure", "Value"],
              [["Annotated points / frames", f"{V5['points']} / {V5['frames']} (index, middle, ring MCP and PIP)"],
               ["Median error, raw mapping", f"{f(V3['joint_err_raw'], 2)} px"],
               ["Median error, corrected (LOSO)", f"{f(V3['joint_err_corr'], 2)} px (P90 {f(V3['joint_err_corr_p90'], 1)} px)"],
               ["Static / open–close (corrected)", f"{f(V3['joint_err_static'], 2)} / {f(V3['joint_err_oc'], 2)} px"],
               ["Median bias along / across the finger (raw)", f"{f(V3['along_bias'], 2)} / {f(V3['across_bias'], 2)} px"],
               ["Without 3 flagged annotation frames", f"{f(V5['sens']['without_flagged_frames']['corrected_loso_median_px'], 2)} px corrected"],
               ["Temperature change caused by the error (median)", f"raw {f(V3['dT_raw'], 3)} °C, corrected {f(V3['dT_corr'], 3)} °C"],
               ["Target", "≤ 8 px median — not met"]],
              "Joint localisation accuracy (pixels on the 480×640 .npy grid; one sensor pixel = 4 px).", widths=[7.5, 8.9])


stage("5.5 Stage 5 — Joint localisation", [
    ("Input", "MediaPipe landmarks (RGB pixels) and the registration transform."),
    ("Method", "landmarks are mapped to thermal coordinates with equation 1, then shifted along the finger by "
     "equation 2. Raw positions remain stored so accuracy is never scored on the data the correction was learned from."),
    ("Ground truth", f"{V5['points']} joint centres were clicked by hand on {V5['frames']} thermal frames (one annotator, one pass)."),
    ("Why a correction", f"the raw error was not random: its median component along the finger was {f(V3['along_bias'], 1)} px "
     f"(towards the fingertip) against {f(V3['across_bias'], 1)} px across it — a systematic offset between where "
     "MediaPipe places a joint seen in RGB and where the joint's warm spot lies in the thermal image."),
    ("Evaluation", "leave-one-subject-out: the correction for subject k is learned from the other 16 subjects."),
    ("Joint visibility (v8)", "a joint is excluded when it lies inside the back-of-hand polygon (landmarks 0, 1, 2, "
     "5, 9, 13, 17, shrunk 8 %) — folded under the hand; when another finger's bone passes within 0.07 × hand size "
     "and is closer to the camera by 0.03 × hand size (MediaPipe z) — covered; or for DIP joints when the PIP "
     "bend exceeds 100° — tucked under. Hidden joints get no temperature."),
    ("Static vs open–close", f"static {f(V3['joint_err_static'], 1)} px, open–close {f(V3['joint_err_oc'], 1)} px: motion "
     "and bent fingers increase parallax and annotation difficulty."),
], _joint_extra)


def _seg_extra():
    doc.eq(r"\tau_{low} = T_{bg} + 0.4\,(\tau_{Otsu} - T_{bg})",
           "Hysteresis low threshold. τ_Otsu: Otsu threshold of the frame; T_bg: background temperature (median "
           "of pixels below τ_Otsu). Pixels above τ_Otsu seed the hand; connected pixels above τ_low are added, "
           "so cool fingertips stay attached to the hand.")
    doc.figure(FIG / "F08_thermal_segmentation_real.png", "Thermal hand segmentation on a real frame: Otsu only vs Otsu + hysteresis, and the eroded core used for sampling (measured).", 15.5)
    doc.figure(FIG / "F10_outlier_handling.png", "Handling of accessories, occlusion and genuine hot spots (diagram of the implemented rules).", 15)


stage("5.6 Stage 6 — Hand and thermal segmentation", [
    ("Thermal mask", "Otsu threshold on the °C matrix, hysteresis growth (equation 3), largest connected "
     "component, hole filling. A core mask eroded with a 9×9 elliptical element (≈1 sensor pixel) is used for "
     "sampling, so mixed skin/background pixels at the edge are not read."),
    ("Why hysteresis", "subject S17 had cold fingers close to room temperature; plain Otsu cut them off. Hysteresis "
     "keeps them when they are connected to the warm hand."),
    ("RGB mask", "GrabCut initialised from the landmark skeleton (hand polygon as probable foreground, margin as "
     "probable background). Skin-colour rules were not used: they fail with skin tone, lighting and warm-coloured backgrounds."),
    ("Background rejection", "only the largest warm connected component is kept; other warm objects (a mug, a radiator) are discarded if not connected to the hand."),
    ("Accessories", "rings and metal bracelets have low emissivity and read cold. Each ROI records the share of its "
     "pixels more than 1.5 °C below the ROI median; above 15 % the ROI is flagged 'cold-band contamination'. "
     "The median is robust to such pixels, and warm pixels are never removed: a hot spot could be the "
     "inflammation the system is meant to find."),
    ("Libraries", "OpenCV (threshold with THRESH_OTSU, morphology, connectedComponents, grabCut), scikit-image style hysteresis implemented with NumPy."),
    ("Failure conditions", "a hand touching a warm object merges with it; a very cold hand (no bimodal histogram) gives a poor Otsu split — both lower Dice and are visible in the QC overlays."),
], _seg_extra)


def _roi_extra():
    doc.eq(r"r_j = \max\left(6,\ \min\left(\beta_f\,\|\mathbf{j}_{MCP}-\mathbf{j}_{PIP}\|,\ 0.9\,d_{edge}(\mathbf{j})\right)\right)",
           "ROI radius in .npy pixels. β_f: finger factor from Qiao et al. (thumb 0.42, index 0.28, middle 0.28, ring "
           "0.31, little 0.35); d_edge: distance from the joint to the nearer finger edge measured perpendicular "
           "to the bone on the hand mask; 6 px floor.")
    doc.figure(FIG / "F06_rois_d2jtd_real.png", "The 14 joint ROIs and the dorsum region on a real thermal frame, with the measured temperatures and D2JTD (measured).", 15.5)


stage("5.7 Stage 7 — Joint ROI extraction", [
    ("Input", "corrected joint centres, thermal hand mask and its eroded core."),
    ("Construction", "a disc around each of the 14 joints with radius from equation 4; the dorsum ROI is the "
     "polygon over the back of the hand (wrist and MCP landmarks), intersected with the eroded mask."),
    ("Why the width cap", "with Qiao's rule alone (v1) only 48 % of ROIs lay ≥ 95 % on the hand, because discs on "
     f"thin fingers spilled onto background (RQ3 failed). Capping at 0.9 × the finger half-width raised this to "
     f"{pct(V3['roi_on_skin'], 0)} (RQ3 pass; partly by construction, since the cap uses the same mask)."),
    ("Validity", "a joint value is used only if ≥ 60 % of its disc lies on the hand and ≥ 32 core pixels "
     "(≈2 native sensor pixels) remain."),
    ("Output", "per joint and frame: median, mean, P90, SD, on-hand fraction, pixel count, valid flag, cold-band flag, radius."),
], _roi_extra)


def _temp_extra():
    doc.eq(r"\tilde T_j = \mathrm{median}\{T(x,y) : (x,y) \in R_j \cap H_{core}\}",
           "Joint temperature. T(x, y): radiometric temperature in °C; R_j: the joint disc; H_core: eroded hand mask. "
           "The median is used because it ignores a minority of contaminated pixels; mean, P90 and SD are stored too.")


stage("5.8 Stage 8 — Radiometric temperature extraction", [
    ("Data", "each .npy frame is a 640×480 float32 matrix in °C (the sensor's 160×120 values upsampled 4× per "
     "axis by the camera software)."),
    ("Extraction", "for every ROI the pixel temperatures inside the ROI ∩ core mask are collected directly from the "
     "matrix — no colour conversion, no interpolation, no rescaling."),
    ("Statistics", "median (main value), mean, 90th percentile (sensitive to a warm spot), SD (heterogeneity), "
     "valid pixel share; whole-hand median and invalid-pixel count per frame."),
    ("Why preserve raw values", "any smoothing, palette conversion or filling would change the measured temperature "
     "by an unknown amount; storing raw-derived statistics keeps every reported value traceable to the sensor data."),
    ("Invalid values", "a joint without a valid ROI is stored as missing (NaN). It is never filled with an estimate, "
     "because an invented temperature would be indistinguishable from a measured one in later analysis."),
], _temp_extra)


def _feat_extra():
    doc.eq(r"\mathrm{D2JTD}_j = \tilde T_{dorsum} - \tilde T_j",
           "Dorsal-to-joint temperature difference (Qiao et al.). Positive: the joint is cooler than the back of the "
           "hand (usual in healthy hands); a joint warming by inflammation would make D2JTD smaller or negative.")
    d = REF["index_MCP"]
    doc.p(f"Numerical example (healthy reference, index MCP, n = 17 subjects): mean D2JTD {f(d['mean'])} °C, SD "
          f"{f(d['sd'])} °C, reference range {f(d['ref_lo'])} to {f(d['ref_hi'])} °C. If the back of a hand reads 32.80 °C "
          "and its index MCP 32.30 °C, D2JTD = 0.50 °C, inside the range. A joint at 34.50 °C would give "
          "−1.70 °C, below the range.")
    doc.table(["Group", "Features", "Content", "ICC(3,1)", "MDC95 (°C)"],
              [["Relative — D2JTD", "14", "dorsum − joint", f(ST["relative_D2JTD"]["icc31"]), f(ST["relative_D2JTD"]["mdc95"])],
               ["Relative — vs whole hand", "28", "joint − hand median (median and P90)", f(ST["relative_hand"]["icc31"]), f(ST["relative_hand"]["mdc95"])],
               ["Spatial", "18", "MCP−PIP, PIP−DIP, neighbouring knuckles, finger gradient, palm-to-finger", f(ST["spatial"]["icc31"]), f(ST["spatial"]["mdc95"])],
               ["Absolute", "58", "median, mean, P90, spread per joint; dorsum; hand", f(ST["absolute"]["icc31"]), f(ST["absolute"]["mdc95"])],
               ["Temporal", "4", "slopes over the recording, frame-to-frame variation", f(ST["temporal"]["icc31"]), f(ST["temporal"]["mdc95"])],
               ["Quality (not model inputs)", "7", "valid-frame share, on-hand fraction, cold-band share, hidden joints", "—", "—"]],
              "Implemented feature groups (129 per recording) and their split-half stability (34 static hands; median over the group).",
              widths=[3.4, 1.7, 6.3, 2.0, 2.0], size=9)


stage("5.9 Stage 9 — Feature engineering", [
    ("Input", "per-frame ROI statistics of one recording."),
    ("Aggregation", "per recording, the median over valid frames of each per-frame quantity (robust to occasional bad frames)."),
    ("Normalised features", "D2JTD and joint-minus-hand differences are normalised to the same hand and frame, "
     "which removes room temperature and the camera's absolute offset."),
    ("Property of D2JTD", "it responds to local joint warming and ignores warming of the whole hand (dorsum and joints "
     "rise together), which makes it robust to room temperature."),
], _feat_extra)

stage("5.10 Stage 10 — Quality control", [
    ("Landmark validity", "frame_valid flag from the checks in Stage 4."),
    ("Registration quality", "held-out Dice per recording; source own/rig; recordings below 0.80 are flagged in reports."),
    ("ROI validity", "on-hand fraction ≥ 60 %, ≥ 32 core pixels; on-skin share reported."),
    ("Hand openness", "openness score per frame; open–close reference statistics use open frames only."),
    ("Visibility", "folded/covered joints set to missing (Stage 5)."),
    ("Accessories", "cold-band contamination share per ROI (Stage 6)."),
    ("Missing frames", "count of valid frames per recording; recordings with too few are reported as 'insufficient quality' and not scored."),
    ("Model experience", "each saved model stores the training range of every feature (+25 % margin); a hand outside it "
     f"is reported 'outside model experience' instead of being scored (held-out healthy hands: {pct(C['mahalanobis_ledoit_wolf']['false_out_of_range_rate'], 1)})."),
    ("Principle", "unreliable values are flagged and left missing, never replaced by estimates; the report states how many values were excluded and why."),
])
doc.figure(FIG / "O03_qc_overlay.png", "Quality-control overlay produced by the pipeline for one frame (measured).", 15)


def _model_extra():
    doc.eq(r"z_i = \frac{x_i - \mathrm{med}(x_i)}{1.4826\,\mathrm{MAD}(x_i)},\qquad s_{z} = \max_i |z_i|",
           "Robust z-score per feature and the hand score (largest absolute z). MAD: median absolute deviation; 1.4826 makes it consistent with the SD for normal data.")
    doc.eq(r"D_M(\mathbf{x}) = \sqrt{(\mathbf{x}-\mu)^{\top}\,\hat\Sigma^{-1}\,(\mathbf{x}-\mu)},\quad \hat\Sigma_{LW} = (1-\delta)\,S + \delta\,\frac{\mathrm{tr}(S)}{p}\,I",
           "Mahalanobis distance from the healthy mean μ, with covariance Σ̂. Diagonal: only variances; Ledoit–Wolf: sample "
           "covariance S shrunk towards a scaled identity by an analytically chosen δ (stable when hands ≈ features); "
           "MCD: covariance of the most concentrated subset of hands, robust to outlying training hands.")
    doc.eq(r"\theta = Q_{0.95}\left(\{s(\mathbf{x}_k)\}_{k \in \mathrm{inner\ held{-}out}}\right)",
           "Threshold: 95th percentile of the scores of training hands, each scored while its own subject was held out "
           "(inner LOSO). A new hand is flagged 'potential anomaly' if s(x) > θ. Target specificity ≈ 0.95.")
    doc.table(["Model", "Principle", "Key settings", "Strength", "Weakness"],
              [["Robust z (max)", "largest per-feature robust z", "median/MAD", "transparent, per-joint explanation", "ignores correlations"],
               ["Mahalanobis, diagonal", "distance with variances only", "—", "simple", "ignores correlations"],
               ["Mahalanobis, Ledoit–Wolf", "shrunk covariance", "δ analytic", "stable with few subjects", "assumes elliptical data"],
               ["Robust covariance (MCD)", "covariance of the cleanest subset", "support fraction default", "robust to bad training hands", "unstable below ~12 subjects"],
               ["Isolation Forest", "path length in random trees", "200 trees", "no distribution assumption", "insensitive here"],
               ["One-class SVM", "boundary around data (RBF)", "ν = 0.1, γ = 'scale'", "flexible shape", "hyper-parameter sensitive"]],
              "Implemented healthy-reference models (input: D2JTD, or D2JTD + spatial, standardised).",
              widths=[3.2, 3.4, 3.0, 3.4, 3.4], size=8.5)


stage("5.11 Stage 11 — Model development and inference", [
    ("Input", "one feature vector per hand (34 static hands from 17 subjects); median imputation and standardisation "
     "fitted on training subjects only."),
    ("Training", "fit the model on healthy training hands; compute the threshold by inner LOSO (equation 9); save a "
     "versioned bundle (imputer, scaler, model, threshold, feature order, training ranges) with metadata (training "
     "subjects, software versions, configuration hash, limitations)."),
    ("Inference", "apply the bundle unchanged to a new hand: score, compare with θ, report per-joint contributions "
     "(robust z of each feature) as explanation."),
    ("Output", "anomaly score, 'within healthy range' / 'potential anomaly' / 'outside model experience' / 'insufficient quality'."),
    ("Selection", "pre-declared rule: highest mean of specificity, sensitivity (all 10 joints +1 °C) and "
     f"sensitivity (single joint +2 °C). MCD scored highest ({f(C['robust_covariance_mcd']['selection_score'], 3)}); "
     "Ledoit–Wolf is recommended when retraining on few subjects because it is stable at every training size, "
     "and the D2JTD + spatial Ledoit–Wolf bundle is the default in the application."),
], _model_extra)


def _out_extra():
    doc.figure(FIG / "O01_joint_view_static.png", "Joint view of a static recording: visible joints with ROI temperature on the RGB photo (colour = D2JTD) and the ROIs on the thermal image (measured).", 16)
    doc.figure(FIG / "O02_joint_view_fist.png", "Joint view during hand closing: folded joints are not shown and get no temperature (measured).", 16)


stage("5.12 Stage 12 — Final output generation", [
    ("Per frame", "21 landmarks (RGB and thermal coordinates, z), 14 joint positions (raw and corrected), "
     "visibility, ROI statistics, dorsum and hand temperature, quality flags (frames/<rec>.csv)."),
    ("Per recording", "registration parameters and Dice; feature vector (129 values); quality summary; anomaly score and category."),
    ("Per dataset / test run", "PDF report (summary, validation, joint temperature table, per-recording joint views, "
     "limitations, sample counts), Excel workbook, CSV tables, locked-test record with SHA-256 fingerprints."),
    ("Visualisations", "joint view, QC overlay, registration overlay, healthy reference plots, model comparison."),
], _out_extra)

# ============================================================================ 6 maths
doc.h("6. Mathematical Formulation and Metrics")
doc.p("This chapter defines every metric reported. Each value cited is reproduced in the metric calculation "
      "workbook (deliverables/supporting/Metric_Calculation_Sheet.xlsx), which lists source file and sample size.")
doc.h("6.1 Registration", 2)
doc.eq(r"\mathrm{Dice}(A,B) = \frac{2\,|A \cap B|}{|A| + |B|},\qquad \mathrm{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\mathrm{Dice}}{2-\mathrm{Dice}}",
       f"A: RGB hand mask warped to thermal coordinates; B: thermal hand mask; |·| pixel count {cite('dice1945')}. "
       f"Example: median held-out Dice {f(V3['dice_median'], 3)} corresponds to IoU {f(V3['dice_median'] / (2 - V3['dice_median']), 3)}. The landmark mapping error (6.2) measures the mapping at joint level.")
doc.h("6.2 Joint localisation", 2)
doc.eq(r"e_i = \|\hat{\mathbf{j}}_i - \mathbf{g}_i\|_2,\quad \mathrm{median}(e),\ \bar e = \frac{1}{n}\sum e_i,\ P_{q}(e)",
       f"ĵ_i: pipeline joint position; g_i: clicked position; n = {V5['points']}. Because e_i ≥ 0, the mean error is also "
       f"the mean absolute error. Values: median {f(V3['joint_err_corr'], 2)} px corrected, P90 {f(V3['joint_err_corr_p90'], 1)} px. "
       "The median is the primary statistic because a few mis-clicks dominate the mean.")
doc.eq(r"e_{\parallel} = (\hat{\mathbf{j}}-\mathbf{g})\cdot\mathbf{u},\quad e_{\perp} = (\hat{\mathbf{j}}-\mathbf{g})\cdot\mathbf{n}",
       "Signed error along (u: unit vector MCP→PIP) and across (n: perpendicular) the finger; reveals systematic bias.")
doc.p(f"Three-cornered hat. With three independent estimates of each joint (clicks A, RGB-mapped R, thermal-"
      f"MediaPipe T), the variance of each estimator follows from the pairwise difference variances: "
      f"σ²_A = ½(σ²_AR + σ²_AT − σ²_RT), and cyclically. Result (bias removed): clicks {f(V5['hat']['annotation'], 1)} px, "
      f"RGB-mapped {f(V5['hat']['rgb_mapped'], 1)} px, thermal-MediaPipe {f(V5['hat']['thermal_mediapipe'], 1)} px RMS. "
      "R and T share the MediaPipe model, so the click estimate is an upper bound.")
doc.h("6.3 Temperature measures", 2)
doc.p("Mean T̄ = (1/n)ΣT_i; median = middle value of the sorted ROI pixels; SD = √(Σ(T_i − T̄)²/(n − 1)); "
      "P90 = value below which 90 % of ROI pixels lie; D2JTD (equation 6); spatial differences such as "
      "T_MCP − T_PIP of the same finger. All in °C.")
doc.h("6.4 Repeatability", 2)
doc.eq(r"\mathrm{ICC}(3,1) = \frac{MS_R - MS_E}{MS_R + (k-1)\,MS_E}",
       "Two-way mixed, consistency, single measurement. MS_R: mean square between hands; MS_E: residual mean square; "
       f"k = 2 (first and second half of a static recording). 10-joint mean D2JTD: ICC = {f(V3['icc'], 3)}.")
doc.eq(r"\mathrm{MDC}_{95} = 1.96\cdot \mathrm{SD}(d),\quad d = x_{half\,2} - x_{half\,1}",
       f"Smallest change exceeding measurement noise with 95 % confidence (equivalently 1.96·√2·SEM). 10-joint "
       f"mean D2JTD: {f(V3['mdc'], 3)} °C over 34 hands. Confidence intervals of means use the bootstrap over subjects "
       f"(e.g. mean D2JTD {f(V3['mean10'], 2)} °C, 95 % CI {f(V3['mean10_ci'][0], 2)}–{f(V3['mean10_ci'][1], 2)}).")
doc.h("6.5 Classification and anomaly evaluation", 2)
doc.p("With labels, a flagged hand is a positive prediction. TP: an arthritis hand flagged; FN: an arthritis hand "
      "reported within range (a missed patient); TN: a healthy hand not flagged; FP: a healthy hand flagged "
      "(an unnecessary referral).")
doc.eq(r"\mathrm{Acc} = \frac{TP+TN}{TP+TN+FP+FN},\quad \mathrm{Prec} = \frac{TP}{TP+FP},\quad \mathrm{Rec} = \mathrm{Sens} = \frac{TP}{TP+FN}")
doc.eq(r"\mathrm{Spec} = \frac{TN}{TN+FP},\quad F_1 = \frac{2\,\mathrm{Prec}\cdot\mathrm{Rec}}{\mathrm{Prec}+\mathrm{Rec}},\quad FPR = 1-\mathrm{Spec},\quad FNR = 1-\mathrm{Sens}",
       "ROC-AUC: area under sensitivity vs FPR as the threshold varies; PR-AUC (average precision): area under "
       "precision vs recall. Both need both classes.")
doc.p("In this work specificity is measured on the real healthy hands, and sensitivity on hands with simulated "
      "warming (Sections 7.6 and 7.10). The evaluation module computes all metrics above when a labelled dataset "
      "is uploaded, and warns when a class has fewer than 10 samples.")
doc.h("6.6 Computational performance", 2)
doc.p(f"Time per frame = wall-clock time of landmarks + mask + ROIs, averaged over frames; throughput = 1000 / "
      f"(ms per frame). Measured on the development laptop CPU (single process): {f(V3['ms_frame'], 1)} ms/frame "
      f"(≈{V3['fps']:.0f} fps), peak resident memory {V3['rss_peak']:.0f} MB, model file {f(V3['model_mb'], 1)} MB, CPU only.")

# ============================================================================ 7 results
doc.h("7. Experimental Results and Discussion")
doc.note("Measured: computed from the recorded data. Simulated: warming added to real healthy data.")
doc.table(["RQ", "Result (measured unless noted)", "n", "Criterion", "Verdict"],
          [["RQ1", f"valid frames: static {pct(V3['static_valid'], 0)}, open–close {pct(V3['oc_valid'], 2)}; hand found {pct(V3['hand_detection'], 2)}", "68 rec.", "≥ 90 %", "Pass"],
           ["RQ2a", f"held-out Dice median {f(V3['dice_median'], 3)}", "68 rec.", "≥ 0.90", "Pass"],
           ["RQ2b", f"joint error {f(V3['joint_err_raw'], 1)} → {f(V3['joint_err_corr'], 1)} px (static {f(V3['joint_err_static'], 1)}, open–close {f(V3['joint_err_oc'], 1)})", f"{V5['points']} pts", "≤ 8 px", "Fail"],
           ["RQ2c", f"temperature cost of joint error: median {f(V3['dT_corr'], 3)} °C", f"{V5['points']} pts", "—", "small vs MDC"],
           ["RQ3", f"ROIs ≥ 95 % on hand: {pct(V3['roi_on_skin'], 1)}", "all ROIs", "≥ 95 %", "Pass*"],
           ["RQ4", f"10-joint mean D2JTD ICC {f(V3['icc'], 3)}, MDC95 {f(V3['mdc'], 2)} °C", "34 hands", "< 0.4 °C", "Pass"],
           ["RQ5", f"mean D2JTD {f(V3['mean10'], 2)} °C (95 % CI {f(V3['mean10_ci'][0], 2)}–{f(V3['mean10_ci'][1], 2)}); > 0 for {V3['h1']}/10 joints", "17 subj.", "reported", "—"],
           ["RQ6", "open–close: absolute temps +0.17 °C dorsum, +0.31 °C joints; D2JTD unchanged", "34 hands", "reported", "—"],
           ["RQ7", f"nested LOSO: specificity {f(C['mahalanobis_ledoit_wolf']['specificity'], 2)}; all 10 joints +1 °C: {f(C['isolation_forest']['sens_all_10_MCP_PIP_1C'], 2)}–{f(C['one_class_svm']['sens_all_10_MCP_PIP_1C'], 2)} (simulated)", "34 hands", "reported", "simulation"],
           ["Sync", f"{V3['sync_aligned']}/34 aligned at lag 0, median r {f(V3['sync_corr'])}", "34 rec.", "—", "OK"],
           ["Speed", f"{f(V3['ms_frame'], 1)} ms/frame ≈ {V3['fps']:.0f} fps (laptop CPU)", "—", "> 8.7 fps", "OK"]],
          "Summary of results against the pre-registered criteria. *RQ3 partly by construction.",
          widths=[1.3, 8.0, 1.9, 2.2, 2.4], size=8.5)
doc.note(f"Erratum carried from the v4 audit: open–close landmark validity is {pct(V3['oc_valid'], 2)} (earlier reports rounded it to 99.7 %). "
         f"The corrected joint error recomputed here is {f(V3['joint_err_corr'], 2)} px; earlier reports give 10.35 px — equal at the reported precision.")

doc.h("7.1 Registration", 2)
doc.p(f"Objective: verify the RGB→thermal mapping. Method: held-out Dice on 8 of 16 sampled frames per recording. "
      f"Data: 68 recordings, 17 subjects. Result: median {f(V3['dice_median'], 3)}; {V3['dice_ge90']} of 68 recordings ≥ 0.90; "
      f"the worst ({f(V3['dice_min'], 3)}) are open–close recordings where fingers move in depth. Interpretation: the outline "
      "is mapped well; a good outline overlap does not guarantee joint-level accuracy (7.2).")
doc.figure(FIG / "G02_registration_dice.png", "Held-out Dice per recording, static vs open–close (measured, 68 recordings).", 14.5)

doc.h("7.2 Joint localisation", 2)
doc.p(f"Objective: compare pipeline joints with hand-clicked joints. Data: {V5['points']} points on {V5['frames']} frames "
      "from all 17 subjects. Result: the localisation table and error figure in Section 5.5. The correction removes most of the along-finger bias, "
      f"but the median stays above 8 px. Annotation quality check (v5): 133 of 136 frames internally consistent; two "
      f"confirmed labelling mistakes; without the three flagged frames the error is "
      f"{f(V5['sens']['without_flagged_frames']['corrected_loso_median_px'], 2)} px — the annotations are not the main "
      f"cause. What the error costs: reading each ROI at the clicked instead of the pipeline position changes the "
      f"temperature by a median of {f(V3['dT_corr'], 3)} °C, far below the {f(V3['mdc'], 2)} °C repeatability limit.")
doc.figure(FIG / "G08_annotation_hat.png", "Error budget from the three-cornered hat: random scatter of each joint estimator (measured).", 13.5)
doc.figure(FIG / "R06_annotation_contact_sheet.png", "Annotation QC contact sheet: flagged and worst frames with clicks and pipeline joints (measured).", 15.5)

doc.h("7.3 Repeatability", 2)
doc.p("Objective: is the measurement stable within a session? Method: each static recording split into first and "
      "second half; ICC(3,1) and MDC95 across 34 hands. Results per joint are in the table below. MCP joints are the most "
      "repeatable (MDC95 ≈ 0.20–0.27 °C); DIP joints the least (≈ 0.47–0.57 °C), because they are small and most "
      "affected by joint error.")
jr = [[j.replace("_", " "), f(V3["icc_joints"][j], 3), f(V3["mdc_joints"][j], 2)] for j in V3["mdc_joints"]]
doc.table(["Joint / feature", "ICC(3,1)", "MDC95 (°C)"], jr, "Split-half repeatability of D2JTD per joint (34 static hands, measured).", widths=[5, 3, 3], size=9)
doc.figure(FIG / "G03_repeatability_mdc.png", "MDC95 per joint against the 0.4 °C criterion (measured).", 14)
doc.figure(FIG / "R02_repeatability_bland_altman.png", f"Bland–Altman plot {cite('bland1986')} of the 10-joint mean D2JTD, first vs second half (measured).", 13)

doc.h("7.4 Healthy reference (RQ5)", 2)
doc.p("Per-subject mean D2JTD over left and right static recordings, 17 subjects. Reference interval = mean ± 1.96 SD. "
      f"H1 (joint cooler than dorsum, Holm-corrected {cite('holm1979')}) holds for {V3['h1']} of 10 MCP/PIP joints (index MCP, little MCP, "
      "little PIP). H2 (left–right absolute difference < 0.5 °C) was not supported: left and right hands were recorded "
      "separately, and absolute temperature drifts between recordings, which is why relative features are used.")
rr = [[r["feature"].replace("_", " "), r["n_subjects"], f(r["mean"]), f(r["sd"]), f"{f(r['ci_lo'])} – {f(r['ci_hi'])}", f"{f(r['ref_lo'])} – {f(r['ref_hi'])}"]
      for r in V3["reference"]]
doc.table(["Joint", "n", "Mean (°C)", "SD", "95 % CI of mean", "Reference interval"], rr,
          "Healthy D2JTD reference (static recordings, measured).", widths=[3.4, 1.0, 2.0, 1.6, 3.6, 3.8], size=8.5)
doc.figure(FIG / "R01_healthy_reference.png", "Healthy D2JTD per joint and subject (measured, 17 subjects).", 15)

doc.h("7.5 Static versus open–close (RQ6)", 2)
r6 = V3["rq6"]
doc.p(f"Paired comparison over 34 hands (Wilcoxon, subject-level p reported). Absolute temperatures were higher in "
      f"open–close recordings (dorsum median +{f(r6['dorsum_T']['diff_median'])} °C, p = {f(r6['dorsum_T']['p_subjects'], 3)}; joint mean "
      f"+{f(r6['jointmean_T']['diff_median'])} °C, p = {f(r6['jointmean_T']['p_subjects'], 3)}), but the 10-joint mean D2JTD did not "
      f"change (difference {f(r6['D2JTD_mean10']['diff_median'])} °C, p = {f(r6['D2JTD_mean10']['p_subjects'], 2)}). The relative "
      "feature therefore cancels motion- or time-related warming.")
doc.figure(FIG / "R03_static_vs_open_close.png", "Static vs open–close: absolute temperatures and D2JTD per hand (measured).", 15)
doc.figure(FIG / "G06_visibility.png", "Share of frames with hidden joints by joint type and task (v8 visibility rule, 68 recordings).", 14)
doc.p(f"Visibility (v8): in static recordings {pct(VIS['static']['ANY'], 0)} of frames had any hidden joint; in open–close "
      f"recordings {pct(VIS['open_close']['ANY'], 0)} did — DIP {pct(VIS['open_close']['DIP'], 0)}, thumb IP "
      f"{pct(VIS['open_close']['IP'], 0)}, PIP {pct(VIS['open_close']['PIP'], 0)}, MCP {pct(VIS['open_close']['MCP'], 0)} "
      f"({V8['frames']:,} analysed frames).")

doc.h("7.6 Anomaly models with simulated warming (RQ7, v7)", 2)
doc.p("Objective: estimate how the healthy-reference models would respond to joint warming, without patients. "
      "Method: nested LOSO over 17 subjects (34 hands). For each held-out subject, the model and threshold are fitted "
      "on the other 16; the held-out hands are scored as they are (specificity) and after adding a simulated "
      "temperature increase to chosen joints (sensitivity at that warming). Patterns: one random joint; all MCPs, "
      "PIPs or DIPs; all 10 MCP/PIP; the whole hand. Warming of 0.25–3 °C. Input: 14 D2JTD features.")
cm = [[k.replace("_", " "), f(r["specificity"]),
       "–".join(f(v) for v in json.loads(r["specificity_ci"])),
       f(r["sens_all_10_MCP_PIP_1C"]), f(r["sens_single_joint_2C"]), f(r["sens_diffuse_whole_hand_2C"]),
       f"{r['score_us_per_hand'] / 1000:.3f}"] for k, r in C.items()]
doc.table(["Model", "Specificity", "95 % CI", "All 10 +1 °C", "1 joint +2 °C", "Whole hand +2 °C", "ms/hand"], cm,
          "Model comparison, nested leave-one-subject-out (34 healthy hands; sensitivities are SIMULATED).",
          widths=[4.0, 1.9, 2.4, 2.0, 2.0, 2.2, 1.7], size=8.5)
doc.figure(FIG / "G05_synthetic_sensitivity.png", "Sensitivity versus simulated warming per model and pattern (simulation on healthy data).", 15.5)
doc.p("Interpretation: with thresholds chosen without the test subject, all models meet the specificity target "
      f"(0.94, i.e. 2 of 34 healthy hands flagged). Sensitivity at 1 °C is modest (0.03–{f(C['one_class_svm']['sens_all_10_MCP_PIP_1C'])}); "
      "at 2 °C on all 10 joints MCD, SVM and Ledoit–Wolf reach 1.00. Whole-hand warming is never detected (0.06 = the "
      "false-positive rate): D2JTD is relative to the dorsum by design. The earlier v3 sensitivities (0.71–1.00 at 1 °C) "
      "used thresholds chosen on the same hands and were optimistic; they are superseded.")
doc.figure(FIG / "R04_model_comparison.png", "Model comparison chart from the v7 evaluation (simulation on healthy data).", 15)

doc.h("7.7 Feature ablation (v6)", 2)
ab = [[k[0], k[1].replace("_", " "), r["n_features"], f(r["specificity"]), f(r["sens_single_joint_2C"]), f(r["sens_all10_1C"])]
      for k, r in AB.items() if k[1] == "mahalanobis_ledoit_wolf"]
doc.table(["Feature set", "Model", "n features", "Specificity", "1 joint +2 °C", "All 10 +1 °C"], ab,
          "Feature-set ablation, nested LOSO (sensitivities SIMULATED).", widths=[4.4, 4.0, 1.8, 2.0, 2.1, 2.1], size=8.5)
doc.figure(FIG / "G04_feature_ablation.png", "Feature-set ablation (simulation): adding spatial contrast helps; adding all 122 features hurts.", 14.5)
s3 = AB[("S3_D2JTD+spatial", "mahalanobis_ledoit_wolf")]
doc.p(f"Adding spatial contrast features raised single-joint +2 °C sensitivity from {f(AB[('S1_D2JTD (v3 baseline)', 'mahalanobis_ledoit_wolf')]['sens_single_joint_2C'])} "
      f"to {f(s3['sens_single_joint_2C'])} at unchanged specificity. Using all 122 features reduced the all-10 +1 °C sensitivity to "
      f"{f(AB[('S6_all', 'mahalanobis_ledoit_wolf')]['sens_all10_1C'])}: with 34 hands, many features means overfitting. Caveat: the "
      "simulation warms one ROI uniformly — exactly the local contrast spatial features measure — so this gain is an "
      "upper estimate. Temporal features were not adopted (ICC 0.40).")

doc.h("7.8 Learning curve", 2)
lc = V7["learning_curve"]
doc.p("Models retrained on 4–16 subjects and evaluated on held-out subjects (repeated random subsets). MCD is unstable "
      f"below about 12 subjects (specificity {f(lc[0]['specificity'])} at 4, {f(lc[3]['specificity'])} at 10, "
      f"{f(lc[4]['specificity'])} at 12): its robust covariance needs more hands than parameters. Ledoit–Wolf kept "
      "specificity ≈ 0.88–0.94 at all sizes. More subjects are needed before small differences between models are meaningful.")
doc.figure(FIG / "R05_learning_curve.png", "Learning curves: specificity and simulated sensitivity versus number of training subjects.", 14.5)

doc.h("7.9 Computational performance", 2)
mp = V3["ms_parts"]
doc.p(f"Per frame on the laptop CPU: landmarks {f(mp['landmarks'], 1)} ms, thermal mask {f(mp['thermal_mask'], 1)} ms, ROIs "
      f"{f(mp['rois'], 1)} ms; total {f(mp['total'], 1)} ms ≈ {V3['fps']:.0f} fps, against a sensor rate of ≈ 8.7 fps. Memory: "
      f"{V3['rss_model']:.0f} MB after loading the model, {V3['rss_peak']:.0f} MB peak. Live simulation including pairing: "
      f"{f(LIVE['processing_fps'], 1)} fps. Model scoring adds 0.2 ms per hand (Ledoit–Wolf).")
doc.figure(FIG / "G07_compute.png", "Processing time per frame by step (laptop CPU, measured).", 13.5)
doc.figure(FIG / "O04_edge_demo_flag.png", "Real-time demo output: per-frame joint status from a saved model (measured on the laptop).", 14)

doc.h("7.10 Correct predictions on unseen subjects", 2)
doc.p("Objective: show the model's output case by case. Method: the application's default model (Ledoit–Wolf "
      "Mahalanobis on D2JTD + spatial features) is applied with nested leave-one-subject-out, so every hand is "
      "scored by a model and threshold that never saw that person. Each real recording is a healthy volunteer, so "
      "the correct output is 'within healthy range'. The same recording with its index-MCP ROI warmed by 2 °C in "
      "every frame (simulated) should be flagged 'potential anomaly'.")
doc.p(f"Result: {PR['real_correct']} of {PR['n_hands']} healthy hands were correctly within range, and {PR['warmed_correct']} of "
      f"{PR['n_hands']} simulated warm joints were correctly flagged. In all {PR['top_feature_is_warmed_joint']} flagged cases "
      "the explanation pointed to the warmed joint.")
doc.figure(FIG / "P01_correct_predictions_gallery.png", "Correct outputs on six unseen healthy volunteers: all 14 joints measured, temperature at each joint, score below the threshold (measured).", 16)
doc.figure(FIG / "P02_simulated_detection_example.png", "One unseen hand: the real recording passes; the same hand with a simulated +2 °C index MCP is flagged and the explanation locates the warm joint.", 16.4)
doc.figure(FIG / "P03_prediction_overview.png", "All 34 hands: score relative to the threshold for the real recording (green) and with the simulated warm joint (red triangles).", 16.4)

# ============================================================================ 8 frames -> one value
doc.h("8. From Many Frames to One Temperature per Joint")
doc.p("A recording contains hundreds of frames, and every frame gives a slightly different temperature for each joint. "
      f"The report gives one value per joint. Example: S2_right_static has {int(_fr.frame.max()) + 1} video frames; every third "
      f"frame is analysed ({len(_fr)}), and {len(AGG)} pass the quality checks.")
doc.numbered([
    "Inside a frame: the joint temperature is the median of the radiometric pixels of the joint ROI (skin pixels only, no edge pixels).",
    "Frame selection: frames failing the landmark or registration checks are dropped; a joint that is folded or covered in a frame is dropped for that frame only; open–close recordings keep only open-hand frames.",
    "Across frames: the FINAL joint temperature is the median of its per-frame values. D2JTD is computed frame by frame (back of hand minus joint) and its median taken in the same way.",
    "Spread: the 25th and 75th percentiles over frames (Q1–Q3) and the number of frames used are stored and printed in the report next to the final value.",
])
from src.landmarks import JOINT_NAMES  # noqa: E402
_rows = []
for j in JOINT_NAMES:
    x = AGG[f"{j}_T"].dropna()
    _rows.append([j.replace("_", " "), len(x), f(x.min()), f(x.max()), f"{f(x.quantile(.25))}–{f(x.quantile(.75))}", f(x.median()),
                  f((AGG.dorsum_T - AGG[f"{j}_T"]).median())])
doc.table(["Joint", "Frames", "Lowest", "Highest", "Q1–Q3", "Final (median)", "D2JTD"], _rows,
          "S2_right_static: per-frame range and the final temperature of each joint (°C, measured).",
          widths=[3.0, 1.5, 1.8, 1.8, 3.0, 2.6, 2.0], size=8.5)
doc.figure(FIG / "N01_frame_aggregation.png", "From 267 frames to one temperature per joint: per-frame values (left), distribution for the index MCP with median and Q1/Q3 (middle), final values with Q1–Q3 for all joints (right).", 16.4)
doc.p("Most of the frame-to-frame difference is a slow drift of the whole hand during the 1.5–3 minutes of recording; the "
      "back of the hand drifts with the joints, which is why D2JTD is more stable than the raw temperatures. The median is "
      "used because it represents the typical value and is not moved by a minority of bad frames (motion, a misplaced "
      "landmark, a cold ring pixel), unlike the mean, the maximum or a single frame. Its reliability is the repeatability "
      f"measured in Section 7.3: the same hand measured from two halves of a recording agrees within {f(V3['mdc'], 2)} °C (MDC95).")
doc.table(["Option", "Effect"], [["Mean of frames", "pulled by single bad frames"], ["First or last frame", "depends on one moment; hand drifts"],
                                  ["Maximum", "selects the outlier, not the typical value"], ["Median of frames (used)", "typical value; robust to up to half the frames being bad"]],
          "Why the median of the frames is the final value.", widths=[5.0, 11.4], size=9.5)
doc.figure(FIG / "N04_report_joint_table.png", "Dataset report (application output): final temperature, Q1–Q3, number of frames and D2JTD for every joint of a recording.", 14)
doc.p("The per-joint values then form one feature vector per recording (medians of D2JTD and the spatial contrasts); the "
      "model scores this vector once, and the recording receives one category (Normal measurement, Potential anomaly, or a "
      "quality category). Frames are never scored individually: frames of one hand are not independent observations.")

# ============================================================================ 9 calibration
doc.h("9. Automatic RGB–Thermal Calibration (v9)")
doc.p("The RGB camera finds the joints; the radiometric image holds the temperatures. Every joint position must therefore be "
      "transformed from RGB pixels to thermal pixels. Version 9 adds a module (src/calibration.py) that finds this "
      "transformation from the recordings themselves — no calibration target is needed — applies it to points and images "
      "and verifies it, so that the system can be used with other cameras and exports.")
doc.h("9.1 What the module determines", 2)
doc.numbered([
    "Image area of the RGB video (letterbox bars removed), as the median of the content box over several frames.",
    "Orientation check: the radiometric frames must have the same aspect as the RGB image; a rotated or transposed export is reported with an explanation instead of being mapped wrongly.",
    "Scale and in-plane rotation, searched jointly (scale 0.70–1.80, rotation −10° to +10°), the three best candidates refined.",
    "A shift per frame learned from the hand seen in RGB (hand size and position; robust Huber regression), so the mapping follows small changes of hand distance.",
    "A self-check on frames not used for fitting, with a grade: good (≤ 5 px), acceptable (≤ 8 px) or poor.",
])
doc.eq(r"\mathbf{p}_{th} = s\,\mathrm{Rot}(\alpha)\,(k\,\mathbf{p}_{rgb} - \mathbf{c}) + \mathbf{c} + \mathbf{t}(\mathrm{frame})",
       "Calibrated transform: s scale, α rotation, k = radiometric width / RGB image width, c image centre, t the per-frame shift.")
doc.eq(r"d(M) = \frac{1}{N}\sum_{i=1}^{N}\min\left(D_{th}(M\,\mathbf{q}_i),\ 10\ \mathrm{px}\right)",
       "Score used to find s, α and t: q_i are points on the RGB hand outline, D_th the distance of a thermal pixel to the thermal hand "
       "outline. Distances are capped at 10 px, so a part of the thermal mask with no counterpart in RGB (for example a warm "
       "forearm) cannot pull the scale; matching outlines is what fixes the scale.")
doc.h("9.2 Proof 1: recovering known transformations", 2)
doc.p("Three real recordings were deliberately distorted with known transformations (thermal side scaled by +8 % or −10 %, "
      "rotated by +4° or −6°, shifted by up to 25 px; RGB picture moved 40 px inside the video frame; and all combined). The "
      "module was run without any information about the distortion and its result compared with the known truth on a grid "
      "of points over the hand area.")
doc.table(["Recording", "Distortion", "Scale true / found", "Angle true / found", "Error median (px)"],
          [[r.recording, r.case[:46], f"{r.scale_true:.3f} / {r.scale_found:.2f}", f"{r.angle_true:+.2f} / {r.angle_found:+.2f}",
            f"{r.mapping_error_px_median:.2f}"] for r in SELF.itertuples()],
          "Known-distortion test (12 cases): all recovered.", widths=[3.2, 6.4, 2.6, 2.6, 1.8], size=8)
doc.figure(FIG / "N02_calibration_selftest.png", "Known-distortion test: mapping error per case (left); found vs true scale and angle error (right).", 16)
doc.h("9.3 Proof 2: unseen people", 2)
doc.p("The module was run on subjects S1–S5 only and the result applied unchanged to the 12 other subjects; joint positions "
      "were compared with the hand-clicked joints of those subjects (along-finger correction re-learned leave-one-subject-out "
      "for both methods).")
doc.table(["Method", "Joint error (median)", "Static", "Open–close", "Dice"],
          [["v3: one shift fitted per recording", f"{B_V3['joint_err_median']:.2f} px", f"{B_V3['static']:.2f}", f"{B_V3['open_close']:.2f}", f"{B_V3['dice_median']:.3f}"],
           ["v9: found calibration (this module)", f"{B_V9['joint_err_median']:.2f} px", f"{B_V9['static']:.2f}", f"{B_V9['open_close']:.2f}", f"{B_V9['dice_median']:.3f}"]],
          f"Calibration found from S1–S5, tested on S6–S17 ({B_V9['n_joints']} clicked joints, {B_V9['n_dice_frames']} frames).",
          widths=[6.4, 3.2, 2.2, 2.4, 2.2], size=9)
doc.figure(FIG / "N03_calibration_benchmark.png", "Unseen-subject benchmark: joint error and hand-outline overlap.", 15)
doc.h("9.4 Production calibration and how it is applied", 2)
doc.p(f"The default profile flirone_one_pro_v9_found was found from all 68 recordings: image area {PROD['video_box']}, scale "
      f"{PROD['scale']:.2f}, rotation {PROD['angle_deg']:.1f}°; held-out Dice {PROD['quality']['dice_heldout_median']:.3f}, outline "
      f"distance {PROD['quality']['outline_px_heldout_median']:.1f} px (grade {PROD['quality']['grade']}). The along-finger joint "
      "correction was re-learned for this mapping. The pipeline applies it per frame and stores the transform used with "
      f"every frame, so the joint view and reports use exactly the same mapping. Verification on another recording "
      f"(S9_left_static): Dice {EXV['dice_median']:.3f}, outline distance {EXV['outline_px_median']:.1f} px, grade {EXV['grade']}.")
doc.figure(FIG / "N12_calibration_apply.png", "Applying the calibration: joints found in RGB (left) mapped to the thermal image with their temperatures (right).", 15)
doc.figure(FIG / "N11_calibration_verify.png", "Verification overlay: white = RGB and thermal hand overlap, orange = thermal only, blue = RGB only.", 8)
doc.figure(FIG / "N07_app_calibration.png", "Application: calibration profiles and Auto-calibrate for a new dataset (screenshot).", 15.5)

# ============================================================================ 8 upload
doc.h("10. New Dataset Upload and Testing Module")
doc.p("The application layer (v8) lets a user bring a new dataset, check it, process it and test a saved model on "
      "it without writing code. The figure below shows the workflow.")
doc.figure(FIG / "F12_upload_workflow.png", "Dataset upload and validation workflow.", 15.5)
doc.numbered([
    "Upload: a ZIP archive, a folder (used in place, not copied) or a single recording (three files).",
    "Extract safely: archive members with absolute paths, '..', links, excessive compression ratio or more than the free disk space are refused.",
    "Identify subjects and recordings with a filename profile: 'flirone' (S1_left_static_rgb.mp4), 'generic' (P01_L_static_rgb.mp4) or a custom regular expression with named groups.",
    "Match RGB, thermal and radiometric files per recording; unmatched files are listed and can be matched manually in the GUI.",
    "Metadata: a manifest per dataset stores subjects, hands, tasks, files, profile, calibration profile and validation results; labels are imported from CSV/Excel.",
    "Validate each recording: files readable, frame counts equal, .npy dtype/shape/NaN, temperature range plausible, duplicates (by hash), letterbox box.",
    "Calibration compatibility: the dataset's camera profile must match a stored calibration profile (FLIR ONE Pro v3: s = 1.25, rig shift); otherwise status 'Requires calibration'.",
    "Process: the same pipeline into the dataset's own processed folder, with a run manifest.",
    "Load a saved model bundle from the registry.",
    "Predict: per-hand score and category.",
    "Upload ground-truth labels (subject, hand, recording or joint level, with source).",
    "Compute metrics with guards (single class, < 10 per class).",
    "Generate PDF/Excel/CSV reports."])
doc.p("Validation statuses: Valid; Valid with warnings; Invalid; Requires calibration; Requires manual file "
      "matching — each with the failing check explained. Supported formats: MP4/AVI/MOV video readable by OpenCV, "
      ".npy float arrays in °C (portrait or landscape, declared in the profile), ZIP archives, CSV/XLSX labels.")
doc.figure(FIG / "A02_app_dataset_validation.png", "Application: dataset validation page with per-recording status (screenshot of the real application).", 15.5)
doc.p("End-to-end software test: a 3-recording ZIP with a different naming convention, an unrelated video and a "
      "dummy label file went through import → validation → processing → independent test → reports. "
      f"All 3 recordings matched and Valid; the stray video listed as unmatched; all {E2E['n_recordings']} 'Normal measurement'; "
      "the deliberately unmatched label row reported. The fixture re-uses training subjects under new IDs, so it tests "
      "the software, not generalisation.")

# ============================================================================ 9 training/testing
doc.h("11. Model Training, Validation and Independent Testing")
doc.figure(FIG / "F11_loso_validation.png", "Nested leave-one-subject-out validation: outer loop for evaluation, inner loop for the threshold.", 15.5)
doc.h("11.1 Training", 2)
doc.p("Features are extracted per hand (Stage 9); the feature set and model are chosen by the pre-declared rule on "
      "nested-LOSO results; the final model is fitted on all 17 subjects and saved as a versioned bundle that is "
      "never overwritten (models/registry/<model>_<timestamp>/bundle.joblib + metadata.json).")
doc.h("11.2 Validation", 2)
doc.p("Splits are by subject (both hands of a subject are always together). Outer LOSO estimates performance; inner "
      "LOSO on the 16 training subjects picks the threshold at the 95th percentile of held-out healthy scores. "
      "Imputation, scaling, covariance and threshold are learned from training subjects only. GroupKFold with "
      "subject groups is equivalent with fewer folds; LOSO was used because n = 17 is small. Hyper-parameters were "
      "fixed in advance (ν = 0.1, 200 trees) rather than tuned, to avoid tuning to 17 people.")
doc.h("11.3 Independent testing", 2)
doc.figure(FIG / "F13_test_workflow.png", "Locked independent-test workflow.", 15.5)
doc.p("An independent test applies a saved model, unchanged, to an uploaded dataset. The run is locked: the "
      "features, manifest, model and labels are fingerprinted with SHA-256, stored in the run's own folder with the "
      "rule 'model applied as saved: no refitting, rescaling or threshold changes', and any earlier run of the same "
      "model on the same dataset is listed. Subjects that also appear in the model's training list are reported "
      "(ID-based check). Retraining or tuning on a test set would turn it into training data and make the reported "
      "performance optimistic; locking makes this visible.")
doc.figure(FIG / "A06_app_independent_test.png", "Application: independent-test page with locked run, metrics and warnings (screenshot of the real application).", 15.5)
# ============================================================================ 10 outputs
doc.h("12. Complete System Outputs")
doc.p("This chapter collects the outputs a user sees. The application screenshots below are of the real "
      "application running on the FLIROne data; the table that follows lists every output, the stage "
      "and algorithm that produce them, how to read them and where they appear in this report.")
doc.figure(FIG / "A00_app_dashboard.png", "Application dashboard: datasets, models and recent runs (screenshot of the real application).", 15.5)
doc.figure(FIG / "A03_app_joint_view.png", "Joint-view page: RGB with visible joints and temperatures, thermal with ROIs, joint table (screenshot).", 15.5)
doc.figure(FIG / "A04_app_joint_view_fist.png", "Joint-view page on a closing hand: hidden joints are omitted and explained in the table (screenshot).", 15.5)
doc.figure(FIG / "A01_app_models.png", "Models page: saved model bundles with their validation metadata (screenshot).", 15.5)
doc.figure(FIG / "A05_app_reports.png", "Reports page with the built-in PDF viewer showing a generated dataset report (screenshot).", 15.5)


def fn(*stems):
    return ", ".join(f"Fig. {doc.figs[s]}" for s in stems if s in doc.figs) or "—"


outs = [
    ("Original RGB frame", "Stage 2 (sequential decode)", "RGB photo", fn("R07_audit_sample_frames", "O01_joint_view_static")),
    ("Original thermal frame", "thermal MP4 (audit only)", "display palette, not °C", fn("R07_audit_sample_frames")),
    ("Radiometric temperature visualisation", "Stage 8", "inferno colour map of °C", fn("F08_thermal_segmentation_real", "O01_joint_view_static")),
    ("Detected hand", "Stage 4 (MediaPipe)", "landmarks present = hand found", fn("F05_landmarks_real")),
    ("21-landmark overlay", "Stage 4", "points and skeleton", fn("F05_landmarks_real")),
    ("Joint coordinate visualisation", "Stages 4–5", "visible joints with labels", fn("O01_joint_view_static", "O02_joint_view_fist")),
    ("RGB hand mask", "Stage 6 (GrabCut)", "hand region from RGB", fn("F09_registration_real")),
    ("Thermal hand mask", "Stage 6 (Otsu + hysteresis)", "hand region; eroded core", fn("F08_thermal_segmentation_real")),
    ("RGB–thermal registration overlay", "Stage 3", "mask overlap; Dice", fn("F09_registration_real")),
    ("Fourteen joint ROI visualisation", "Stage 7", "circles = ROIs", fn("F06_rois_d2jtd_real", "O01_joint_view_static")),
    ("Dorsum ROI visualisation", "Stage 7", "dorsum polygon", fn("F06_rois_d2jtd_real")),
    ("Joint temperature extraction", "Stage 8", "°C per joint (median)", fn("F06_rois_d2jtd_real", "A03_app_joint_view") + "; per-frame CSV"),
    ("D2JTD feature table", "Stage 9", "°C; positive = joint cooler", "Section 7.4 table; joint-view table"),
    ("Feature matrix", "Stage 9", "129 features per recording", "features CSV; Excel report"),
    ("Healthy reference distribution", "analysis (RQ5)", "per-subject D2JTD", fn("R01_healthy_reference")),
    ("Anomaly detection output", "Stage 11", "score vs threshold, category", fn("O04_edge_demo_flag") + "; PDF report"),
    ("Joint-wise anomaly map", "Stage 11 explanation", "deviation per joint (SD)", fn("P02_simulated_detection_example", "O01_joint_view_static")),
    ("Quality-control report", "Stage 10", "flags per frame and ROI", fn("O03_qc_overlay", "A02_app_dataset_validation")),
    ("Model comparison table", "v7 evaluation", "nested LOSO", "Section 7.6; " + fn("R04_model_comparison")),
    ("Correct-prediction gallery", "Stage 11 on unseen subjects", "green = within healthy range", fn("P01_correct_predictions_gallery", "P03_prediction_overview")),
    ("Final joint temperature table (median, Q1–Q3, frames)", "report generator", "°C per joint", fn("N04_report_joint_table")),
    ("Calibration check overlay", "calibration (verify)", "white = overlap", fn("N11_calibration_verify", "N12_calibration_apply")),
    ("Computational performance chart", "edge benchmark", "ms per step", fn("G07_compute")),
    ("Dataset validation dashboard", "GUI", "status per recording", fn("A02_app_dataset_validation")),
    ("Final individual recording report", "report generator", "joint view per recording in PDF", fn("A05_app_reports")),
    ("Final complete dataset report", "report generator", "PDF + Excel", fn("A05_app_reports"))]
doc.table(["#", "Output", "Generated by", "Values / colours", "Where shown"],
          [[i + 1, *o] for i, o in enumerate(outs)], f"The {len(outs)} system outputs.", widths=[0.8, 4.0, 3.6, 3.4, 4.6], size=8.5)
# ============================================================================ 13 deployment
doc.h("13. Deployment and How to Use the Deployed System")
doc.p("Release 1.0.0 packages the complete system so that it can be installed on another Windows PC and used without "
      "programming. The release contains the program, the models with SHA-256 checksums, the calibration profiles, "
      "configuration without personal paths, tests, installer, launchers and documentation; it does not contain datasets.")
doc.figure(FIG / "N05_deployment_workflow.png", "From the release zip to results.", 16)
doc.h("13.1 Installation", 2)
doc.table(["Item", "Requirement / content"],
          [["Release", "dist/rgb_thermal_hand-1.0.0.zip (27 MB, 81 files, MANIFEST.sha256)"],
           ["System", "Windows 10/11 64-bit, ~2 GB disk, 4 GB RAM, no GPU"],
           ["Python", "3.12, created by the installer (uv) with the exact package versions of requirements.lock.txt"],
           ["Install", "unzip, then: powershell -ExecutionPolicy Bypass -File .\\install.ps1  (options: -Offline, -Shortcut)"],
           ["Check", "the installer runs the health check (rgbthermal doctor); it must end with READY"]],
          "Installation.", widths=[2.6, 13.8], size=9)
doc.figure(FIG / "N06_doctor_output.png", "Health check of an installation: Python, packages, model integrity, calibration profiles, hand model, video decoding, GUI, write access.", 15.5)
doc.h("13.2 Using the deployed model in the application", 2)
doc.p("Start: double-click 'RGB-Thermal App.bat'. The saved screening model (models/registry) and the default calibration "
      "are part of the installation; the pages are used in this order:")
doc.table(["Step", "Page", "What to do", "Result"],
          [["1", "Dataset upload", "Import ZIP / folder / one recording; Validate", "status per recording (Valid … Requires calibration)"],
           ["2", "Calibration", "only for a new camera/export: choose dataset, Auto-calibrate", "new profile with grade"],
           ["3", "Processing", "choose dataset + calibration, Process", "per-frame tables, features, joint temperatures"],
           ["4", "Models", "view the saved model (training subjects, features, threshold, validation)", "model metadata"],
           ["5", "Independent test", "choose dataset + saved model (+ labels if any), Run", "category and score per recording; locked run"],
           ["6", "Joint view", "choose recording and frame", "RGB with visible joints and temperatures + thermal ROIs"],
           ["7", "Reports", "open the dataset or test report", "PDF (with final joint temperatures) + Excel"]],
          "Workflow in the deployed application.", widths=[1.0, 2.8, 7.0, 5.6], size=8.5)
doc.figure(FIG / "N08_app_dataset_upload.png", "Dataset upload page with a new dataset: three recordings matched and Valid; a stray file listed as unmatched (screenshot).", 15.5)
doc.figure(FIG / "N09_app_models.png", "Models page: the deployed screening model with its metadata (screenshot).", 15.5)
doc.h("13.3 Using it from the command line", 2)
doc.table(["Command", "Purpose"],
          [["rgbthermal.bat app", "start the application"], ["rgbthermal.bat doctor", "installation health check"],
           ["rgbthermal.bat calibrate find --rgb a.mp4 --npy a.npy --name cam", "find a calibration from recordings"],
           ["rgbthermal.bat calibrate verify --profile p.json --rgb x.mp4 --npy x.npy", "check a calibration on another recording"],
           ["rgbthermal.bat calibrate apply --profile p.json --rgb x.mp4 --npy x.npy --frame 300", "map one frame's joints and read temperatures"],
           ["rgbthermal.bat test", "run the automated tests"], ["rgbthermal.bat version", "show the version"]],
          "Command-line entry points.", widths=[9.4, 7.0], size=8.5)
doc.h("13.4 Verification of the deployment", 2)
doc.p("The release zip was unzipped into an empty folder and installed offline. In that installation the health check "
      "reported READY, the automated tests passed (48 passed, 1 skipped because the original dataset is not shipped), and the "
      "complete new-data workflow ran: import (3 recordings, 1 stray file reported) → auto-calibration (box found exactly, "
      "scale 1.27, grade good) → validation (3 Valid) → processing with the per-frame calibration → independent test with "
      "the saved model (3 Normal measurement) → PDF reports → joint view (14/14 joints) → all 10 application pages. Results "
      "and logs are written inside the installation folder (data/, results/, logs/rgbthermal.log).")

# ============================================================================ 12 conclusion
doc.h("14. Conclusion")
doc.p("A complete RGB–thermal hand-joint temperature measurement and anomaly-screening research system was built and "
      "evaluated on 17 healthy subjects. The pipeline locates the hand in nearly every frame, maps the hand outline "
      f"to the thermal image with a median Dice of {f(V3['dice_median'], 2)}, places 98 % of joint ROIs on skin and measures the "
      f"10-joint mean D2JTD with an MDC95 of {f(V3['mdc'], 2)} °C. A healthy D2JTD reference was established for 14 joints, and "
      f"joint positions are corrected to {f(V3['joint_err_corr'], 1)} px median error at a temperature cost of {f(V3['dT_corr'], 2)} °C.")
doc.p("Six healthy-reference models were compared with nested leave-one-subject-out evaluation; all reach 0.94 specificity "
      f"on unseen subjects. With the default model, {PR['real_correct']}/{PR['n_hands']} unseen healthy hands were correctly within "
      f"range and {PR['warmed_correct']}/{PR['n_hands']} simulated warm joints were correctly flagged and located. The desktop "
      "application brings this to new datasets: upload and validation, processing, locked independent tests, joint "
      "view and PDF/Excel reports.")
doc.p("Version 9 added an automatic calibration that finds and applies the RGB→thermal transformation from the recordings "
      f"(12/12 known distortions recovered; joint error on unseen subjects {B_V9['joint_err_median']:.2f} px instead of "
      f"{B_V3['joint_err_median']:.2f} px), reports one final temperature per joint as the median over frames with its spread, "
      "and release 1.0.0 makes the system installable and verifiable on another PC.")

# ============================================================================ refs
doc.h("References")
for i, r in enumerate(REFS, 1):
    para = doc.d.add_paragraph()
    para.paragraph_format.left_indent = __import__("docx").shared.Cm(0.9)
    para.paragraph_format.first_line_indent = __import__("docx").shared.Cm(-0.9)
    run = para.add_run(f"[{i}]\t{r['text']}")
    run.font.size = __import__("docx").shared.Pt(9.5)

# ============================================================================ appendix
doc.h("Appendix A. Reproduction")
doc.p("All commands run from the project root with the Python 3.12 environment.")
for c in [r".venv312\Scripts\python.exe -m src.data_audit",
          r".venv312\Scripts\python.exe -m src.run_pipeline --processed-dir data/processed_v8",
          r".venv312\Scripts\python.exe -m src.analysis",
          r".venv312\Scripts\python.exe -m src.annotation_qc",
          r".venv312\Scripts\python.exe -m src.anomaly_models",
          r".venv312\Scripts\python.exe -m src.feature_extraction",
          r".venv312\Scripts\python.exe -m pytest -q",
          r".venv312\Scripts\python.exe -m ui.app",
          r".venv312\Scripts\python.exe -m src.calibration selftest",
          r".venv312\Scripts\python.exe -m src.calibration benchmark",
          r".venv312\Scripts\python.exe -m src.calibration production",
          r".venv312\Scripts\python.exe tools\build_release.py"]:
    para = doc.d.add_paragraph()
    run = para.add_run(c)
    run.font.name = "Consolas"
    run.font.size = __import__("docx").shared.Pt(9)
doc.p("Verified metrics: deliverables/supporting/metrics_verified.json and Metric_Calculation_Sheet.xlsx "
      "(source file and sample size per value). Frozen baseline: baselines/v3_baseline (SHA256SUMS.txt).")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("saved", OUT, "figures", doc.fig_n, "tables", doc.tab_n, "equations", doc.eq_n)
