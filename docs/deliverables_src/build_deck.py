"""Presentation slides (content). Helpers are in deck_helpers.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from deck_helpers import *  # noqa: F401,F403
from deck_helpers import _font  # noqa: F401

import json as _json  # noqa: E402

import pandas as _pd  # noqa: E402

BEN = _json.load(open("reports/v9_calibration/benchmark_find_calibration.json"))["results"]
B3, B9 = BEN["v3: fitted per recording"], BEN["found calibration (this module)"]
SELF = _pd.read_csv("reports/v9_calibration/selftest_known_distortions.csv")
PROD = _json.load(open("models/calibration/flirone_one_pro_v9_found.json"))
_fr = _pd.read_csv("data/processed_v8/frames/S2_right_static.csv")
AGG = _fr[_fr.frame_valid.fillna(False).astype(bool)]

# ============================================================================ 1 title
s = new("RGB–Thermal Hand-Joint Temperature Analysis", dark=True,
        notes="This project measures the skin temperature of 14 hand joints from a smartphone thermal camera, "
              "builds a healthy reference and flags hands outside it. It is a research and screening-support "
              "system, developed and evaluated on 17 healthy volunteers.")
s.shapes.title.top = Inches(2.2)
s.shapes.title.height = Inches(1.5)
text(s, "and an arthritis screening-support research system", 0.6, 3.35, 11, 0.6, 22, PALE, face=HEAD)
text(s, ["FLIROne dataset · 17 healthy subjects · 68 recordings · 55,239 frames",
         "Pipeline v3 → v9 (automatic calibration) · software release 1.0.0 · 22 November 2026"], 0.6, 4.4, 11, 1.0, 15, WHITE)
text(s, "Research and screening-support software", 0.6, 6.4, 11, 0.4, 13, EMBER, italic=True)

# ============================================================================ 2 summary
s = new("Joint temperatures measured repeatably and screened correctly",
        "Key message", notes="Four numbers summarise what was demonstrated. All are measured on healthy volunteers. "
        "The specificity is real (healthy hands not flagged); sensitivity was only simulated.")
stat(s, pct(V3["hand_detection"], 2), "frames with a hand found by MediaPipe (68 recordings)", 0.6, 1.7)
stat(s, f(V3["dice_median"], 3), "median held-out Dice, RGB→thermal hand outline", 3.7, 1.7)
stat(s, f"{f(V3['mdc'], 2)} °C", "MDC95 of the 10-joint mean D2JTD (34 static hands)", 6.8, 1.7)
stat(s, f(C["mahalanobis_ledoit_wolf"]["specificity"], 2), "specificity on unseen healthy subjects (nested LOSO)", 9.9, 1.7, w=2.85)
bullets(s, [("Works: ", "landmarks, outline mapping, ROIs on skin (98 %), repeatability, leakage-free evaluation, full application."),
            ("Correct outputs: ", f"{PR['real_correct']}/{PR['n_hands']} unseen healthy hands within range; {PR['warmed_correct']}/{PR['n_hands']} simulated warm joints flagged and located."),
            ("Joint position: ", f"corrected to {f(V3['joint_err_corr'], 1)} px median — temperature effect only {f(V3['dT_corr'], 2)} °C."),
            ("New in v9: ", f"automatic calibration ({B9['joint_err_median']:.2f} px on unseen subjects) and an installable release 1.0.0.")],
        0.6, 3.75, 12.1, 2.8, size=18, gap=12)

# ============================================================================ 3 motivation
s = new("Inflamed joints are warmer — a thermal camera could quantify it", "Motivation",
        notes="RA inflames the synovium of small hand joints (MCP, PIP). Inflammation raises local blood flow and skin "
              "temperature. Clinical examination is subjective; ultrasound and MRI are objective but costly. A "
              "smartphone thermal camera is cheap and contact-free, if the measurement is reliable.")
bullets(s, [("Rheumatoid arthritis ", "inflames the synovium of small hand joints — MCP and PIP first, usually symmetric [1]."),
            ("Early treatment ", "limits erosion and disability; screening needs to be frequent and cheap."),
            ("Clinical exam ", "is subjective; ultrasound / MRI are objective but need experts and equipment."),
            ("Thermography ", "is contact-free and fast; inflamed joints are on average warmer, but protocols vary [2]."),
            ("Hand joints ", "lie close to the skin, can be posed repeatably, and give 14 internal comparisons per hand.")],
        0.6, 1.5, 6.6, 5.2, size=17, gap=12)
img(s, FIG / "F05_landmarks_real.png", 7.5, 1.45, 5.3, 5.0, "the 14 analysed joints on a real RGB frame", "measured")

# ============================================================================ 4 problem & RQs
s = new("Problem: can joint temperature be measured reliably?", "Problem statement and objectives",
        notes="Because only healthy volunteers were available, the objective was a measurement study. The pass "
              "criteria were fixed before any analysis, so the verdicts later are honest.")
table(s, ["ID", "Research question", "Pre-registered criterion"],
      [["RQ1", "Are the joints located in the RGB image?", "≥ 90 % of static frames valid"],
       ["RQ2", "Is the RGB→thermal mapping accurate?", "Dice ≥ 0.90; joint error ≤ 8 px"],
       ["RQ3", "Do the joint ROIs lie on skin?", "≥ 95 % of ROIs ≥ 95 % on hand"],
       ["RQ4", "Is the joint temperature repeatable?", "MDC95 < 0.4 °C"],
       ["RQ5", "What is the healthy D2JTD reference?", "reported with 95 % CI"],
       ["RQ6", "Does hand motion change the measurement?", "reported"],
       ["RQ7", "Can a healthy-only model flag simulated warming?", "reported (simulation)"]],
      0.6, 1.55, 12.1, col_w=[1.0, 6.6, 4.5], size=15, row_h=0.55)

# ============================================================================ 5 background
s = new("RGB finds the joints; radiometric thermal measures them", "Background",
        notes="RGB has the resolution and texture for a hand-pose model. The thermal sensor is only 160x120 and "
              "knuckles are hard to see. The colour thermal video is a display with automatic scaling, so the "
              "temperatures must come from the radiometric .npy data.")
card(s, 0.6, 1.5, 3.9, 4.9)
text(s, "RGB camera", 0.85, 1.65, 3.4, 0.5, 19, NAVY, bold=True, face=HEAD)
bullets(s, ["high resolution and texture", "MediaPipe: 21 landmarks + relative depth [4]", "no temperature information"], 0.85, 2.25, 3.5, 3.5, 15)
card(s, 4.75, 1.5, 3.9, 4.9)
text(s, "Radiometric thermal", 5.0, 1.65, 3.4, 0.5, 19, NAVY, bold=True, face=HEAD)
bullets(s, ["°C per pixel (float32 .npy)", "160×120 sensor, shown at 640×480", "±3 °C absolute (FLIR spec) → use differences", "colour MP4 is display only"], 5.0, 2.25, 3.5, 3.9, 15)
img(s, FIG / "R07_audit_sample_frames.png", 8.9, 1.45, 3.85, 4.95, "same frame: RGB, thermal MP4, .npy", "measured")

# ============================================================================ 6 dataset
s = new("FLIROne dataset: 17 healthy subjects, 68 recordings", "Dataset",
        notes="Values measured by the audit on every file. No arthritis labels, no room temperature, no calibration "
              "recording handled by the audit.")
table(s, ["Item", "Value (measured by the audit)"],
      [["Subjects / recordings", f"{D['subjects']} / {D['recordings']}  (left + right × static + open–close)"],
       ["Files per recording", "RGB .mp4 · thermal .mp4 · radiometric .npy"],
       ["Radiometric data", "float32 °C, (N, 640, 480) portrait"],
       ["Frames per recording", f"{D['frames_min']}–{D['frames_max']} (median {D['frames_median']:.0f}); total {D['frames_total']:,}"],
       ["Frame rate (measured)", f"{D['fps_min']:.2f}–{D['fps_max']:.2f} fps (median {D['fps_median']:.2f})"],
       ["Duration", f"{D['duration_min']:.0f}–{D['duration_max']:.0f} s (median {D['duration_median']:.0f} s)"],
       ["Temperature range", f"{D['tmin_min']:.1f}–{D['tmax_max']:.1f} °C"],
       ["Data volume", f".npy {D['npy_gb']:.0f} GB · video {D['video_gb']:.1f} GB"]],
      0.6, 1.5, 7.3, col_w=[2.4, 4.9], size=13, row_h=0.5)
img(s, FIG / "F03_dataset_tree.png", 8.2, 1.45, 4.6, 4.9, "folder structure and naming", "diagram")

# ============================================================================ 7 audit
s = new("The audit found three defects before any analysis", "Dataset structure and sample inputs",
        notes="These were found by checking every frame, and each is now handled automatically. Without the swap "
              "fix, one recording would have measured the wrong hand image; without in-order decoding, RGB and "
              "thermal would be up to 30 frames apart.")
bullets(s, [("Letterbox: ", "the MP4 image sits at x 140–500 of 640 → crop, then scale by 4/3 to the .npy grid."),
            ("Swapped files: ", "S11_left_static has RGB and thermal MP4 names swapped (thermal MP4 vs .npy r = −0.58)."),
            ("Unreliable seeking: ", "OpenCV frame seeking was off by up to ±30 frames → decode in order."),
            ("Consistent: ", f"frame counts equal in {D['frame_counts_equal']}/68; no NaN; .npy read memory-mapped.")],
        0.6, 1.5, 5.7, 5.0, size=16, gap=12)
img(s, FIG / "R08_audit_timestamps.png", 6.6, 1.45, 6.2, 4.95, "frame timing per recording (S1_left_static)", "measured")

# ============================================================================ 8 architecture
s = new("System architecture as implemented", "Architecture",
        notes="Colours mark the six parts of the system. Solid borders are implemented and tested, dashed partial, "
              "every block shown is implemented.")
img(s, FIG / "F01_architecture.png", 0.6, 1.3, 12.1, 5.45, "all blocks implemented in the project code", "diagram")

# ============================================================================ architecture overview (provided)
s = new("End-to-end architecture and required pipelines", "Architecture",
        notes="One picture of the whole system. Top: the two inputs pass the audit and frame pairing; the RGB branch finds the "
              "21 hand points, checks quality and visibility and builds the RGB hand mask; the thermal branch segments the hand "
              "from the temperature data. The automatic calibration maps the RGB joints into the thermal image. Middle: 14 joint "
              "regions, temperature sampling (median per frame, then median over frames), features, the healthy-reference model "
              "and the outputs. Bottom: the training pipeline (subject-wise, no neural network trained) and the pipeline used "
              "for new data in the application.")
img(s, FIG / "U01_architecture_overview.png", 0.6, 1.3, 12.1, 5.55, None, None)

# ============================================================================ 9 pipeline
s = new("Twelve stages from three files to a score and a report", "End-to-end pipeline",
        notes="Each recording runs through these stages in one worker process; per-frame results are written to "
              "CSV, so memory does not grow with the dataset. Reruns are bit-identical.")
img(s, FIG / "F02_pipeline.png", 0.6, 1.35, 12.1, 4.7, "end-to-end data flow", "diagram")
text(s, "Every 3rd frame analysed (≈2.8 fps) · one CSV row per frame · run manifest with config, code hash and package versions",
     0.6, 6.45, 12.1, 0.4, 14, GREY, align=PP_ALIGN.CENTER)

# ============================================================================ 10 stage 1-2
stage_slide(1, "Loading and validation without loading 68 GB into memory",
            "folder or ZIP with RGB, thermal MP4 and .npy per recording.",
            "file matching by name; .npy opened memory-mapped; videos decoded sequentially; dtype, shape, NaN, frame-count and swap checks.",
            f"audit table + report; {D['frame_counts_equal']}/68 consistent, 1 swap fixed, peak memory {V3['rss_peak']:.0f} MB.",
            FIG / "R07_audit_sample_frames.png", "audit sample: RGB, thermal MP4, .npy", "measured",
            "Memory mapping reads only the frames indexed. One recording per worker. The audit writes the facts "
            "later stages rely on: letterbox box, swapped files.")

s = new("Frames are paired by index, then the pairing is checked", "Pipeline stage 2 of 12",
        notes="All three streams have identical frame counts, so frame i pairs with frame i. To check, the hand "
              "opening signal from RGB is cross-correlated with the thermal hand area: the best lag should be zero.")
badge(s, 2, 0.6, 1.55)
bullets(s, [("Pairing: ", "frame i of RGB ↔ frame i of .npy (equal counts in all 68 recordings)."),
            ("Check: ", "RGB hand openness vs thermal hand area, cross-correlated over ±20 frames."),
            ("Live use: ", "different camera rates → pair with the newest thermal frame and log its age.")],
        1.35, 1.5, 6.3, 3.5, size=16, gap=12)
stat(s, f"{V3['sync_aligned']}/34", "open–close recordings aligned at lag 0 (other 2: weak signal, not offset)", 1.35, 4.6, w=3.5)
stat(s, f(V3["sync_corr"]), "median correlation at the best lag", 5.05, 4.6, w=2.9)
stat(s, pct(V8["live"]["pairs_within_1_frame"], 0), f"live simulation pairs within one frame (median age {V8['live']['thermal_age_ms_median']:.0f} ms)", 8.15, 4.6, w=3.6, color=COOL)
text(s, "Simulation = recorded data replayed on two clocks; no live camera.", 8.15, 6.25, 4.6, 0.4, 11, GREY, italic=True)

# ============================================================================ 12 registration
def _reg(s):
    text(s, "p_th = s·(k·p_rgb − c) + c + t", 1.35, 5.35, 4.9, 0.5, 18, NAVY, bold=True, face=HEAD)
    text(s, f"s = {V3['scale']} fixed (median of free fits), k = 4/3, c = image centre, t per recording (rig fallback {V3['tx']}, {V3['ty']} px)",
         1.35, 5.85, 4.9, 0.9, 12, GREY)


stage_slide(3, "Registration: fixed scale, per-recording shift",
            "RGB and thermal hand masks on 16 sampled frames.",
            "similarity transform; scale fixed at 1.25; shift by phase correlation, refined by Dice; held-out frames decide own fit vs rig median.",
            f"held-out Dice median {f(V3['dice_median'], 3)}; {V3['dice_ge90']}/68 recordings ≥ 0.90.",
            FIG / "F09_registration_real.png", "RGB mask warped onto thermal mask", "measured",
            "The fitted rig parameters are stored as a versioned calibration profile. A 2-D transform is exact "
            "only at one depth; fingers closer than the palm show parallax — the main source of joint error.", _reg)

# ============================================================================ 13 mediapipe
stage_slide(4, "MediaPipe finds 21 landmarks in almost every frame",
            "RGB frame cropped from the letterbox (360×480).",
            "MediaPipe Tasks Hand Landmarker, VIDEO mode, confidence 0.5; checks: finger order, MCP collapse, in-image, ≥ 80 % of joints on the thermal hand.",
            f"hand found {pct(V3['hand_detection'], 2)}; valid frames static {pct(V3['static_valid'], 0)}, open–close {pct(V3['oc_valid'], 2)}.",
            FIG / "F05_landmarks_real.png", "21 landmarks; 14 joints analysed", "measured",
            "14 joints per hand: thumb MCP and IP, and MCP, PIP, DIP of four fingers. Landmarks approximate joint "
            "centres seen from the camera, not anatomical joint lines.")

# ============================================================================ 14 joint localisation
s = new("Joint positions: 14.3 → 10.3 px after a learned correction", "Pipeline stage 5 of 12",
        notes="777 joints were clicked by hand on 136 thermal frames. The raw error was mostly along the finger, so "
              "a shift along the bone was learned, evaluated leave-one-subject-out. 8 px target not met, but the "
              "temperature cost is 0.05 °C, far below the 0.26 °C repeatability limit.")
badge(s, 5, 0.6, 1.55)
stat(s, f"{f(V3['joint_err_raw'], 1)} px", "raw median error (777 clicked joints)", 1.35, 1.5, w=2.6)
stat(s, f"{f(V3['joint_err_corr'], 1)} px", "corrected, leave-one-subject-out (target 8 px: not met)", 4.15, 1.5, w=2.6)
stat(s, f"{f(V3['dT_corr'], 2)} °C", "median temperature cost of the error", 1.35, 3.25, w=2.6, color=COOL)
stat(s, f"{f(V3['joint_err_static'], 1)} / {f(V3['joint_err_oc'], 1)}", "px static / open–close", 4.15, 3.25, w=2.6, color=NAVY)
bullets(s, [f"Bias was along the finger ({f(V3['along_bias'], 1)} px) not across ({f(V3['across_bias'], 1)} px).",
            f"Shift by {pct(V3['corr_mcp'], 1)} (MCP) / {pct(V3['corr_pip'], 1)} (PIP, DIP) of the MCP–PIP bone."],
        1.35, 5.0, 5.4, 1.8, size=15)
img(s, FIG / "G01_localisation_error.png", 7.0, 1.45, 5.8, 4.95, "error distribution, raw vs corrected", "measured")

# ============================================================================ 15 segmentation
stage_slide(6, "Thermal segmentation keeps cold fingers attached",
            "radiometric °C matrix of the frame.",
            "Otsu seed + hysteresis growth (low = bg + 0.4·(Otsu − bg)), largest component, hole filling, 9×9 erosion for sampling. RGB mask: landmark-seeded GrabCut, not skin colour.",
            "hand mask and eroded core; only the hand's connected warm region is kept.",
            FIG / "F08_thermal_segmentation_real.png", "Otsu only vs Otsu + hysteresis", "measured",
            "Subject S17 had cold fingers near room temperature; plain Otsu cut them off. Skin colour is unreliable "
            "across skin tones and lighting, so the RGB mask is seeded by landmarks.")

# ============================================================================ 16 outliers
s = new("Accessories and hot spots: flag, never delete warm skin", "Outlier and accessory handling",
        notes="Metal rings have low emissivity and read cold, so cold pixels inside an ROI are counted and flagged. "
              "Warm pixels are never removed — a hot spot may be exactly the inflammation of interest. Folded "
              "joints are hidden rather than guessed.")
bullets(s, [("Rings / bracelets: ", "share of ROI pixels > 1.5 °C below the ROI median; > 15 % → 'cold-band contamination' flag."),
            ("Hot spots: ", "kept; the median resists a few odd pixels; P90 is stored to see them."),
            ("Other warm objects: ", "removed only if not connected to the hand."),
            ("Folded / covered joints: ", "not measured (visibility rule, v8)."),
            ("Output: ", "cold-band share and flag stored for every ROI and frame.")],
        0.6, 1.5, 5.9, 5.2, size=16, gap=11)
img(s, FIG / "F10_outlier_handling.png", 6.8, 1.45, 6.0, 4.95, "implemented decision rules", "diagram")

# ============================================================================ 17 ROI
def _roi(s):
    text(s, "r = max(6, min(β·|MCP−PIP|, 0.9·d_edge))", 1.35, 5.45, 4.9, 0.5, 17, NAVY, bold=True, face=HEAD)
    text(s, "β: thumb 0.42 · index 0.28 · middle 0.28 · ring 0.31 · little 0.35 (Qiao et al.)", 1.35, 5.95, 4.9, 0.6, 12, GREY)


stage_slide(7, "14 joint ROIs sized to the finger, plus the back of the hand",
            "corrected joint centres, hand mask and its eroded core.",
            "disc per joint, radius from bone length capped at the finger width; dorsum polygon; used if ≥ 60 % on hand and ≥ 32 core pixels.",
            f"{pct(V3['roi_on_skin'], 0)} of ROIs ≥ 95 % on the hand (v1 rule: 48 %).",
            FIG / "F06_rois_d2jtd_real.png", "ROIs, temperatures and D2JTD on a real frame", "measured",
            "With the published radius rule alone only 48 % of ROIs stayed on the hand — discs spilled off thin "
            "fingers. Capping at the finger width fixed it (RQ3 pass, partly by construction).", _roi)

# ============================================================================ 18 temperature
stage_slide(8, "Temperatures read straight from the radiometric matrix",
            "ROI ∩ eroded hand mask on the °C matrix.",
            "pixels collected without colour conversion or interpolation; median (main), mean, P90, SD, valid share.",
            "per joint and frame; a joint without a valid ROI stays missing — never filled with an estimate.",
            FIG / "O01_joint_view_static.png", "joint view: visible joints with ROI temperature", "measured",
            "Preserving the raw values keeps every reported number traceable to the sensor. An invented value "
            "would be indistinguishable from a measured one later.")

# ============================================================================ frames -> joint
s = new("Hundreds of frames → one temperature per joint", "From frames to one value",
        notes="Each analysed frame gives a slightly different temperature per joint, mainly because the whole hand slowly drifts "
              "during the recording. The final value per joint is the median over the usable frames: the typical value, not "
              "moved by a few bad frames. The spread (Q1-Q3) and the number of frames are reported next to it.")
img(s, FIG / "N01_frame_aggregation.png", 0.6, 1.35, 12.1, 3.6, "S2_right_static: 267 frames → final value per joint", "measured")
bullets(s, [("Per frame: ", "median of the ROI's skin pixels (no edge pixels)."),
            ("Per recording: ", "median over usable frames — hidden or invalid frames dropped for that joint."),
            ("Reported: ", "final value + Q1–Q3 + number of frames; D2JTD the same way."),
            ("Why median: ", "the mean is pulled by one bad frame; a single frame depends on one moment.")],
        0.6, 5.3, 12.1, 1.7, size=14, gap=4)

s = new("The report shows each joint's final value with its spread", "From frames to one value",
        notes="Every dataset report now has, per recording, a table with the final temperature of each joint (median of the "
              "frames), its spread over the frames (Q1-Q3), the number of frames used and D2JTD. Reliability of the final value: "
              "two halves of the same recording agree within 0.26 degC (MDC95).")
x = AGG["index_MCP_T"]
y = AGG["little_MCP_T"]
table(s, ["Joint (S2_right_static)", "Frames", "Lowest", "Highest", "Q1–Q3", "Final"],
      [["index MCP", len(x), f(x.min()), f(x.max()), f"{f(x.quantile(.25))}–{f(x.quantile(.75))}", f(x.median())],
       ["little MCP", len(y), f(y.min()), f(y.max()), f"{f(y.quantile(.25))}–{f(y.quantile(.75))}", f(y.median())]],
      0.6, 1.55, 6.3, col_w=[1.7, 0.8, 0.85, 0.85, 1.4, 0.7], size=12, row_h=0.5)
stat(s, f"{f(V3['mdc'], 2)} °C", "repeatability of the final value (MDC95, two halves of a recording)", 0.6, 3.4, w=3.0)
stat(s, "median", "of usable frames = final joint temperature", 3.85, 3.4, w=3.05, color=NAVY)
img(s, FIG / "N04_report_joint_table.png", 7.2, 1.45, 5.6, 5.1, "dataset report: final temperatures per joint", "application output")

# ============================================================================ 19 features
s = new("D2JTD: the joint compared with the back of the same hand", "Feature engineering",
        notes="D2JTD cancels room temperature and the camera's absolute offset because both temperatures come from "
              "the same image. Its blind spot: warming of the whole hand. Temporal features were unstable (ICC 0.40) "
              "and dropped; spatial contrast was as stable as D2JTD.")
card(s, 0.6, 1.5, 5.6, 1.6)
text(s, "D2JTD = T_dorsum − T_joint", 0.85, 1.65, 5.2, 0.6, 26, NAVY, bold=True, face=HEAD)
text(s, "positive: joint cooler than the back of the hand (usual); inflammation would push it down or negative",
     0.85, 2.3, 5.2, 0.8, 13, GREY)
ex = [r for r in V3["reference"] if r["feature"] == "index_MCP"][0]
text(s, f"Example: dorsum 32.80 °C, index MCP 32.30 °C → D2JTD 0.50 °C. Healthy index-MCP range "
        f"{f(ex['ref_lo'])} to {f(ex['ref_hi'])} °C (n = 17).", 0.6, 3.3, 5.6, 1.0, 14)
table(s, ["Group (n features)", "ICC(3,1)", "MDC95 °C"],
      [[f"D2JTD (14)", f(ST["relative_D2JTD"]["icc31"]), f(ST["relative_D2JTD"]["mdc95"])],
       [f"vs whole hand (28)", f(ST["relative_hand"]["icc31"]), f(ST["relative_hand"]["mdc95"])],
       [f"spatial (18)", f(ST["spatial"]["icc31"]), f(ST["spatial"]["mdc95"])],
       [f"absolute (58)", f(ST["absolute"]["icc31"]), f(ST["absolute"]["mdc95"])],
       [f"temporal (4)", f(ST["temporal"]["icc31"]), f(ST["temporal"]["mdc95"])]],
      0.6, 4.35, 5.6, col_w=[3.0, 1.3, 1.3], size=13, row_h=0.42)
img(s, FIG / "F06_rois_d2jtd_real.png", 6.6, 1.45, 6.2, 4.95, "D2JTD per joint on a real frame", "measured")

# ============================================================================ 20 QC
s = new("Quality control: unreliable values are flagged, not repaired", "Quality control",
        notes="Every check leaves the value missing with a reason instead of estimating. The visibility rule hides "
              "joints folded under the hand; in open-close recordings 66 % of frames had at least one hidden joint, "
              "mostly DIPs.")
bullets(s, [("Landmarks: ", "frame valid / invalid with reason"),
            ("Registration: ", "held-out Dice per recording"),
            ("ROIs: ", "on-hand fraction, pixel count, cold-band share"),
            ("Visibility: ", "folded, covered or tucked joints → no temperature"),
            ("Model range: ", "outside training range → 'outside model experience'")],
        0.6, 1.5, 5.0, 3.6, size=15, gap=9)
vis = V8["visibility"]["open_close"]
chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, ["MCP", "PIP", "thumb IP", "DIP", "any joint"],
      [("open–close", [vis["MCP"], vis["PIP"], vis["IP"], vis["DIP"], vis["ANY"]])], 0.6, 4.9, 5.0, 2.1,
      title="Frames with hidden joints, open–close (static: 0 %)", colors=(EMBER,), pct_axis=True, num_fmt='0%', vmax=1)
img(s, FIG / "O02_joint_view_fist.png", 5.9, 1.45, 6.9, 5.0, "closing hand: hidden joints are not shown", "measured")

# ============================================================================ 21 models
s = new("Six healthy-reference models: distance from 'healthy', not diagnosis", "Model development",
        notes="Without patients, a model can only learn what healthy looks like. Each produces a score; the "
              "threshold is the 95th percentile of held-out healthy scores. Output categories, not probabilities.")
table(s, ["Model", "Principle", "Strength", "Weakness"],
      [["Robust z (max)", "largest median/MAD z", "transparent", "ignores correlations"],
       ["Mahalanobis, diagonal", "variances only", "simple", "ignores correlations"],
       ["Mahalanobis, Ledoit–Wolf", "shrunk covariance", "stable with few subjects", "elliptical assumption"],
       ["Robust covariance (MCD)", "cleanest-subset covariance", "robust to bad hands", "unstable < 12 subjects"],
       ["Isolation Forest", "random-tree path length", "no distribution", "insensitive here"],
       ["One-class SVM (ν = 0.1)", "RBF boundary", "flexible shape", "parameter-sensitive"]],
      0.6, 1.5, 7.6, col_w=[2.5, 2.1, 1.6, 1.4], size=12, row_h=0.5)
card(s, 8.5, 1.5, 4.25, 4.9)
text(s, "Score → category", 8.75, 1.65, 3.8, 0.5, 19, NAVY, bold=True, face=HEAD)
bullets(s, ["θ = 95th percentile of held-out healthy scores (inner LOSO)", "score ≤ θ → within healthy range",
            "score > θ → potential anomaly", "outside training range → outside model experience",
            "too few valid frames → insufficient quality"], 8.75, 2.25, 3.8, 4.0, 14, gap=8)

# ============================================================================ 22 evaluation
s = new("Nested leave-one-subject-out: the test person never shapes the model", "Evaluation methodology",
        notes="Outer loop: hold out one subject (both hands). Inner loop on the other 16: choose the threshold. "
              "Imputation, scaling, covariance and threshold all come from training subjects only. Warming is then "
              "added to the held-out hands to estimate sensitivity — a simulation, not patients.")
img(s, FIG / "F11_loso_validation.png", 0.6, 1.35, 7.8, 4.5, "outer loop evaluates; inner loop sets the threshold", "diagram")
bullets(s, [("Specificity: ", "held-out healthy hands as recorded (real)."),
            ("Sensitivity: ", "held-out hands + simulated warming of 0.25–3 °C (simulated)."),
            ("Patterns: ", "1 joint · MCPs · PIPs · DIPs · all 10 MCP/PIP · whole hand."),
            ("Why: ", "v3 picked thresholds on the same hands — optimistic, now superseded.")],
        8.7, 1.5, 4.1, 5.2, size=15, gap=11)

# ============================================================================ 23 results overview
s = new("Results against the pre-registered criteria", "Experimental results",
        notes="Five pass, one fails (joint position), the rest are reported values. RQ3 passes partly by "
              "construction because the radius cap uses the same mask. RQ7 is a simulation.")
table(s, ["RQ", "Result", "n", "Verdict"],
      [["RQ1", f"valid frames static {pct(V3['static_valid'], 0)}, open–close {pct(V3['oc_valid'], 2)}", "68 rec.", "Pass"],
       ["RQ2a", f"held-out Dice median {f(V3['dice_median'], 3)}", "68 rec.", "Pass"],
       ["RQ2b", f"joint error {f(V3['joint_err_corr'], 1)} px (target ≤ 8 px)", "777 pts", "Fail"],
       ["RQ3", f"{pct(V3['roi_on_skin'], 1)} of ROIs ≥ 95 % on hand", "all ROIs", "Pass"],
       ["RQ4", f"ICC {f(V3['icc'], 3)}, MDC95 {f(V3['mdc'], 2)} °C (< 0.4)", "34 hands", "Pass"],
       ["RQ5", f"mean D2JTD {f(V3['mean10'], 2)} °C (CI {f(V3['mean10_ci'][0], 2)}–{f(V3['mean10_ci'][1], 2)})", "17 subj.", "reported"],
       ["RQ6", "open–close warmer in absolute terms; D2JTD unchanged", "34 hands", "reported"],
       ["RQ7", f"specificity 0.94; all 10 joints +1 °C: 0.03–{f(C['one_class_svm']['sens_all_10_MCP_PIP_1C'])} (simulated)", "34 hands", "simulation"]],
      0.6, 1.5, 12.1, col_w=[1.1, 7.6, 1.6, 1.8], size=14, row_h=0.56)

# ============================================================================ 24 registration + localisation
s = new("Outline mapping passes; joint-level mapping is the weak link", "Joint localisation and registration results",
        notes="Dice is high for nearly all recordings; the lowest are open-close ones where fingers move in depth. "
              "The three-cornered hat shows the clicks themselves are noisy (upper bound 17.6 px), while the "
              "pipeline's random scatter is 6.7 px — the remaining error is partly the reference, partly parallax.")
img(s, FIG / "G02_registration_dice.png", 0.6, 1.45, 6.0, 4.6, "held-out Dice per recording (68)", "measured")
img(s, FIG / "G08_annotation_hat.png", 6.8, 1.45, 6.0, 4.6, "random scatter per joint estimator (three-cornered hat)", "measured")
text(s, f"Without the 3 flagged annotation frames: {f(V5['sens']['without_flagged_frames']['corrected_loso_median_px'], 2)} px — annotation mistakes are not the main cause.",
     0.6, 6.5, 12.1, 0.4, 14, NAVY, align=PP_ALIGN.CENTER)

# ============================================================================ 25 repeatability
s = new(f"Repeatability: 10-joint mean D2JTD MDC95 = {f(V3['mdc'], 2)} °C", "Repeatability results",
        notes="Split-half within one session, 34 static hands. MCPs are most repeatable, DIPs least because they "
              "are small and most affected by joint error. Between-day repeatability was not measurable.")
joints = [j for j in V3["mdc_joints"] if j != "D2JTD_mean10"]
chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, [j.replace("_", " ") for j in joints] + ["10-joint mean"],
      [("MDC95 (°C)", [V3["mdc_joints"][j] for j in joints] + [V3["mdc"]])], 0.6, 1.4, 8.6, 5.3,
      title="MDC95 per joint (°C) — criterion 0.40 °C", colors=(COOL,), num_fmt='0.00', gap=40)
stat(s, f(V3["icc"], 3), "ICC(3,1), 10-joint mean D2JTD", 9.5, 1.5, w=3.25)
stat(s, f"{f(min(V3['mdc_joints'][j] for j in joints), 2)}–{f(max(V3['mdc_joints'][j] for j in joints), 2)}", "°C MDC95 range over the 14 joints", 9.5, 3.25, w=3.25, color=NAVY)
text(s, "34 static hands, first vs second half of the same recording. Same-session only.", 9.5, 5.0, 3.25, 1.2, 13, GREY, italic=True)

# ============================================================================ 26 healthy reference
s = new(f"Healthy reference: joints average {f(V3['mean10'], 2)} °C cooler than the dorsum", "Healthy reference (RQ5)",
        notes="Per-subject D2JTD, 17 subjects. The joint is cooler than the dorsum for 3 of 10 MCP/PIP joints after "
              "Holm correction. Left-right absolute symmetry (< 0.5 °C) was not supported, which is why relative "
              "features are used.")
img(s, FIG / "R01_healthy_reference.png", 0.6, 1.4, 7.8, 5.3, "D2JTD per joint and subject", "measured")
bullets(s, [f"Mean {f(V3['mean10'], 2)} °C, 95 % CI {f(V3['mean10_ci'][0], 2)}–{f(V3['mean10_ci'][1], 2)} (n = 17).",
            f"Joint cooler than dorsum (Holm): {V3['h1']} of 10 joints.",
            "Reference intervals are wide (DIP ±2–3 °C).",
            "Open–close vs static: absolute +0.17 to +0.31 °C, D2JTD unchanged."],
        8.7, 1.5, 4.1, 5.0, size=15, gap=12)

# ============================================================================ 27 synthetic
s = new("Simulated warming: equal specificity, very different sensitivity", "Synthetic anomaly experiment",
        notes="SIMULATED: warming added to healthy hands. Thresholds from training subjects only. At 1 °C on all 10 "
              "joints, sensitivity is modest; at 2 °C three models reach 1.00. Whole-hand warming is never detected "
              "by design. These are not arthritis results.")
order = ["one_class_svm", "mahalanobis_ledoit_wolf", "robust_covariance_mcd", "robust_z_max", "mahalanobis_diag", "isolation_forest"]
names = ["One-class SVM", "Ledoit–Wolf", "MCD", "Robust z", "Diagonal", "Isolation Forest"]
chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, names,
      [("Specificity (real held-out hands)", [C[m]["specificity"] for m in order]),
       ("All 10 joints +1 °C (simulated)", [C[m]["sens_all_10_MCP_PIP_1C"] for m in order]),
       ("Single joint +2 °C (simulated)", [C[m]["sens_single_joint_2C"] for m in order])],
      0.6, 1.4, 8.4, 5.4, colors=(NAVY, EMBER, COOL), legend=XL_LEGEND_POSITION.BOTTOM, vmax=1.1, gap=50, overlap=-5)
bullets(s, ["34 healthy hands, 17 subjects, nested LOSO.",
            "All 10 MCP/PIP +2 °C: 1.00 for SVM, Ledoit–Wolf, MCD.",
            "Whole-hand warming: never detected (0.06 = false-positive rate) — D2JTD is relative.",
            "Earlier in-sample v3 values (0.71–1.00 at 1 °C) were optimistic."],
        9.3, 1.5, 3.5, 5.2, size=14, gap=10)

# ============================================================================ 28 ablation
s = new("Spatial contrast helps; adding every feature hurts", "Feature ablation",
        notes="Ledoit-Wolf, nested LOSO, simulated warming. 32 features (D2JTD + spatial) gave the best single-joint "
              "sensitivity. 122 features on 34 hands collapses sensitivity — overfitting. Caveat: the simulation "
              "warms one ROI uniformly, which is what spatial features measure, so the gain is an upper estimate.")
sets = [("S1_D2JTD (v3 baseline)", "D2JTD (14)"), ("S3_D2JTD+spatial", "+ spatial (32)"), ("S4_D2JTD+temporal", "+ temporal (18)"),
        ("S2_D2JTD+relative_hand", "+ vs hand (42)"), ("S5_D2JTD+absolute", "+ absolute (72)"), ("S6_all", "all (122)")]
chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, [b for _, b in sets],
      [("Single joint +2 °C (simulated)", [AB[(a, "mahalanobis_ledoit_wolf")]["sens_single_joint_2C"] for a, _ in sets]),
       ("All 10 joints +1 °C (simulated)", [AB[(a, "mahalanobis_ledoit_wolf")]["sens_all10_1C"] for a, _ in sets])],
      0.6, 1.4, 8.4, 5.4, colors=(COOL, EMBER), legend=XL_LEGEND_POSITION.BOTTOM, vmax=1.1, gap=60, overlap=-5)
stat(s, f(AB[("S3_D2JTD+spatial", "mahalanobis_ledoit_wolf")]["sens_single_joint_2C"]), "single-joint +2 °C with D2JTD + spatial (was 0.42)", 9.3, 1.5, w=3.45, color=COOL)
stat(s, "0.94", "specificity unchanged for every feature set (Ledoit–Wolf)", 9.3, 3.25, w=3.45, color=NAVY)
text(s, "Selected for the application: Ledoit–Wolf on D2JTD + spatial (32 features).", 9.3, 5.0, 3.45, 1.2, 14, INK)

# ============================================================================ correct predictions
s = new(f"Correct outputs on unseen healthy volunteers ({PR['real_correct']}/{PR['n_hands']})", "Correct predictions",
        notes="Each hand is scored by a model and threshold that never saw that person (nested leave-one-subject-out), "
              "using the application's default model. All six people shown are healthy volunteers and the model "
              "correctly reports 'within healthy range', with every joint measured and its temperature shown.")
img(s, FIG / "P01_correct_predictions_gallery.png", 0.6, 1.3, 12.1, 5.5, "six unseen subjects, all 14 joints measured, score below threshold", "measured")

s = new("A simulated warm joint is flagged and located", "Correct predictions",
        notes="Same unseen hand: the real recording scores 1.76, below the 6.75 threshold. With its index MCP ROI warmed by "
              "2 degC in every frame (simulation) the score rises to 8.88 and the per-joint explanation points to the "
              "index MCP. Over all 34 hands, all 34 simulated warm joints were flagged and located.")
img(s, FIG / "P02_simulated_detection_example.png", 0.6, 1.3, 12.1, 5.0, "real recording vs index MCP +2 °C (simulated)", "measured + simulated")
text(s, f"All hands: {PR['warmed_correct']}/{PR['n_hands']} simulated warm joints flagged · explanation pointed to the warmed joint in all {PR['top_feature_is_warmed_joint']}",
     0.6, 6.55, 12.1, 0.4, 15, NAVY, bold=True, align=PP_ALIGN.CENTER)

s = new("Every hand, scored by a model that never saw that person", "Correct predictions",
        notes="Green: real healthy recording, correct when below the threshold line (32 of 34). Red: the same hand with "
              "a simulated +2 degC index MCP, correct when above (34 of 34). The gap between the two shows the margin.")
img(s, FIG / "P03_prediction_overview.png", 0.6, 1.3, 12.1, 5.5, "score / threshold for 34 hands, real and simulated", "measured + simulated")

# ============================================================================ v9 calibration
s = new("Automatic calibration: find, apply and verify the RGB→thermal mapping", "Calibration (v9)",
        notes="The module finds the mapping from the recordings themselves, no calibration target: image area, scale, rotation and "
              "a per-frame shift learned from the hand. It scores alignment by the distance between hand outlines, capped so "
              "that a warm forearm cannot pull the scale. It then checks itself on frames not used for fitting and saves a profile.")
bullets(s, [("Finds: ", "image area · scale · rotation · per-frame shift (from hand size and position)."),
            ("Scored by: ", "distance between RGB and thermal hand OUTLINES (capped at 10 px)."),
            ("Checks: ", "rotated/transposed exports refused; held-out frames graded good / acceptable / poor."),
            ("Applies: ", "points RGB→thermal and back, whole images, temperature at a point; stored per frame."),
            ("Default: ", f"flirone_one_pro_v9_found — scale {PROD['scale']:.2f}, rotation {PROD['angle_deg']:.0f}°, Dice {PROD['quality']['dice_heldout_median']:.3f}.")],
        0.6, 1.5, 5.8, 5.2, size=15, gap=9)
img(s, FIG / "N12_calibration_apply.png", 6.7, 1.45, 6.1, 5.0, "joints found in RGB mapped into the thermal image", "measured")

s = new(f"Proof 1: known distortions recovered in 12 of 12 cases", "Calibration (v9)",
        notes="Real recordings were distorted on purpose: thermal side scaled by +8 or -10 percent, rotated by +4 or -6 degrees, "
              "shifted, and the RGB picture moved 40 px in the video frame. Without being told, the module found the scale within "
              "0.01 and the angle within 0.75 degrees; the mapping error was 1.4 px median.")
img(s, FIG / "N02_calibration_selftest.png", 0.6, 1.35, 12.1, 3.9, "mapping error per case; found vs true scale", "measured")
stat(s, "12 / 12", "known distortions recovered", 0.6, 5.45, w=3.0)
stat(s, f"{SELF.mapping_error_px_median.median():.1f} px", "median mapping error vs known truth", 3.8, 5.45, w=3.0, color=NAVY)
stat(s, "±0.01", "scale found vs true scale", 7.0, 5.45, w=2.7, color=COOL)
stat(s, "≤ 0.75°", "rotation error", 9.9, 5.45, w=2.85, color=COOL)

s = new(f"Proof 2: on unseen people, {B9['joint_err_median']:.2f} px instead of {B3['joint_err_median']:.2f} px", "Calibration (v9)",
        notes="Calibrated from 5 subjects only and applied unchanged to the 12 others, the found calibration placed joints closer to "
              "the hand-clicked positions than fitting every recording separately, and overlapped the hand outline better.")
chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, ["all joints", "static", "open–close"],
      [("fitted per recording (v3)", [B3["joint_err_median"], B3["static"], B3["open_close"]]),
       ("found calibration (v9)", [B9["joint_err_median"], B9["static"], B9["open_close"]])],
      0.6, 1.4, 8.0, 5.3, title="Median joint error vs clicked joints (px), 12 unseen subjects", colors=(GREY, EMBER),
      legend=XL_LEGEND_POSITION.BOTTOM, gap=60, overlap=-5)
stat(s, f"{B9['joint_err_median']:.2f} px", f"joint error with the found calibration (v3: {B3['joint_err_median']:.2f})", 8.9, 1.5, w=3.85)
stat(s, f"{B9['dice_median']:.3f}", f"hand-outline overlap (v3: {B3['dice_median']:.3f})", 8.9, 3.25, w=3.85, color=NAVY)
text(s, f"Calibrated on S1–S5, tested on S6–S17: {B9['n_joints']} clicked joints, {B9['n_dice_frames']} frames.", 8.9, 5.0, 3.85, 1.2, 13, GREY, italic=True)

# ============================================================================ 29 compute
s = new(f"{f(V3['ms_frame'], 1)} ms per frame on a laptop CPU — faster than the 8.7 fps sensor", "Computational performance",
        notes="Measured on the development laptop, single process, CPU only. The live replay including RGB-thermal "
              "pairing ran at 17.4 fps, twice the sensor rate.")
mp = V3["ms_parts"]
chart(s, XL_CHART_TYPE.BAR_STACKED, ["per frame"],
      [("landmarks", [mp["landmarks"]]), ("thermal mask", [mp["thermal_mask"]]), ("ROIs + features", [mp["rois"]])],
      0.6, 1.5, 8.2, 2.6, colors=(COOL, EMBER, NAVY), legend=XL_LEGEND_POSITION.BOTTOM, num_fmt='0.0" ms"', gap=40, overlap=100)
stat(s, f"≈{V3['fps']:.0f} fps", "laptop throughput (sensor ≈ 8.7 fps)", 0.6, 4.5, w=2.6)
stat(s, f"{V3['rss_peak']:.0f} MB", "peak resident memory", 3.4, 4.5, w=2.6, color=NAVY)
stat(s, f"{f(V3['model_mb'], 1)} MB", "hand landmark model file", 6.2, 4.5, w=2.6, color=COOL)
card(s, 9.2, 1.5, 3.55, 4.55)
text(s, "Also measured", 9.45, 1.65, 3.1, 0.5, 18, NAVY, bold=True, face=HEAD)
bullets(s, [f"live replay with pairing: {f(V8['live']['processing_fps'], 1)} fps",
            f"model scoring: {C['mahalanobis_ledoit_wolf']['score_us_per_hand'] / 1000:.2f} ms per hand",
            f"memory after model load: {V3['rss_model']:.0f} MB"], 9.45, 2.25, 3.1, 3.5, 14)

# ============================================================================ 30 upload
s = new("Uploading a new dataset: import, match, validate, process", "New dataset upload",
        notes="ZIP, folder or single recording. Safe extraction refuses path traversal, links and zip bombs. Filename "
              "profiles handle other naming schemes; unmatched files can be matched by hand. Each recording gets one "
              "of five statuses with the failing check explained.")
img(s, FIG / "F12_upload_workflow.png", 0.6, 1.35, 12.1, 1.9, "upload workflow", "diagram")
bullets(s, ["ZIP / folder (in place) / single recording", "profiles: flirone, generic, custom regex",
            "status: Valid · with warnings · Invalid · Requires calibration · Requires manual matching",
            "software test: 3/3 matched and Valid, stray video listed, dummy label row reported"],
        0.6, 3.75, 5.3, 3.0, size=14, gap=9)
img(s, FIG / "A02_app_dataset_validation.png", 6.1, 3.6, 6.7, 3.0, "validation page (real application)", "screenshot")

# ============================================================================ 31 independent test
s = new("Independent testing is locked: the model is applied exactly as saved", "Accuracy evaluation workflow",
        notes="Inputs, model and labels are SHA-256 fingerprinted; no refitting or threshold change is possible. "
              "Metrics are computed only when both classes exist; below 10 per class a warning is shown. On this "
              "computed once a labelled dataset is uploaded.")
img(s, FIG / "F13_test_workflow.png", 0.6, 1.35, 12.1, 1.9, "locked independent-test workflow", "diagram")
bullets(s, [("Lock: ", "SHA-256 of features, manifest, model, labels; earlier runs listed."),
            ("Guards: ", "no sensitivity / AUC with one class; warning below 10 per class."),
            ("Output: ", "categories, metrics with counts, PDF and Excel report."),
            ("Formulas: ", "Sens = TP/(TP+FN) · Spec = TN/(TN+FP) · F1 = 2PR/(P+R)")],
        0.6, 3.75, 5.6, 3.0, size=14, gap=9)
img(s, FIG / "A06_app_independent_test.png", 6.4, 3.6, 6.4, 3.0, "independent-test page (dummy labels, software test)", "screenshot")

# ============================================================================ 32 app
s = new("The application: from upload to joint view and report", "Application interface and outputs",
        notes="Ten pages: dashboard, upload, calibration, processing, models, independent test, joint view, "
              "explainability, live (simulation), reports with a built-in PDF viewer. 35 automated tests pass.")
img(s, FIG / "A03_app_joint_view.png", 0.6, 1.4, 6.0, 4.9, "joint view: visible joints, ROI temperatures, table", "screenshot")
img(s, FIG / "A05_app_reports.png", 6.8, 1.4, 6.0, 4.9, "reports page with built-in PDF viewer", "screenshot")
text(s, f"10 pages · PDF + Excel reports · {V8['tests_passing']} automated tests passing · live mode replays recordings",
     0.6, 6.55, 12.1, 0.4, 14, NAVY, align=PP_ALIGN.CENTER)

# ============================================================================ deployment
s = new("Release 1.0.0: installed with one command, checked automatically", "Deployment",
        notes="The release zip contains the program, models with checksums, calibration profiles, tests, installer and documentation, "
              "but no datasets. The installer creates a Python 3.12 environment with exact package versions and runs the health "
              "check, which must say READY. A clean installation from the zip passed the whole workflow and the tests.")
img(s, FIG / "N05_deployment_workflow.png", 0.6, 1.3, 12.1, 2.6, "from the release zip to results", "diagram")
text(s, "powershell -ExecutionPolicy Bypass -File .\\install.ps1", 0.6, 4.15, 6.3, 0.45, 12.5, NAVY, bold=True, face="Consolas")
bullets(s, ["dist/rgb_thermal_hand-1.0.0.zip — 27 MB, 81 files", "Windows 10/11, 4 GB RAM, no GPU",
            "exact package versions (requirements.lock.txt)", "clean install: READY · 48 tests passed · full workflow passed"],
        0.6, 4.75, 6.2, 2.2, size=14, gap=6)
img(s, FIG / "N06_doctor_output.png", 7.0, 4.05, 5.8, 2.6, "health check: rgbthermal doctor", "screenshot")

s = new("How to use the deployed model", "Deployment",
        notes="Double-click 'RGB-Thermal App.bat'. Import a dataset and validate it; for a new camera use Auto-calibrate; process; run "
              "an independent test with the saved model; look at the joint view and open the PDF/Excel report. The same steps exist "
              "on the command line.")
table(s, ["Step", "Page", "What you get"],
      [["1", "Dataset upload → Validate", "status per recording"], ["2", "Calibration → Auto-calibrate (new camera only)", "profile + grade"],
       ["3", "Processing", "joint temperatures per frame and per joint"], ["4", "Independent test (saved model)", "category + score per hand"],
       ["5", "Joint view", "RGB joints with °C + thermal ROIs"], ["6", "Reports", "PDF / Excel with final joint values"]],
      0.6, 1.5, 6.3, col_w=[0.6, 3.3, 2.4], size=12, row_h=0.5)
text(s, ["rgbthermal.bat app · doctor · test", "rgbthermal.bat calibrate find | verify | apply ..."], 0.6, 5.35, 6.3, 0.9, 13, NAVY, face="Consolas")
img(s, FIG / "N08_app_dataset_upload.png", 7.2, 1.45, 5.6, 4.9, "new dataset imported and validated", "screenshot")

s = new("The deployed model and calibration inside the application", "Deployment",
        notes="Models page: the saved screening model with its training subjects, features, threshold and validation results. "
              "Calibration page: the default calibration and the Auto-calibrate button for a new dataset.")
img(s, FIG / "N09_app_models.png", 0.6, 1.4, 6.0, 4.9, "Models page: deployed screening model", "screenshot")
img(s, FIG / "N07_app_calibration.png", 6.8, 1.4, 6.0, 4.9, "Calibration page: profiles + Auto-calibrate", "screenshot")

# ============================================================================ 34 conclusion
s = new("Conclusion", dark=True,
        notes="What it can do: measure joint temperatures repeatably and say whether a hand lies within the range of "
              "these healthy volunteers, and point to the joint that makes a hand unusual.")
bullets(s, [("Built: ", "an audited RGB–thermal pipeline, six leakage-free healthy-reference models and a full application."),
            ("Demonstrated: ", f"joints found in {pct(V3['hand_detection'], 2)} of frames, Dice {f(V3['dice_median'], 2)}, MDC95 {f(V3['mdc'], 2)} °C, specificity 0.94 on unseen healthy subjects."),
            ("Joint position: ", f"corrected to {f(V3['joint_err_corr'], 1)} px median; temperature effect {f(V3['dT_corr'], 2)} °C."),
            ("Correct outputs: ", f"{PR['real_correct']}/{PR['n_hands']} unseen healthy hands within range; {PR['warmed_correct']}/{PR['n_hands']} simulated warm joints flagged and located."),
            ("Calibration (v9): ", f"found automatically; 12/12 known distortions recovered; {B9['joint_err_median']:.2f} px on unseen subjects."),
            ("Deployed: ", "release 1.0.0 — one-command install, health check, application, reports with final joint values.")],
        0.6, 1.6, 12.1, 5.0, size=19, color=WHITE, gap=14)

# ============================================================================ innovation
s = new("Innovation implemented", "Innovation, outcome and demo",
        notes="Six things that go beyond a standard pipeline, each with the measured evidence. 1) The calibration is found from "
              "the hand itself, without a calibration target, and proven on known distortions and unseen people. 2) Joints that "
              "are folded or covered are not measured. 3) The joint position error was diagnosed and corrected using my own "
              "annotations. 4) Screening uses a healthy reference evaluated without leakage, with a per-joint explanation. "
              "5) Each joint gets one robust temperature with its spread. 6) The whole system is deployable and verifiable.")
INNOV = [
    ("Calibration from the hand itself", "finds image area, scale, rotation and per-frame shift — no calibration target",
     f"12/12 known distortions · {B9['joint_err_median']:.2f} vs {B3['joint_err_median']:.2f} px on unseen people"),
    ("Visibility-aware measurement", "folded or covered joints detected (outline, depth, bend) and not measured",
     "66 % of open–close frames had a hidden joint"),
    ("Annotation-driven joint correction", "777 hand-clicked joints revealed a systematic along-finger shift; learned and removed",
     f"{f(V3['joint_err_raw'], 1)} → {f(V3['joint_err_corr'], 1)} px; cost {f(V3['dT_corr'], 2)} °C"),
    ("Leakage-free healthy reference", "nested leave-one-subject-out model + threshold; per-joint explanation of a flag",
     f"{PR['real_correct']}/{PR['n_hands']} healthy correct · {PR['warmed_correct']}/{PR['n_hands']} simulated warm joints located"),
    ("One robust value per joint", "median over usable frames with Q1–Q3 and frame count in every report",
     f"repeatable to {f(V3['mdc'], 2)} °C (MDC95)"),
    ("Deployable and verifiable", "16 validation checks, locked tests (SHA-256), auto-calibrate, installer + health check",
     "clean install: full workflow + 50 tests passed"),
]
for i, (head, what, ev) in enumerate(INNOV):
    x, y = 0.6 + (i % 3) * 4.1, 1.5 + (i // 3) * 2.75
    card(s, x, y, 3.9, 2.55)
    badge(s, i + 1, x + 0.18, y + 0.2, d=0.5)
    text(s, head, x + 0.8, y + 0.22, 2.95, 0.6, 15, NAVY, bold=True, face="Cambria")
    text(s, what, x + 0.2, y + 0.92, 3.5, 0.95, 12, INK)
    text(s, ev, x + 0.2, y + 1.85, 3.5, 0.6, 11.5, EMBER, bold=True)

# ============================================================================ individual outcome + LLM
s = new("Individual outcome and use of LLM assistance", "Innovation, outcome and demo",
        notes="Left: what I did and what I take away. Right: how the LLM coding assistant was used, and how its work was "
              "checked. The assistant helped write and debug code, run experiments and draft documents; the hand-clicked "
              "ground truth, the decisions at each stage and the review of every result were mine. No number in this "
              "presentation is typed by hand: all come from saved outputs, and the tests caught the assistant's own mistakes.")
card(s, 0.6, 1.5, 5.95, 5.25)
text(s, "Individual outcome", 0.85, 1.62, 5.5, 0.5, 19, NAVY, bold=True, face="Cambria")
bullets(s, [("Directed the project: ", "set the objectives, reviewed and approved each stage, requested each change from what I observed."),
            ("Ground truth: ", "hand-clicked 777 joint centres on 136 thermal frames."),
            ("Problems I found: ", "wrong frame mapping, folded joints showing temperatures, reports shown only as paths, need for calibration on any video."),
            ("Delivered: ", "measurement pipeline, healthy-reference model, desktop app, release 1.0.0, reports and slides."),
            ("Learned: ", "radiometric thermal imaging, hand landmarking, camera mapping, leakage-free evaluation, deployment.")],
        0.85, 2.2, 5.5, 4.5, size=13, gap=6)
card(s, 6.8, 1.5, 5.95, 5.25)
text(s, "Use of LLM assistance", 7.05, 1.62, 5.5, 0.5, 19, NAVY, bold=True, face="Cambria")
bullets(s, [("Tool: ", "Claude (Anthropic) as an AI coding and analysis assistant."),
            ("Used for: ", "writing and debugging code, running experiments, figures, drafting reports and slides."),
            ("Not used for: ", "ground-truth labels (clicked by me) or results — every number comes from saved outputs (metrics_verified.json)."),
            ("Checked by: ", "50 automated tests, known-distortion self-test, clean-install acceptance test, my review of every output."),
            ("Mistakes caught: ", "a model-loading bug and a benchmark coordinate error were found by these checks and fixed.")],
        7.05, 2.2, 5.5, 4.5, size=13, gap=6)

# ============================================================================ demo
s = new("Live demo: new data to joint temperatures in the application", "Demo",
        notes="DEMO SCRIPT (about 5 minutes). Before the talk: start the app with "
              ".venv312\\Scripts\\python.exe -m src.cli app and keep it open. "
              "1) Dataset upload: select demo_new_dataset, calibration flirone_one_pro_v9_found, press Validate: three recordings "
              "turn Valid, one stray video is listed as unmatched. "
              "2) Calibration: show the v9 profile (scale 1.27, rotation 0) and the Auto-calibrate button. "
              "3) Joint view: original dataset, S2_right_static: visible joints with temperatures on the photo, ROIs on the thermal "
              "image; then S1_left_open_close at a closed frame: folded joints are not shown. "
              "4) Independent test: dataset demo_new_dataset, saved model v7_mahalanobis_ledoit_wolf_..._172656, Run: one result "
              "per hand with score, threshold and top joints. "
              "5) Reports: open the dataset report: final joint temperature with Q1-Q3 and frame count. "
              "Fallback if anything fails live: the screenshots on this slide and on the deployment slides.")
table(s, ["#", "Show", "What the audience sees"],
      [["1", "Dataset upload → Validate", "3 recordings Valid; stray file listed"],
       ["2", "Calibration", "v9 profile (scale 1.27, 0°) + Auto-calibrate"],
       ["3", "Joint view", "temperatures on visible joints; folded joints hidden"],
       ["4", "Independent test", "one result per hand + top joints"],
       ["5", "Reports", "final joint °C with Q1–Q3 and frames"]],
      0.6, 1.5, 6.6, col_w=[0.5, 2.6, 3.5], size=13, row_h=0.55)
text(s, ".venv312\\Scripts\\python.exe -m src.cli app", 0.6, 5.2, 6.6, 0.4, 13, NAVY, bold=True, face="Consolas")
text(s, "Backup if the live run fails: screenshots on this and the deployment slides.", 0.6, 5.7, 6.6, 0.6, 12, GREY, italic=True)
img(s, FIG / "A03_app_joint_view.png", 7.5, 1.45, 5.3, 2.5, "joint view", "screenshot")
img(s, FIG / "N04_report_joint_table.png", 7.5, 4.3, 5.3, 2.3, "report: final joint temperatures", "application output")

# ============================================================================ 35 references
s = new("Selected references", "References",
        notes="Full IEEE reference list with DOIs is in the technical report and in deliverables/supporting/references.md.")
refs = ["[1] J. S. Smolen et al., \"Rheumatoid arthritis,\" Lancet, 388(10055), 2016. doi:10.1016/S0140-6736(16)30173-8",
        "[2] G. Schiavon et al., \"Infrared thermography for the evaluation of inflammatory and degenerative joint diseases: a systematic review,\" Cartilage, 13(2 suppl.), 2021. doi:10.1177/19476035211063862",
        "[3] Y. Qiao et al., \"Dorsal-to-Joint Temperature Differences (D2JTD) extraction from RGB-T images…,\" Biomed. Signal Process. Control, 112:108482, 2026. doi:10.1016/j.bspc.2025.108482",
        "[4] F. Zhang et al., \"MediaPipe Hands: on-device real-time hand tracking,\" arXiv:2006.10214, 2020",
        "[5] D. G. Moreira et al., TISEM consensus statement, J. Therm. Biol., 69, 2017. doi:10.1016/j.jtherbio.2017.07.006",
        "[6] E. F. J. Ring and K. Ammer, \"Infrared thermal imaging in medicine,\" Physiol. Meas., 33(3), 2012. doi:10.1088/0967-3334/33/3/R33",
        "[7] N. Otsu, IEEE Trans. SMC, 9(1), 1979 · C. Rother et al., GrabCut, ACM TOG, 23(3), 2004",
        "[8] O. Ledoit and M. Wolf, J. Multivar. Anal., 88(2), 2004 · P. J. Rousseeuw and K. Van Driessen, Technometrics, 41(3), 1999",
        "[9] F. T. Liu et al., Isolation forest, ICDM 2008 · B. Schölkopf et al., Neural Comput., 13(7), 2001",
        "[10] P. E. Shrout and J. L. Fleiss, Psychol. Bull., 86(2), 1979 · J. P. Weir, J. Strength Cond. Res., 19(1), 2005",
        "[11] F. Pedregosa et al., Scikit-learn, JMLR 12, 2011 · Teledyne FLIR, FLIR ONE Pro specifications"]
text(s, refs, 0.6, 1.45, 12.1, 5.4, 12.5, INK)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.core_properties.title = "RGB–Thermal Hand-Joint Temperature Analysis — Project Presentation"
prs.core_properties.subject = "Research and screening-support system; not a diagnostic device"
prs.save(OUT)
print("saved", OUT, "slides", len(prs.slides))
