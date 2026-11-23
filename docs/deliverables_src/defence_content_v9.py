# v9 / release 1.0.0 updates to defence_content (executed at the end of defence_content.py, same namespace).
# Adds: automatic calibration (src/calibration.py), one final temperature per joint with its spread, deployment.
import pandas as _pd

BEN = json.load(open("reports/v9_calibration/benchmark_find_calibration.json"))["results"]
B3, B9 = BEN["v3: fitted per recording"], BEN["found calibration (this module)"]
SELF = _pd.read_csv("reports/v9_calibration/selftest_known_distortions.csv")
PROD = json.load(open("models/calibration/flirone_one_pro_v9_found.json"))
SELF_MED = float(SELF.mapping_error_px_median.median())
_fr = _pd.read_csv("data/processed_v8/frames/S2_right_static.csv")
AGG = _fr[_fr.frame_valid.fillna(False).astype(bool)]           # usable frames of the worked example


def _set(lst, key, new, pos=0):
    for i, r in enumerate(lst):
        if r[pos] == key:
            lst[i] = new
            return
    raise KeyError(key)


def _after(lst, key, new, pos=0):
    for i, r in enumerate(lst):
        if r[pos] == key:
            lst.insert(i + 1, new)
            return
    raise KeyError(key)


# ---------------------------------------------------------------- status
_set(STATUS, "RGB→thermal registration", ("RGB→thermal registration (v3)", "src/registration.py", "one shift per recording (kept as profile v3)",
                                          "both masks, 16 frames", "s, tx, ty per recording + Dice", "Completed (superseded by v9 as default)"))
_after(STATUS, "RGB→thermal registration (v3)", ("Automatic calibration (v9)", "src/calibration.py",
       "find the RGB→thermal mapping from the recordings, apply and verify it", "paired recordings",
       "profile: image area, scale, rotation, per-frame shift + grade", "Completed (12/12 known distortions; default)"))
_set(STATUS, "Camera (stereo) calibration", ("Target-based stereo calibration", "—", "exact 3-D mapping with a calibration target",
                                             "checkerboard recording", "—", "Missing (no target recording; v9 calibrates from the hand)"))
_after(STATUS, "Joint ROIs + temperatures", ("Final joint temperature per recording", "src/feature_extraction.py, report_generator.py",
       "one value per joint from many frames", "per-frame table", "median + Q1–Q3 + frames used", "Completed"))
_set(STATUS, "Edge deployment", ("Desktop deployment (release 1.0.0)", "install.ps1, src/doctor.py, src/cli.py, tools/build_release.py",
                                 "install and run on another PC", "release zip", "installed app + health check", "Completed (clean-install test passed)"))
STATUS.append(("Edge-device deployment", "src/edge_demo.py", "on-device use", "frames", "timing", "Partial (laptop CPU only)"))

# ---------------------------------------------------------------- workflow
_set(WORKFLOW, "6. RGB→thermal alignment", (
    "6. RGB→thermal alignment (v9 automatic calibration)",
    "The calibration found from the recordings is applied per frame: p_th = s·Rot(α)(k·p − c) + c + t(frame).",
    "The two cameras see the hand from different positions and with different lenses.",
    "Joint search of scale and rotation scored by the distance between hand OUTLINES (capped at 10 px); shift learned per frame from the hand",
    "OpenCV, NumPy, scikit-learn (Huber)", "paired recordings / each frame's landmarks", "profile; transform per frame",
    "Joints would land beside the finger; a new camera could not be used.",
    f"12/12 known distortions recovered ({SELF_MED:.1f} px); unseen subjects {B9['joint_err_median']:.2f} px vs {B3['joint_err_median']:.2f} px (v3)."))
_after(WORKFLOW, "9. Temperature sampling", (
    "9b. One temperature per joint",
    "The joint's per-frame temperatures (usable frames only) are reduced to their median; Q1–Q3 and the number of frames are kept.",
    "The report and the model need one value per joint per recording; frames of one hand are not independent.",
    "Median over frames (robust to a minority of bad frames)", "pandas, NumPy", "per-frame table", "final °C per joint + spread",
    "A single bad frame, or the moment chosen, would decide the result.",
    f"Repeatability of the final value: MDC95 {f(V3['mdc'], 2)} °C (two halves of a recording)."))

# ---------------------------------------------------------------- robustness
_set(ROBUST, "Incorrect alignment", ("Incorrect alignment", "v9 automatic calibration + held-out self-check grade; validation probe (Dice)",
                                     f"joint error {B9['joint_err_median']:.2f} px median still above 8 px", "target-based stereo calibration"))
_set(ROBUST, "Different resolutions", ("Different resolutions", "Auto-calibrate finds image area and scale of a new camera; profile check",
                                       "aspect must match (rotated exports are refused)", "rotate/transpose option for exports"))
_set(ROBUST, "Camera angle", ("Camera angle", "rotation between the sensors is estimated (−10° … +10°)", "large viewpoint changes untested",
                              "record from several angles"))
ROBUST.append(("Installation on another PC", "installer with exact versions, health check, model checksums", "Windows only",
               "Linux/macOS install scripts"))

# ---------------------------------------------------------------- critical review
_set(REVIEW, "Joint localisation fails the 8 px target", (
    "Joint localisation still above the 8 px target", "High",
    f"v9 reduced the error on unseen subjects to {B9['joint_err_median']:.2f} px (v3 {B3['joint_err_median']:.2f}), still above 8 px.",
    "DIP joints least repeatable; temperature cost small (≈0.05 °C).", "Depth-aware mapping; thermal-domain joint refinement.",
    "Present as improved but not met."))
_set(REVIEW, "No geometric calibration", (
    "No target-based calibration", "Medium",
    "v9 finds scale, rotation and shift from the hand (proven on 12 known distortions) but has no 3-D/stereo model.",
    f"Parallax of fingers at other depths remains (open–close {B9['open_close']:.1f} px).", "Heated checkerboard; stereo calibration.",
    "Future work."))
_set(REVIEW, "Laptop-only timing", ("Laptop-only timing", "Low", "Deployment targets Windows desktops; no edge board measured.",
                                    "Real-time claim limited to laptop CPU.", "Benchmark on target device.", "Future work."))


# ---------------------------------------------------------------- slides
def _slide_index(title):
    return next(i for i, s_ in enumerate(SLIDES) if s_["t"] == title)


SLIDES[_slide_index("RGB–thermal alignment")] = dict(
    t="RGB–thermal alignment: automatic calibration (v9)", k="Calibration and alignment", layout="split",
    fig="N12_calibration_apply.png", cap="joints found in RGB mapped into the thermal image",
    b=["p_th = s·Rot(α)(k·p_rgb − c) + c + t(frame)", "Found from the recordings: image area, scale, rotation, per-frame shift",
       "Scored by distance between hand OUTLINES (capped 10 px)",
       f"Default: scale {PROD['scale']:.2f}, rotation {PROD['angle_deg']:.0f}°, Dice {PROD['quality']['dice_heldout_median']:.3f}",
       "No target-based stereo calibration (no target recorded)"],
    s="The two lenses sit side by side and see the hand slightly differently, so a joint found in the RGB image is not at the "
      "same pixel in the thermal image. My calibration module finds the mapping from the recordings themselves: the picture "
      "area, the scale, a possible rotation and a small shift per frame learned from the hand. It scores a candidate by how "
      "close the RGB hand outline lands to the thermal hand outline, with distances capped, so an extra warm forearm cannot "
      "pull the scale. Then it checks itself on frames it did not use and stores the result as a calibration profile.",
    q=[("Why not a homography?", "A homography has 8 parameters and would overfit the hand outline; scale, rotation and shift are "
        "what the rigid side-by-side rig needs. None of them models depth — that needs a stereo calibration with a target."),
       ("How do you know the calibration is correct?", f"I distorted real recordings with known transforms and the module recovered "
        f"all 12 (scale within 0.01, angle within 0.75°, {SELF_MED:.1f} px median); and on 12 unseen subjects joints land closer to "
        f"the clicked positions ({B9['joint_err_median']:.2f} vs {B3['joint_err_median']:.2f} px).")])

SLIDES.insert(_slide_index("RGB–thermal alignment: automatic calibration (v9)") + 1, dict(
    t="Calibration proof: known distortions and unseen people", k="Calibration and alignment", layout="split",
    fig="N03_calibration_benchmark.png", cap="calibrated on S1–S5, tested on S6–S17",
    b=["Known distortions (scale ±8–10 %, rotation ±4–6°, shift, moved picture): 12/12 recovered",
       f"Mapping error vs truth: {SELF_MED:.1f} px median",
       f"Unseen subjects: {B9['joint_err_median']:.2f} px vs {B3['joint_err_median']:.2f} px (v3)",
       f"Hand-outline overlap: Dice {B9['dice_median']:.3f} vs {B3['dice_median']:.3f}"],
    s="I tested the calibration in two ways. First, I took real recordings, distorted the thermal side with transformations I "
      "chose, and checked that the module finds exactly those transformations without being told; it did in all 12 cases. "
      "Second, I calibrated using only five subjects and applied it unchanged to the twelve others: the joints landed closer to "
      "the positions I had clicked than with fitting every recording separately.",
    q=[("Is 9.97 px good enough?", "It is better than before and the temperature effect of joint error is small (about 0.05 °C), but "
        "it is still above my 8 px target; the remaining error comes from depth (bent fingers) and where MediaPipe places joints.")]))

SLIDES.insert(_slide_index("Features: D2JTD and spatial contrast"), dict(
    t="Hundreds of frames → one temperature per joint", k="Feature extraction", layout="split",
    fig="N01_frame_aggregation.png", cap="S2_right_static: 267 frames → final value per joint",
    b=["Per frame: median of the ROI's skin pixels", "Per recording: median over usable frames (hidden joints dropped)",
       "Reported with Q1–Q3 and number of frames", f"Reliable to {f(V3['mdc'], 2)} °C (MDC95)",
       "The model scores one vector per hand, not frames"],
    s="Every frame gives a slightly different temperature for each joint, mostly because the whole hand slowly drifts during "
      "the recording. For the report and the model I need one number per joint, so I take the median of the joint's values over "
      "the usable frames. The median is the typical value and is not moved by a few bad frames. The report also shows the "
      "middle half of the frames, Q1 to Q3, and how many frames were used, so the reader can see how stable the value was.",
    q=[("Why not the mean or the last frame?", "The mean is pulled by single bad frames and the last frame depends on one moment; "
        "the median represents the typical value even if many frames are bad."),
       ("Why not classify every frame?", "Frames of one hand are not independent; 267 frames are not 267 samples. Scoring per hand "
        "avoids inflated results and leakage.")]))

SLIDES.insert(_slide_index("Robustness and efficiency") + 1, dict(
    t="Deployment: install and use the system on another PC", k="Deployment", layout="split",
    fig="N08_app_dataset_upload.png", cap="deployed application with a new dataset",
    b=["Release zip 1.0.0 → install.ps1 (exact package versions) → health check READY", "Start: RGB-Thermal App.bat",
       "Upload → Validate → (Auto-calibrate) → Process → Independent test → Joint view → Reports",
       "Command line: rgbthermal calibrate | test | doctor", "Clean install verified: full workflow + tests passed"],
    s="The project is packaged as release 1.0.0. On another Windows PC you unzip it and run one install command, which creates "
      "the Python environment with the exact package versions and runs a health check that must say READY. Then you start the "
      "application, import a dataset, validate it, auto-calibrate if the camera is new, process, and run an independent test "
      "with the saved model; the joint view and the PDF report show the result. I verified this on a clean installation.",
    q=[("How do you know the installation works?", "The health check tests Python, the exact package versions, model files by "
        "SHA-256, the hand model, video decoding, the GUI and write access; a clean install also passed the full workflow and tests."),
       ("Can it run without internet?", "Yes: the installer has an offline mode using a local package cache, and the program "
        "itself needs no internet.")]))

_conc = SLIDES[_slide_index("Conclusion")]
_conc["b"] = ["Audited RGB–thermal pipeline from raw files to per-joint °C",
              f"One temperature per joint (median of frames); MDC95 {f(V3['mdc'], 2)} °C",
              f"Automatic calibration: 12/12 distortions; {B9['joint_err_median']:.2f} px on unseen subjects",
              f"Leakage-free model: specificity {f(LW['specificity'])}; {PR['real_correct']}/{PR['n_hands']} healthy correct",
              "Deployed as release 1.0.0; next: patients, stereo calibration, repeat sessions"]
_res = SLIDES[_slide_index("Results against the criteria")]
_res["tab"][1][2] = ["RQ2b", f"joint error {f(V3['joint_err_corr'], 1)} px (v3) → {B9['joint_err_median']:.2f} px (v9, unseen)",
                     "Fail (improved)"]

# ---------------------------------------------------------------- viva
for _i, (_cat, _q, _a) in enumerate(VIVA):
    if _q == "Did you calibrate the cameras?":
        VIVA[_i] = (_cat, _q, "Not with a target; the dataset has none. The v9 module calibrates from the recordings themselves: it finds "
                    "the image area, scale, rotation and a per-frame shift by matching the RGB and thermal hand outlines, then checks "
                    f"itself on held-out frames. Default profile: scale {PROD['scale']:.2f}, rotation {PROD['angle_deg']:.0f}°, Dice "
                    f"{PROD['quality']['dice_heldout_median']:.3f}.")
    elif _q == "How do you calculate alignment error?":
        VIVA[_i] = (_cat, _q, "Three ways: Dice of the warped RGB hand mask with the thermal mask on held-out frames; the distance between "
                    "the two hand outlines (the calibration grade); and Euclidean error of mapped joints against clicked joints "
                    f"({B9['joint_err_median']:.2f} px on unseen subjects with v9).")
    elif _q == "Explain the coordinate transformation with an example.":
        VIVA[_i] = (_cat, _q, "p_th = s·Rot(α)(k·p − c) + c + t. With α = 0, the RGB point (180, 240) becomes (240, 320) after k = 4/3, "
                    "which is the centre c, so scaling and rotation do nothing and only the shift moves it; e.g. t = (−12.3, −31.5) "
                    "gives (227.7, 288.5). Points far from the centre move more because of s.")

VIVA += [
    ("Frames to one value", "Each frame gives a different temperature; what is the final temperature of a joint?",
     "The median of that joint's per-frame temperatures over the usable frames (frames failing checks, or where the joint is hidden, "
     "are dropped). Each per-frame value is itself the median of the ROI's skin pixels."),
    ("Frames to one value", "Why the median and not the mean?",
     "The mean is pulled by a few bad frames (motion, a misplaced landmark, a cold ring pixel); the median is the typical value and "
     "stays put even if up to half the frames are bad."),
    ("Frames to one value", "What do Q1–Q3 and 'frames' in the report mean?",
     "Q1–Q3 is the range of the middle half of the per-frame values, showing how stable the joint was during the recording; "
     "'frames' is how many frames contributed to the final value."),
    ("Frames to one value", "Why do temperatures change across frames?",
     "Mostly a slow drift of the whole hand during the recording; the back of the hand drifts with the joints, so D2JTD is more "
     "stable than the raw temperature."),
    ("Frames to one value", "How reliable is the final value?",
     f"Two halves of the same recording agree within {f(V3['mdc'], 2)} °C (MDC95, 34 hands); smaller differences should not be "
     "interpreted."),
    ("Calibration", "How does the automatic calibration work?",
     "It finds the image area, searches scale and rotation jointly, scoring each candidate by the mean distance from the RGB hand "
     "outline to the thermal hand outline (capped at 10 px), refines the best candidates, learns a per-frame shift from the hand's "
     "size and position, and grades itself on frames it did not use."),
    ("Calibration", "Why outline distance and not area overlap?",
     "In one recording the thermal mask included the warm forearm; area overlap then rewarded a too-large scale (1.63). The outline "
     "distance is set by the finger contours, and capping ignores the forearm."),
    ("Calibration", "How did you prove it finds the correct transformation?",
     f"I distorted real recordings with known scale, rotation, shift and image position, and the module recovered all 12 cases "
     f"(scale within 0.01, angle within 0.75°, {SELF_MED:.1f} px median error)."),
    ("Calibration", "Does it work on other cameras?",
     "It is designed to: it finds image area, scale and rotation from the data and refuses rotated exports. It was validated on the "
     "FLIR ONE Pro and on synthetic geometric changes, not yet on a second physical camera."),
    ("Deployment", "How is the system deployed?",
     "As release 1.0.0: unzip, run install.ps1 (Python 3.12 with exact package versions), check that the health check says READY, "
     "then start RGB-Thermal App.bat. A clean installation passed the full workflow and the tests."),
    ("Deployment", "How does a user apply the deployed model to new data?",
     "Upload and validate the dataset, auto-calibrate if the camera is new, process it, run an independent test with the saved model, "
     "then read the joint view and the PDF/Excel report."),
    ("Deployment", "How do you make sure the deployed model is the validated one?",
     "Model and calibration files are checked against SHA-256 checksums by the health check, and every independent test stores the "
     "model's fingerprint in its lock file."),
]
