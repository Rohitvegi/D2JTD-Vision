"""Complete project report = the technical report + development history, formula compendium, results compendium,
folded-joint handling and open items. Reuses build_report.py (same chapters 1-13) and inserts chapters 14-18.

Run from the project root:
  .venv312\\Scripts\\python.exe docs/deliverables_src/build_complete_report.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import doc_helpers  # noqa: E402

doc_helpers.EQ_DIR = Path("deliverables/figures/equations_complete")      # keep the other documents' equation images

NEW_CHAPTERS = r'''
LW = C["mahalanobis_ledoit_wolf"]
# ============================================================================ 14 history
doc.h("14. Development History: Problems, Observations and Changes")
doc.p("The architecture was not designed in one step. Each version answered a problem observed in the previous one, "
      "measured against a reference (files, held-out frames, hand-clicked joints, held-out people). This chapter lists "
      "those steps in order, so that every module can be traced back to the problem that created it.")
hist = [
    ["Stage 1 audit", "Radiometric array assumed transposed", "overlay of .npy on the video had the wrong shape", ".npy is upright portrait; resize into the video picture area", "correct geometry"],
    ["Stage 1 audit", "Video letterboxed", "non-black area of the frames is x 140–500", "crop the picture, scale coordinates by k = 4/3", "coordinates on the .npy grid"],
    ["Stage 1 audit", "RGB and thermal frames did not match", "OpenCV seeking returned frames up to ±30 away", "decode frames in order, never seek", "frame i ↔ frame i verified"],
    ["Stage 1 audit", "S11_left_static files swapped", "thermal video vs .npy correlation r = −0.58 (others ≈ 1)", "swap list in config + automatic swap detection", "68/68 usable"],
    ["Stage 1 audit", "68 GB of temperature data", "does not fit in RAM", "memory-mapped .npy, one recording per worker", f"peak memory {V3['rss_peak']:.0f} MB"],
    ["Stage 3 objectives", "Only healthy volunteers", "no labels for a classifier", "measurement study with RQ1–RQ7 fixed in advance", "honest pass/fail"],
    ["v1 pipeline", "Cold fingers lost from thermal mask", "S17's fingers near room temperature; Otsu cut them", "hysteresis growth from the Otsu seed", "fingers kept"],
    ["v1 pipeline", "Registration unstable", "free scale fits scattered", "scale fixed at 1.25 (median of 68 free fits) + shift per recording + rig fallback", f"Dice {f(V3['dice_median'], 3)}"],
    ["v1 → v2", "RQ3 failed: ROIs off the fingers", "48 % of ROIs ≥ 95 % on hand with Qiao's radius", "radius capped at 0.9 × distance to finger edge", f"{pct(V3['roi_on_skin'], 0)} on skin"],
    ["v2", "Reruns differed", "GrabCut uses random numbers", "fixed OpenCV seed, versioned output folders", "bit-identical reruns"],
    ["Annotation", "RQ2b failed", f"{f(V3['joint_err_raw'], 1)} px; error along finger {f(V3['along_bias'], 1)} px vs across {f(V3['across_bias'], 1)} px", "v3: learned along-finger correction (LOSO)", f"{f(V3['joint_err_corr'], 1)} px; cost {f(V3['dT_corr'], 3)} °C"],
    ["v4 audit", "Own numbers re-checked", "open–close validity 99.57 %, not 99.7 %", "erratum; v3 frozen with checksums", "baseline protected"],
    ["v5", "'Is my annotation wrong?'", "133/136 frames consistent; 2 mistakes; clicks ≈17.6 px scatter", "annotation QC + three-cornered hat", f"{f(V5['sens']['without_flagged_frames']['corrected_loso_median_px'], 2)} px without flagged frames"],
    ["v7", "v3 thresholds chosen on the same hands", "sensitivity 0.71–1.00 was optimistic", "nested leave-one-subject-out for model and threshold", "specificity 0.94; honest sensitivity"],
    ["v7", "Best-scoring model unstable", "MCD specificity 0.59–0.99 with ≤ 12 subjects", "Ledoit–Wolf as default", "stable at every size"],
    ["v6", "Which features help?", "122 features worse than 32 (overfitting); temporal ICC 0.40", "D2JTD + spatial (32 features)", f"1 joint +2 °C: {f(AB[('S3_D2JTD+spatial', 'mahalanobis_ledoit_wolf')]['sens_single_joint_2C'])}"],
    ["v8", "Use without code", "—", "upload, validation, locked tests, reports, GUI", "end-to-end software test"],
    ["v8", "'Show joints with temperatures on the photo'", "—", "joint view", "visual check per frame"],
    ["v8", "'Folded joints still show a temperature'", "MediaPipe guesses hidden joints", "visibility rule; hidden joints get no temperature", "open–close: 66 % frames with a hidden joint"],
    ["v9", "'Find the calibration and apply it'", "shift almost independent of hand size (r = 0.09)", "learned per-frame shift", "same accuracy without per-video fitting"],
    ["v9", "Calibration by area overlap", "warm forearm pulled the scale to 1.63 (S7)", "outline distance, capped at 10 px", "11/12 known distortions"],
    ["v9", "Scale searched at angle 0", "combined distortion settled on a wrong scale", "joint scale × angle search, 3 refined starts", "12/12; 1.4 px median"],
    ["v9", "Image box 137 instead of 140", "compression noise at the edge of single frames", "median box over frames", "box found exactly"],
    ["v9", "Unseen people", "—", "calibrate on S1–S5, test on S6–S17", f"{B_V9['joint_err_median']:.2f} vs {B_V3['joint_err_median']:.2f} px"],
    ["v9", "'One temperature per joint?'", "per-frame values drift", "median of frames + Q1–Q3 + frames in reports", f"MDC95 {f(V3['mdc'], 2)} °C"],
    ["Release 1.0.0", "Install on another PC", "clean install found a model-loading bug (classes saved as __main__)", "robust loader + test; installer, health check, CLI", "acceptance test passed"],
]
doc.table(["Step", "Problem", "Observation / evidence", "Change", "Effect"], hist, "Development history (in order).",
          widths=[2.1, 3.4, 4.3, 4.2, 2.4], size=7.5)
doc.p("How the architecture emerged: data defects were fixed first, so every later number rests on correct files and "
      "frame pairs; RGB is used to locate joints and the radiometric data to measure them, joined by calibration; every "
      "failure produced one targeted module (hysteresis, width cap, joint correction, nested validation, visibility rule, "
      "outline-based calibration); every change was kept only if it improved a measured reference.")

# ============================================================================ 15 formulas
doc.h("15. Formula Compendium")
doc.p("Every formula used by the software, with its variables and the values used or obtained in this project.")
doc.h("15.1 Geometry and calibration", 2)
doc.eq(r"\mathbf{p}_{crop} = \mathbf{p}_{video} - (x_0, y_0),\qquad k = \frac{W_{npy}}{W_{crop}} = \frac{480}{360} = \frac{4}{3}",
       "Crop the letterbox (x0 = 140, y0 = 0) and scale RGB pixels to the radiometric grid.")
doc.eq(r"\mathbf{p}_{th} = s\,\mathrm{Rot}(\alpha)\,(k\,\mathbf{p}_{crop} - \mathbf{c}) + \mathbf{c} + \mathbf{t}",
       f"Calibrated transform; c = (240, 320). Default profile: s = {PROD['scale']:.2f}, α = {PROD['angle_deg']:.0f}°, t predicted per frame. "
       "v3 used s = 1.25, α = 0 and one t per recording.")
doc.eq(r"\mathbf{t} = \mathbf{a} + B\,[h,\ c_x,\ c_y]^{\top},\qquad h = \|\mathbf{p}_0 - \mathbf{p}_9\|",
       "Per-frame shift model: h = hand size (wrist to middle knuckle, px), (cx, cy) = landmark centre; a and B by Huber "
       "regression (ε = 1.5), which limits the influence of frames where the masks disagree.")
doc.eq(r"d(M) = \frac{1}{N}\sum_{i=1}^{N}\min\left(D_{th}(M\mathbf{q}_i),\ 10\right)",
       f"Calibration score (px): q_i = N points on the RGB hand outline, D_th = distance to the thermal outline. Grade: good ≤ 5, "
       f"acceptable ≤ 8. Default profile: {PROD['quality']['outline_px_heldout_median']:.1f} px on held-out frames.")
doc.eq(r"\mathrm{Dice}(A,B) = \frac{2|A\cap B|}{|A|+|B|},\qquad \mathrm{IoU} = \frac{\mathrm{Dice}}{2-\mathrm{Dice}}",
       f"Mask overlap: v3 held-out median {f(V3['dice_median'], 3)}; v9 on unseen subjects {B_V9['dice_median']:.3f}.")
doc.h("15.2 Segmentation", 2)
doc.eq(r"\tau^{*} = \arg\max_{\tau}\ \omega_0(\tau)\,\omega_1(\tau)\,[\mu_0(\tau)-\mu_1(\tau)]^2",
       "Otsu threshold on the blurred frame rescaled between its 1st and 99.5th percentile; ω = class fractions, μ = class means.")
doc.eq(r"\tau_{low} = T_{bg} + 0.4\,(\tau_{Otsu} - T_{bg})",
       "Hysteresis: T_bg = median outside the Otsu seed; pixels above τ_low connected to the seed are added (keeps cold fingers).")
doc.h("15.3 Landmarks and visibility", 2)
doc.eq(r"O = \frac{1}{4}\sum_{f}\frac{\|\mathbf{p}_0 - \mathbf{p}_{tip,f}\|}{\|\mathbf{p}_0 - \mathbf{p}_{MCP,f}\|}",
       "Hand openness over the four fingers: about 1.9 or more for an open hand, 1.0–1.3 for a fist; open–close statistics use frames with O ≥ 1.88.")
doc.eq(r"\theta = \arccos\frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|},\quad \mathbf{u} = \mathbf{p}_{PIP}-\mathbf{p}_{MCP},\ \mathbf{v} = \mathbf{p}_{DIP}-\mathbf{p}_{PIP}",
       "Middle-joint bend (with MediaPipe depth z as third coordinate); the end joint (DIP) is hidden when θ > 100°.")
doc.eq(r"\mathbf{q}_i = \mathbf{c} + 0.92\,(\mathbf{p}_i - \mathbf{c}),\quad i \in \{0, 1, 2, 5, 9, 13, 17\}",
       "Back-of-hand polygon shrunk by 8 %; a PIP/DIP inside it is folded under the hand.")
doc.eq(r"t = \mathrm{clip}\left(\frac{(\mathbf{p}-\mathbf{a})\cdot(\mathbf{b}-\mathbf{a})}{\|\mathbf{b}-\mathbf{a}\|^2}, 0, 1\right),\quad \|\mathbf{a}+t(\mathbf{b}-\mathbf{a})-\mathbf{p}\| < 0.07h,\ \ z_{bone} < z_{p} - 0.03h",
       "Covered joint: another finger's bone (a→b) passes within 0.07 hand sizes and is closer to the camera by 0.03 hand sizes.")
doc.h("15.4 Joint position", 2)
doc.eq(r"e = \sqrt{(x_p-x_g)^2+(y_p-y_g)^2},\quad e_{\parallel} = (\mathbf{p}-\mathbf{g})\cdot\hat{\mathbf{u}},\quad e_{\perp} = (\mathbf{p}-\mathbf{g})\cdot\hat{\mathbf{n}}",
       f"Error vs a clicked joint g; split along (u, finger direction) and across (n). Raw: median {f(V3['joint_err_raw'], 2)} px, along {f(V3['along_bias'], 1)}, across {f(V3['across_bias'], 1)} px.")
doc.eq(r"\mathbf{j}' = \mathbf{j} + \alpha_k\,L\,\hat{\mathbf{u}},\qquad L = \|\mathbf{p}_{PIP} - \mathbf{p}_{MCP}\|",
       f"Along-finger correction. v3: α = {f(V3['corr_mcp'], 3)} (MCP), {f(V3['corr_pip'], 3)} (PIP/DIP/IP) → {f(V3['joint_err_corr'], 2)} px; "
       f"re-learned for the v9 mapping: α = {PROD['along_frac']['MCP']:.3f} / {PROD['along_frac']['PIP']:.3f}.")
doc.eq(r"\sigma^2_A = \frac{1}{2}\left(\sigma^2_{AR} + \sigma^2_{AT} - \sigma^2_{RT}\right)",
       f"Three-cornered hat: own error of estimator A from the pairwise difference variances (clicks A, RGB-mapped R, thermal "
       f"MediaPipe T). Result: clicks {f(V5['hat']['annotation'], 1)}, RGB-mapped {f(V5['hat']['rgb_mapped'], 1)}, thermal {f(V5['hat']['thermal_mediapipe'], 1)} px RMS.")
doc.h("15.5 ROIs and temperature", 2)
doc.eq(r"r_j = \max\left(6,\ \min\left(\beta_f\,L_f,\ 0.9\,d_{edge}\right)\right)",
       "ROI radius (px): β = 0.42 thumb, 0.28 index, 0.28 middle, 0.31 ring, 0.35 little; d_edge = distance to the finger edge. "
       "Used if ≥ 60 % of the disc is on the hand and ≥ 32 core pixels remain.")
doc.eq(r"T_{j,f} = \mathrm{median}\{T(x,y) : (x,y) \in R_j \cap H_{core}\}",
       "Temperature of joint j in frame f (°C) from the radiometric matrix; H_core = hand mask eroded by about one sensor pixel.")
doc.eq(r"T_j = \mathrm{median}_{f \in F_j}\,T_{j,f},\quad Q1_j = P_{25},\ Q3_j = P_{75},\ n_j = |F_j|",
       f"FINAL joint temperature: median over the usable frames F_j (hidden or invalid frames excluded). Example S2_right_static, "
       f"index MCP: {f(AGG['index_MCP_T'].median())} °C, Q1–Q3 {f(AGG['index_MCP_T'].quantile(.25))}–{f(AGG['index_MCP_T'].quantile(.75))}, n = {AGG['index_MCP_T'].notna().sum()}.")
doc.eq(r"c_j = \frac{|\{T \in R_j : T < \mathrm{med}_j - 1.5\}|}{|R_j|},\qquad \mathrm{flag\ if}\ c_j > 0.15",
       "Cold-band share (metal rings read cold); warm pixels are never removed.")
doc.h("15.6 Features", 2)
doc.eq(r"\mathrm{D2JTD}_j = \mathrm{median}_f\left(T_{dorsum,f} - T_{j,f}\right)",
       f"Main feature; healthy 10-joint mean {f(V3['mean10'], 2)} °C (95 % CI {f(V3['mean10_ci'][0], 2)}–{f(V3['mean10_ci'][1], 2)}).")
doc.eq(r"\Delta_{MCP-PIP} = \mathrm{median}_f(T_{MCP}-T_{PIP}),\quad g_f = \arg\min_b \sum_i (T_i - a - b\,i)^2",
       "Spatial features: differences between joints of one finger and neighbouring knuckles, and the temperature gradient g along a finger.")
doc.eq(r"\mathrm{trend} = 60\cdot\mathrm{median}_{i<j}\frac{y_j - y_i}{t_j - t_i}",
       "Theil–Sen slope (°C per minute) of the 10-joint D2JTD over time (temporal group; not used by the model, ICC 0.40).")
doc.h("15.7 Healthy-reference models", 2)
doc.eq(r"z_i = \frac{x_i - \mathrm{med}(x_i)}{\max(1.4826\,\mathrm{MAD}_i,\ 0.05)},\qquad s = \max_i |z_i|",
       "Robust z model (baseline).")
doc.eq(r"D_M(\mathbf{x}) = \sqrt{(\mathbf{x}-\mu)^{\top}\hat\Sigma^{-1}(\mathbf{x}-\mu)},\qquad \hat\Sigma = (1-\delta)S + \delta\frac{\mathrm{tr}(S)}{p}I",
       "Default model: Mahalanobis distance with Ledoit–Wolf shrinkage (p = 32 features, 34 hands); δ is computed analytically.")
doc.eq(r"\theta = Q_{0.95}\left(\{D_M(\mathbf{x}_k)\}_{k\ \mathrm{held\ out}}\right)",
       "Threshold: 95th percentile of the scores of training hands, each scored by a model fitted without its own subject.")
doc.eq(r"x_i < \min_i - 0.25R_i\ \ \mathrm{or}\ \ x_i > \max_i + 0.25R_i \ \Rightarrow\ \mathrm{outside\ model\ experience}",
       "Range check (R_i = training range of feature i); such hands are reported, not scored.")
doc.eq(r"\zeta_i = \frac{x_i - \mu_i}{\sqrt{\hat\Sigma_{ii}}}",
       "Per-feature explanation (SD units): the joints with the largest |ζ| are reported as the cause of a flag.")
doc.h("15.8 Evaluation", 2)
doc.eq(r"\mathrm{Sens} = \frac{TP}{TP+FN},\ \ \mathrm{Spec} = \frac{TN}{TN+FP},\ \ \mathrm{Prec} = \frac{TP}{TP+FP},\ \ F_1 = \frac{2PR}{P+R},\ \ \mathrm{Acc} = \frac{TP+TN}{N}",
       f"Specificity on unseen healthy hands {f(LW['specificity'])}; accuracy/F1/AUC require labels with both classes.")
doc.eq(r"\mathrm{ICC}(3,1) = \frac{MS_R - MS_E}{MS_R + (k-1)MS_E},\qquad \mathrm{MDC}_{95} = 1.96\,\mathrm{SD}(x_2 - x_1)",
       f"Repeatability between the two halves of a static recording (k = 2, 34 hands): ICC {f(V3['icc'], 3)}, MDC95 {f(V3['mdc'], 2)} °C.")
doc.eq(r"\mathrm{CI}_{95} = [\,P_{2.5},\ P_{97.5}\,]\ \mathrm{of\ the\ statistic\ over\ 2000\ subject\ resamples}",
       "Bootstrap over subjects (not frames), so correlated frames do not shrink the interval.")
doc.eq(r"L^{*} = \arg\max_{L}\ \mathrm{corr}\left(O_{f},\ A_{f+L}\right)",
       f"Synchronisation check: openness O from RGB vs thermal hand area A; {V3['sync_aligned']}/34 recordings best at L = 0.")
doc.eq(r"\mathrm{fps} = \frac{1000}{t_{frame}\,[\mathrm{ms}]}",
       f"Throughput: {f(V3['ms_frame'], 1)} ms per frame ≈ {V3['fps']:.0f} fps on the laptop CPU.")

# ============================================================================ 16 results
doc.h("16. Results Compendium")
doc.p("All measured results in one place (simulated results are marked).")
doc.table(["Area", "Measure", "Result", "n"],
          [["Detection", "hand found", pct(V3["hand_detection"], 2), "68 rec."],
           ["Detection", "valid frames static / open–close", f"{pct(V3['static_valid'], 0)} / {pct(V3['oc_valid'], 2)}", "68 rec."],
           ["Sync", "aligned at lag 0", f"{V3['sync_aligned']}/34 (median r {f(V3['sync_corr'])})", "34 rec."],
           ["Registration v3", "held-out Dice median / P10 / min", f"{f(V3['dice_median'], 3)} / {f(V3['dice_p10'], 3)} / {f(V3['dice_min'], 3)}", "68 rec."],
           ["Calibration v9", "known distortions recovered", f"12/12, {SELF.mapping_error_px_median.median():.1f} px median", "12 cases"],
           ["Calibration v9", "unseen subjects joint error / Dice", f"{B_V9['joint_err_median']:.2f} px / {B_V9['dice_median']:.3f} (v3 {B_V3['joint_err_median']:.2f} / {B_V3['dice_median']:.3f})", f"{B_V9['n_joints']} joints"],
           ["Joint position v3", "raw → corrected (LOSO)", f"{f(V3['joint_err_raw'], 1)} → {f(V3['joint_err_corr'], 2)} px (P90 {f(V3['joint_err_corr_p90'], 1)})", "777 pts"],
           ["Joint position v3", "static / open–close", f"{f(V3['joint_err_static'], 2)} / {f(V3['joint_err_oc'], 2)} px", "777 pts"],
           ["Joint position", "temperature cost", f"{f(V3['dT_corr'], 3)} °C median", "777 pts"],
           ["ROIs", "≥ 95 % on hand", pct(V3["roi_on_skin"], 1), "all ROIs"],
           ["Repeatability", "10-joint D2JTD ICC / MDC95", f"{f(V3['icc'], 3)} / {f(V3['mdc'], 2)} °C", "34 hands"],
           ["Reference", "mean D2JTD (95 % CI)", f"{f(V3['mean10'], 2)} °C ({f(V3['mean10_ci'][0], 2)}–{f(V3['mean10_ci'][1], 2)})", "17 subj."],
           ["Visibility", "frames with a hidden joint static / open–close", f"{pct(V8['visibility']['static']['ANY'], 0)} / {pct(V8['visibility']['open_close']['ANY'], 0)}", f"{V8['frames']:,} frames"],
           ["Model", "specificity on unseen healthy hands", f"{f(LW['specificity'])} (2 of 34 flagged)", "34 hands"],
           ["Model (simulated)", "index MCP +2 °C flagged and located", f"{PR['warmed_correct']}/{PR['n_hands']}", "34 hands"],
           ["Model", "healthy hands correctly normal", f"{PR['real_correct']}/{PR['n_hands']}", "34 hands"],
           ["Compute", "time per frame / peak memory", f"{f(V3['ms_frame'], 1)} ms / {V3['rss_peak']:.0f} MB", "laptop CPU"],
           ["Software", "automated tests; clean-install acceptance", "50 passed; passed", "—"]],
          "Summary of all results.", widths=[3.0, 5.0, 6.4, 2.0], size=8.5)
rq6 = V3["rq6"]
doc.table(["Measure", "Static", "Open–close", "Difference", "p (subjects)"],
          [[k.replace("_", " "), f(v["static_median"], 3), f(v["open_close_median"], 3), f(v["diff_median"], 3), f(v["p_subjects"], 3)]
           for k, v in rq6.items()],
          "Static vs open–close per hand (median over 34 hands; Wilcoxon).", widths=[4.4, 2.6, 2.6, 2.6, 2.6], size=8.5)
lc = V7["learning_curve"]
doc.table(["Training subjects", "Specificity", "Sensitivity all 10 +1 °C (simulated)"],
          [[r["n_train"], f(r["specificity"]), f(r["sens_all10_1C"])] for r in lc],
          "Learning curve of the selected-by-rule model (MCD): unstable below about 12 subjects.", widths=[4.0, 4.0, 6.0], size=9)
doc.table(["Feature group", "Features", "ICC(3,1)", "MDC95 (°C)"],
          [[g.replace("_", " "), {"relative_D2JTD": 14, "relative_hand": 28, "spatial": 18, "absolute": 58, "temporal": 4}[g], f(r["icc31"]), f(r["mdc95"])]
           for g, r in ST.items()],
          "Stability of each feature group (34 static hands).", widths=[5.0, 2.5, 3.0, 3.0], size=9)
doc.table(["Pattern (simulated)", "+0.5 °C", "+1 °C", "+2 °C"],
          [[p_, f(LW[f"sens_{p_}_0.5C"]), f(LW[f"sens_{p_}_1C"]), f(LW[f"sens_{p_}_2C"])]
           for p_ in ("single_joint", "MCP_only", "PIP_only", "DIP_only", "all_10_MCP_PIP", "diffuse_whole_hand")],
          "Ledoit–Wolf (D2JTD) sensitivity to simulated warming by pattern; whole-hand warming is invisible to D2JTD by design.",
          widths=[6.0, 2.5, 2.5, 2.5], size=9)

# ============================================================================ 17 folded joints
doc.h("17. Folded Joints and How the Model Was Trained")
doc.p("MediaPipe always returns all 21 points, also for joints it cannot see (it guesses them from the hand shape). The "
      "visibility rule is applied after detection; it decides which detections are trusted for temperature.")
doc.numbered(["Detection: MediaPipe returns 21 points (x, y, depth z) for every frame.",
              "Quality checks reject implausible hands (finger order, hand size, points outside the image).",
              "Visibility rule (Section 15.3): folded under the hand, covered by a nearer finger, or end joint tucked under (bend > 100°).",
              "Hidden joints keep their position (the skeleton and ROI sizes need it) but their temperature is set to missing (NaN), never estimated.",
              "The final joint value is the median over the frames where the joint was visible; the report shows how many frames that was.",
              "Training used the 34 static (open-hand) recordings, in which no joint was hidden (0 % of frames); open–close recordings were not used for training and are reported as 'Not applicable' when scored."])
doc.table(["Situation in new data", "What the system does"],
          [["joint hidden in some frames", "uses its visible frames; frame count reported"],
           ["joint missing for the whole recording", "the model imputes that feature with the training median"],
           ["< 50 % of joint ROIs usable", "Invalid measurement (not scored)"],
           ["< 80 % usable or registration Dice < 0.7", "Low-confidence measurement"],
           ["< 10 usable frames", "Insufficient data"]],
          "Handling of hidden and missing joints.", widths=[6.4, 10.0], size=9)
vis = V8["visibility"]["open_close"]
doc.p(f"Observed in open–close recordings: any hidden joint {pct(vis['ANY'], 0)} of frames; DIP {pct(vis['DIP'], 0)}, thumb IP "
      f"{pct(vis['IP'], 0)}, PIP {pct(vis['PIP'], 0)}, MCP {pct(vis['MCP'], 0)}. Static recordings: 0 %. The rule's thresholds were set "
      "by inspecting the data and have not been checked against hand-labelled visibility.")

# ============================================================================ 18 open items
doc.h("18. Open Items")
doc.table(["Item", "Effect", "Next step"],
          [["No arthritis patients", "detection of disease not measurable; sensitivity only simulated", "labelled patient cohort with reference diagnosis"],
           ["17 subjects", "wide confidence intervals", "≥ 50 subjects"],
           ["Joint error above 8 px", f"{B_V9['joint_err_median']:.2f} px with v9; DIP joints least repeatable", "depth-aware mapping; target-based stereo calibration"],
           ["One session per subject", "day-to-day repeatability unknown", "repeat sessions"],
           ["One annotator", "reference error only bounded", "second annotator"],
           ["Visibility and ring rules", "not checked against labels / real rings", "label visibility; record hands with rings"],
           ["Laptop timing only", "edge feasibility unknown", "benchmark on the target device"]],
          "Open items.", widths=[4.0, 6.4, 6.0], size=9)

'''


def main():
    src_path = HERE / "build_report.py"
    s = src_path.read_text(encoding="utf-8")
    s = s.replace('OUT = Path("deliverables/RGB_Thermal_Hand_Arthritis_Technical_Research_Report.docx")',
                  'OUT = Path("deliverables/RGB_Thermal_Hand_Complete_Project_Report.docx")')
    s = s.replace('    "Technical Research Report",', '    "Complete Project Report — methods, formulas, results and development history",')
    s = s.replace("doc = Doc()\n", "doc = Doc()\ndoc.d.sections[0].footer.paragraphs[0].runs[0].text = "
                  "\"RGB–Thermal Hand-Joint Temperature Analysis — Complete Project Report · page \"\n", 1)
    anchor = "# ============================================================================ 12 conclusion"
    assert anchor in s and 'doc.h("14. Conclusion")' in s
    s = s.replace(anchor, NEW_CHAPTERS + anchor).replace('doc.h("14. Conclusion")', 'doc.h("19. Conclusion")')
    g = {"__file__": str(src_path), "__name__": "__complete__"}
    exec(compile(s, str(src_path), "exec"), g)
    print("complete report:", g["OUT"])


if __name__ == "__main__":
    main()
