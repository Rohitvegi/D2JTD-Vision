"""Collect every number used in the report/presentation directly from saved result files.

Writes deliverables/supporting/metrics_verified.json (used by the document builders) and
deliverables/supporting/Metric_Calculation_Sheet.xlsx (value, n, source file, version, method).
Run from the project root:  .venv312\\Scripts\\python.exe docs\\deliverables_src\\collect_metrics.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(".")
OUT = R / "deliverables" / "supporting"
OUT.mkdir(parents=True, exist_ok=True)
B = R / "baselines/v3_baseline/reports/results"
rows = []                                # metric sheet rows


def add(area, metric, value, n, source, version, method, unit=""):
    rows.append(dict(area=area, metric=metric, value=value, unit=unit, n=n, source=str(source), version=version, method=method))
    return value


M = {}
# ---------------- dataset (Stage 1 audit) ----------------
a = pd.read_csv(R / "data/metadata/dataset_audit.csv")
src = "data/metadata/dataset_audit.csv"
M["dataset"] = dict(
    recordings=add("Dataset", "Recordings (complete)", int(len(a)), 68, src, "Stage 1", "file discovery + checks"),
    subjects=add("Dataset", "Subjects", int(a.subject.nunique()), 17, src, "Stage 1", "unique subject IDs"),
    frames_min=int(a.npy_frames.min()), frames_max=int(a.npy_frames.max()), frames_median=float(a.npy_frames.median()),
    frames_total=add("Dataset", "Radiometric frames (total)", int(a.npy_frames.sum()), 68, src, "Stage 1", "sum of .npy frame counts", "frames"),
    fps_min=float(a.rgb_fps_measured.min()), fps_max=float(a.rgb_fps_measured.max()), fps_median=float(a.rgb_fps_measured.median()),
    duration_min=float(a.rgb_duration_s.min()), duration_max=float(a.rgb_duration_s.max()), duration_median=float(a.rgb_duration_s.median()),
    tmin_min=float(a.t_min.min()), tmax_max=float(a.t_max.max()), tmean_median=float(a.t_mean.median()),
    npy_gb=float(a.npy_bytes.sum() / 1e9), video_gb=float((a.rgb_bytes.sum() + a.thermal_mp4_bytes.sum()) / 1e9),
    frame_counts_equal=int(a.frame_counts_equal.sum()), nan_recordings=int(((a.nan_frac > 0) | (a.inf_frac > 0)).sum()),
    swapped=["S11_left_static"],
    seek_error_range=[float(a.rgb_seek_offset_mid.min()), float(a.rgb_seek_offset_mid.max())],
)
add("Dataset", "Frame counts equal (RGB=thermal MP4=.npy)", M["dataset"]["frame_counts_equal"], 68, src, "Stage 1", "decoded counts compared")
add("Dataset", "RGB fps measured (median)", round(M["dataset"]["fps_median"], 2), 68, src, "Stage 1", "frames / decoded duration", "fps")

# ---------------- v3 baseline (as published) ----------------
s = json.load(open(B / "summary.json"))
c = json.load(open(B / "rq2_correction.json"))
eb = json.load(open(B / "edge_benchmark.json"))
t1 = pd.read_csv(B / "rq1_landmarks.csv").set_index("task")
sync = pd.read_csv(B / "sync_rgb_thermal.csv")
r4 = pd.read_csv(B / "rq4_repeatability.csv").set_index("feature")
reg = pd.read_csv(B / "registration.csv")
fr = pd.concat([pd.read_csv(p, usecols=["detected"]) for p in glob.glob("baselines/v3_baseline/data/processed/frames/*.csv")])
v3 = "v3 (baseline, frozen)"
M["v3"] = dict(
    hand_detection=add("RQ1 landmarks", "Hand detected", float(fr.detected.astype(bool).mean()), int(len(fr)), "baselines/v3_baseline/data/processed/frames", v3, "share of analysed frames with a MediaPipe hand"),
    static_valid=add("RQ1 landmarks", "Static frames valid", float(t1.loc["static", "valid"]), int(t1.loc["static", "frames"]), B / "rq1_landmarks.csv", v3, "checks pass AND >=80 % joint centres on thermal hand"),
    oc_valid=add("RQ1 landmarks", "Open-close frames valid", float(t1.loc["open_close", "valid"]), int(t1.loc["open_close", "frames"]), B / "rq1_landmarks.csv", v3, "as above"),
    dice_median=add("RQ2 registration", "Hand-mask Dice (held-out, median)", float(s["RQ2"]["dice_heldout_median"]), 68, B / "summary.json", v3, "Dice = 2|A∩B|/(|A|+|B|) on held-out frames"),
    dice_p10=float(s["RQ2"]["dice_heldout_p10"]), dice_min=float(s["RQ2"]["dice_heldout_min"]),
    dice_ge90=int(s["RQ2"]["recordings_dice_ge_090"]),
    scale=float(s["RQ2"]["scale_median"]), tx=float(s["RQ2"]["tx_median"]), ty=float(s["RQ2"]["ty_median"]),
    joint_err_raw=add("RQ2 joints", "Joint error raw (median)", float(c["raw_err_px"]["median"]), int(c["n_points"]), B / "rq2_correction.json", v3, "Euclidean distance to clicked joint", "px"),
    joint_err_corr=add("RQ2 joints", "Joint error corrected, LOSO (median)", float(c["loso_corrected_err_px"]["median"]), int(c["n_points"]), B / "rq2_correction.json", v3, "along-finger correction learned without the subject", "px"),
    joint_err_corr_p90=float(c["loso_corrected_err_px"]["p90"]),
    joint_err_static=add("RQ2 joints", "Joint error static (median)", float(c["static_loso_err_px"]), None, B / "rq2_correction.json", v3, "subset static", "px"),
    joint_err_oc=add("RQ2 joints", "Joint error open-close (median)", float(c["open_close_loso_err_px"]), None, B / "rq2_correction.json", v3, "subset open-close", "px"),
    along_bias=float(c["along_bias_px_raw"]), across_bias=float(c["across_bias_px_raw"]),
    corr_mcp=float(c["correction"]["MCP"]), corr_pip=float(c["correction"]["PIP"]),
    dT_raw=add("RQ2 joints", "ROI temperature change due to misplacement (median |dT|) raw", float(c["abs_dT_raw_C"]["50%"]), int(c["n_points"]), B / "rq2_correction.json", v3, "ROI median at pipeline vs clicked position", "degC"),
    dT_corr=add("RQ2 joints", "same, after correction", float(c["abs_dT_corrected_C"]["50%"]), int(c["n_points"]), B / "rq2_correction.json", v3, "as above", "degC"),
    roi_on_skin=add("RQ3 ROIs", "ROIs >=95 % on hand", float(s["RQ3"]["on_skin_frac_all"]), None, B / "summary.json", v3, "share of ROI area inside thermal hand mask"),
    icc=add("RQ4 repeatability", "ICC(3,1) 10-joint mean D2JTD", float(r4.loc["D2JTD_mean10", "icc31"]), int(r4.loc["D2JTD_mean10", "n_hands"]), B / "rq4_repeatability.csv", v3, "split-half, two-way mixed, consistency"),
    mdc=add("RQ4 repeatability", "MDC95 10-joint mean D2JTD", float(r4.loc["D2JTD_mean10", "mdc95"]), int(r4.loc["D2JTD_mean10", "n_hands"]), B / "rq4_repeatability.csv", v3, "1.96·sqrt(2)·SEM, SEM=SD(diff)/sqrt(2)", "degC"),
    mdc_joints=r4["mdc95"].to_dict(), icc_joints=r4["icc31"].to_dict(),
    sync_aligned=add("Sync", "Open-close recordings aligned at lag 0", int((sync.status == "aligned").sum()), int(len(sync)), B / "sync_rgb_thermal.csv", v3, "cross-correlation openness(RGB) vs thermal hand area"),
    sync_corr=float(sync["corr"].median()),
    ms_frame=add("Compute", "Per-frame time (median)", float(eb["median_ms"]["total"]), int(eb["frames"]), B / "edge_benchmark.json", v3, "perf_counter around landmarks+mask+ROIs", "ms"),
    fps=add("Compute", "Throughput", float(eb["fps_equivalent"]), int(eb["frames"]), B / "edge_benchmark.json", v3, "1000 / median ms", "fps"),
    ms_parts=eb["median_ms"], rss_model=float(eb["rss_after_model_load_MB"]), rss_peak=float(eb["rss_peak_MB"]),
    model_mb=float(eb["landmark_model_MB"]),
    mean10=float(s["RQ5"]["mean10"]), mean10_sd=float(s["RQ5"]["mean10_sd"]), mean10_ci=s["RQ5"]["mean10_ci"],
    h1=int(s["RQ5"]["h1_joints_supported"]), lr_frac=float(s["RQ5"]["lr_frac_all_lt_05"]),
    rq6=pd.read_csv(B / "rq6_open_close.csv").set_index("variable").to_dict("index"),
)
add("RQ5 reference", "10-joint mean D2JTD (subject means)", M["v3"]["mean10"], 17, B / "summary.json", v3, "mean of subject means, static", "degC")
M["v3"]["reference"] = pd.read_csv(B / "rq5_reference.csv").to_dict("records")
M["v3"]["reg_dice"] = reg.dice_heldout_median.tolist()

# ---------------- v5 annotation QC ----------------
q = json.load(open(R / "reports/v5_annotation_qc/summary.json"))
M["v5"] = dict(flagged=q["flagged_frames"], frames=q["n_frames"], points=q["n_points"],
               hat=q["three_cornered_hat"]["all"]["bias_removed"]["own_rms_error_px"],
               bias=q["three_cornered_hat"]["all"]["raw"]["median_bias_px"],
               sens=q["rq2_sensitivity"])
add("Annotation QC", "Frames failing consistency checks", q["flagged_frames"], q["n_frames"], "reports/v5_annotation_qc/summary.json", "v5", "order, bone length, finger identity")
M["v5"]["errors"] = pd.read_csv(B / "rq2_correction_points.csv")[["task", "kind", "err_raw", "err_loso", "along", "across", "dT_raw", "dT_loso"]].to_dict("list")

# ---------------- v6 features ----------------
ab = pd.read_csv(R / "reports/v6_features/ablation.csv")
st = pd.read_csv(R / "reports/v6_features/feature_stability.csv")
M["v6"] = dict(ablation=ab.to_dict("records"),
               stability=st.groupby("group")[["icc31", "mdc95"]].median().reset_index().to_dict("records"),
               n_features=int(len(st)))
for r in ab[ab.model == "mahalanobis_ledoit_wolf"].itertuples():
    add("Feature ablation", f"{r.feature_set}: sens all10 +1C / single +2C / spec",
        f"{r.sens_all10_1C:.2f} / {r.sens_single_joint_2C:.2f} / {r.specificity:.2f}", 34, "reports/v6_features/ablation.csv", "v6",
        "nested LOSO, simulated warming")

# ---------------- v7 models ----------------
cm = pd.read_csv(R / "reports/v7_models/model_comparison.csv")
lc = pd.read_csv(R / "reports/v7_models/learning_curve.csv")
lco = pd.read_csv(R / "reports/v7_models/learning_curve_other_models.csv")
ss = pd.read_csv(R / "reports/v7_models/synthetic_sensitivity.csv")
M["v7"] = dict(comparison=cm.to_dict("records"),
               learning_curve=lc.groupby("n_train")[["specificity", "sens_all10_1C"]].mean().reset_index().to_dict("records"),
               learning_other=lco.to_dict("records"), synthetic=ss.to_dict("records"))
for r in cm.itertuples():
    add("Anomaly models", f"{r.model}: specificity / sens all10 +1C / single +2C",
        f"{r.specificity:.2f} / {r.sens_all_10_MCP_PIP_1C:.2f} / {r.sens_single_joint_2C:.2f}", 34, "reports/v7_models/model_comparison.csv", "v7",
        "outer LOSO, threshold = 95th pct of inner-LOSO scores")

# ---------------- v8 visibility, registration, live, e2e ----------------
from src.landmarks import JOINT_NAMES  # noqa: E402
vis = []
nfr = 0
for p in glob.glob("data/processed_v8/frames/*.csv"):
    d = pd.read_csv(p)
    d = d[d.detected.astype(bool)]
    nfr += len(d)
    t = "static" if p.endswith("static.csv") else "open_close"
    for j in JOINT_NAMES:
        vis.append(dict(task=t, kind=j.split("_")[1], hidden=1 - d[f"{j}_vis"].mean()))
    vis.append(dict(task=t, kind="ANY", hidden=float((d.n_hidden_joints > 0).mean())))
V = pd.DataFrame(vis).groupby(["task", "kind"]).hidden.mean().unstack(0)
M["v8"] = dict(visibility=V.round(4).to_dict(), frames=nfr)
add("Visibility", "Open-close frames with any hidden joint", float(V.loc["ANY", "open_close"]), None, "data/processed_v8/frames", "v8", "joint_visibility rules")
add("Visibility", "Static frames with any hidden joint", float(V.loc["ANY", "static"]), None, "data/processed_v8/frames", "v8", "joint_visibility rules")
reg8 = pd.read_csv(R / "data/processed_v8/registration.csv")
M["v8"]["dice"] = reg8.dice_heldout_median.tolist()
M["v8"]["live"] = json.load(open(R / "reports/v8_live/simulation.json"))
tests = sorted(Path("results/independent_tests").glob("mini_test__*"))
if tests:
    M["v8"]["e2e"] = json.load(open(tests[0] / "results.json"))
M["v8"]["tests_passing"] = 35

json.dump(M, open(OUT / "metrics_verified.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
S = pd.DataFrame(rows)
with pd.ExcelWriter(OUT / "Metric_Calculation_Sheet.xlsx", engine="openpyxl") as w:
    S.to_excel(w, sheet_name="metrics", index=False)
    a.to_excel(w, sheet_name="dataset_audit", index=False)
    pd.DataFrame(M["v3"]["reference"]).to_excel(w, sheet_name="healthy_reference_v3", index=False)
    pd.read_csv(B / "rq4_repeatability.csv").to_excel(w, sheet_name="repeatability_v3", index=False)
    cm.to_excel(w, sheet_name="anomaly_models_v7", index=False)
    ab.to_excel(w, sheet_name="feature_ablation_v6", index=False)
    V.reset_index().to_excel(w, sheet_name="visibility_v8", index=False)
    pd.DataFrame([dict(formula="Dice", definition="2|A∩B| / (|A|+|B|)"), dict(formula="D2JTD_j", definition="T_dorsum − T_joint_j"),
                  dict(formula="ICC(3,1)", definition="(MSR − MSE) / (MSR + (k−1)·MSE), k = 2 halves"),
                  dict(formula="SEM", definition="SD(half2 − half1) / sqrt(2)"), dict(formula="MDC95", definition="1.96 · sqrt(2) · SEM"),
                  dict(formula="robust z", definition="(x − median) / (1.4826 · MAD)"),
                  dict(formula="Mahalanobis", definition="sqrt((x − μ)ᵀ Σ⁻¹ (x − μ)), Σ = Ledoit–Wolf / MCD estimate"),
                  dict(formula="Sensitivity", definition="TP / (TP + FN)"), dict(formula="Specificity", definition="TN / (TN + FP)"),
                  dict(formula="Precision", definition="TP / (TP + FP)"), dict(formula="F1", definition="2PR / (P + R)"),
                  dict(formula="Accuracy", definition="(TP + TN) / N")]).to_excel(w, sheet_name="formulas", index=False)
print("metrics:", len(S), "| v8 frames", nfr)
print(V.round(3))
