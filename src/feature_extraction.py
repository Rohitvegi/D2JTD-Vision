"""v6 — Joint-wise feature extraction (per recording) and feature-group ablation.

Feature groups (computed from the per-frame tables of src.run_pipeline):
  absolute  : joint median / mean / P90 / spread (SD), dorsum and whole-hand temperature   [deg C]
  relative  : D2JTD (dorsum - joint), joint - whole hand, within-hand z-score
  spatial   : MCP-PIP and PIP-DIP per finger, neighbouring-knuckle differences,
              along-finger gradient, palm-to-finger difference
  temporal  : trend of the 10-joint D2JTD and joint mean (deg C/min), frame-to-frame variation, SD
  quality   : valid frames, joint centres on hand, openness, contaminated ROIs, registration Dice
              (quality features gate measurements; they are never used as anomaly evidence)

Ablation (pre-declared before running): every feature set is evaluated with the same
leave-one-subject-out procedure and nested thresholds as v7, with SIMULATED warming injected into
the per-frame joint temperatures before features are computed. A richer set replaces D2JTD-only only
if its selection score is at least 0.02 higher AND its specificity is >= 0.90; otherwise the
simpler set is kept (parsimony protects unseen-data performance). Stability of every feature is
measured as split-half ICC and MDC95 on the static recordings.

Usage:  .venv312\\Scripts\\python.exe -m src.feature_extraction --processed-dir data/processed_v6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.landmarks import JOINT_NAMES, QIAO_JOINTS
from src.rois import COLD_FLAG_FRAC

OUT = Path("reports/v6_features")
OPEN_THR = 1.88                     # v3 open-hand threshold (5th pct of static openness)
FINGERS = {"thumb": ["thumb_MCP", "thumb_IP"], "index": ["index_MCP", "index_PIP", "index_DIP"],
           "middle": ["middle_MCP", "middle_PIP", "middle_DIP"], "ring": ["ring_MCP", "ring_PIP", "ring_DIP"],
           "little": ["little_MCP", "little_PIP", "little_DIP"]}
MCPS = ["index_MCP", "middle_MCP", "ring_MCP", "little_MCP"]
DISTAL = [j for j in JOINT_NAMES if j.endswith(("PIP", "DIP", "_IP"))]


def _slope(t, y):
    m = np.isfinite(y)
    return float(stats.theilslopes(y[m], t[m])[0] * 60) if m.sum() >= 10 else np.nan


def usable_frames(g: pd.DataFrame, task: str) -> pd.DataFrame:
    g = g[g["frame_valid"].fillna(False).astype(bool)]
    if task == "open_close" and "openness" in g:
        g = g[g["openness"] >= OPEN_THR]
    return g.sort_values("t_s")


def recording_features(g: pd.DataFrame, task: str, dice: float = np.nan) -> dict:
    """All feature groups for one recording from its per-frame table."""
    n_all = len(g)
    g = usable_frames(g, task)
    f = {}
    T = {j: g[f"{j}_T"].to_numpy(float) for j in JOINT_NAMES}
    dors = g["dorsum_T"].to_numpy(float)
    hand = g["hand_T"].to_numpy(float) if "hand_T" in g else np.full(len(g), np.nan)
    med = lambda a: float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan  # noqa: E731
    J = np.column_stack([T[j] for j in JOINT_NAMES]) if len(g) else np.empty((0, 14))
    jmean = np.nanmean(J, axis=1) if len(g) else np.array([])
    jsd = np.nanstd(J, axis=1) if len(g) else np.array([])
    for j in JOINT_NAMES:
        f[f"{j}_Tmed"] = med(T[j])                    # FINAL joint temperature: median over usable frames
        ok = np.isfinite(T[j])
        f[f"{j}_Tq1"] = float(np.percentile(T[j][ok], 25)) if ok.any() else np.nan   # spread over frames (report only,
        f[f"{j}_Tq3"] = float(np.percentile(T[j][ok], 75)) if ok.any() else np.nan   # not model inputs)
        f[f"{j}_nframes"] = int(ok.sum())
        f[f"{j}_Tmean"] = med(g[f"{j}_mean"].to_numpy(float)) if f"{j}_mean" in g else np.nan
        f[f"{j}_Tp90"] = med(g[f"{j}_p90"].to_numpy(float)) if f"{j}_p90" in g else np.nan
        f[f"{j}_spread"] = med(g[f"{j}_sd"].to_numpy(float))
        f[f"{j}_D2JTD"] = med(dors - T[j])
        f[f"{j}_rel_hand"] = med(T[j] - hand)
        with np.errstate(invalid="ignore", divide="ignore"):
            f[f"{j}_z_hand"] = med((T[j] - jmean) / np.where(jsd > 0.05, jsd, np.nan))
    f["dorsum_T"] = med(dors)
    f["hand_T"] = med(hand)
    for fin, js in FINGERS.items():
        f[f"{fin}_MCP_minus_next"] = med(T[js[0]] - T[js[1]])
        if len(js) == 3:
            f[f"{fin}_PIP_minus_DIP"] = med(T[js[1]] - T[js[2]])
        pos = np.arange(len(js))
        with np.errstate(invalid="ignore"):
            slopes = [np.polyfit(pos, row, 1)[0] for row in np.column_stack([T[j] for j in js]) if np.isfinite(row).all()]
        f[f"{fin}_gradient"] = float(np.median(slopes)) if slopes else np.nan
    for a, b in zip(MCPS[:-1], MCPS[1:]):
        f[f"{a}_minus_{b}"] = med(T[a] - T[b])
    distal = np.nanmean(np.column_stack([T[j] for j in DISTAL]), axis=1) if len(g) else np.array([])
    f["palm_to_finger"] = med(dors - distal)
    t = g["t_s"].to_numpy(float)
    d10 = np.nanmean(np.column_stack([dors - T[j] for j in QIAO_JOINTS]), axis=1) if len(g) else np.array([])
    f["D2JTD10_trend"] = _slope(t, d10) if len(g) else np.nan
    f["jointmean_trend"] = _slope(t, jmean) if len(g) else np.nan
    f["D2JTD10_frame_var"] = float(np.nanmedian(np.abs(np.diff(d10)))) if len(d10) > 2 else np.nan
    f["D2JTD10_temporal_sd"] = float(np.nanstd(d10)) if len(d10) > 2 else np.nan
    # quality (gates, not anomaly evidence)
    f["q_frames_used"] = int(len(g))
    f["q_valid_frac"] = float(len(g) / max(n_all, 1))
    f["q_centres_on_hand"] = med(g["joint_centres_on_hand"].to_numpy(float)) if len(g) else np.nan
    f["q_openness"] = med(g["openness"].to_numpy(float)) if len(g) else np.nan
    f["q_roi_usable"] = float(np.isfinite(J).mean()) if len(g) else 0.0
    cold = [c for c in (f"{j}_cold" for j in JOINT_NAMES) if c in g]
    f["q_contaminated_rois"] = int(sum(np.nanmedian(g[c]) > COLD_FLAG_FRAC for c in cold)) if cold and len(g) else 0
    f["q_registration_dice"] = dice
    return f


GROUPS = {
    "relative_D2JTD": [f"{j}_D2JTD" for j in JOINT_NAMES],
    "relative_hand": [f"{j}_rel_hand" for j in JOINT_NAMES] + [f"{j}_z_hand" for j in JOINT_NAMES],
    "absolute": [f"{j}_{s}" for j in JOINT_NAMES for s in ("Tmed", "Tmean", "Tp90", "spread")] + ["dorsum_T", "hand_T"],
    "spatial": ([f"{fin}_MCP_minus_next" for fin in FINGERS] + [f"{fin}_PIP_minus_DIP" for fin in FINGERS if fin != "thumb"]
                + [f"{fin}_gradient" for fin in FINGERS] + [f"{a}_minus_{b}" for a, b in zip(MCPS[:-1], MCPS[1:])]
                + ["palm_to_finger"]),
    "temporal": ["D2JTD10_trend", "jointmean_trend", "D2JTD10_frame_var", "D2JTD10_temporal_sd"],
}
QUALITY = ["q_frames_used", "q_valid_frac", "q_centres_on_hand", "q_openness", "q_roi_usable",
           "q_contaminated_rois", "q_registration_dice"]
FEATURE_SETS = {
    "S1_D2JTD (v3 baseline)": GROUPS["relative_D2JTD"],
    "S2_D2JTD+relative_hand": GROUPS["relative_D2JTD"] + GROUPS["relative_hand"],
    "S3_D2JTD+spatial": GROUPS["relative_D2JTD"] + GROUPS["spatial"],
    "S4_D2JTD+temporal": GROUPS["relative_D2JTD"] + GROUPS["temporal"],
    "S5_D2JTD+absolute": GROUPS["relative_D2JTD"] + GROUPS["absolute"],
    "S6_all": sum(GROUPS.values(), []),
}


def load_frames(frames_dir: Path) -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_csv(p) for p in sorted(Path(frames_dir).glob("*.csv"))}


def parse_rec(rec_id: str):
    s, h, *t = rec_id.split("_")
    return s, h, "_".join(t)


def extract_dataset(frames: dict[str, pd.DataFrame], registration: pd.DataFrame | None = None) -> pd.DataFrame:
    dice = dict(zip(registration.rec_id, registration.dice_heldout_median)) if registration is not None else {}
    rows = []
    for rec, g in frames.items():
        s, h, task = parse_rec(rec)
        rows.append(dict(rec_id=rec, subject=s, hand=h, task=task, **recording_features(g, task, dice.get(rec, np.nan))))
    return pd.DataFrame(rows)


def warm_frames(g: pd.DataFrame, joints: list[str] | None, d: float) -> pd.DataFrame:
    """Simulated warming of whole joint ROIs by d degC (None = diffuse: every joint, dorsum and hand)."""
    g = g.copy()
    js = JOINT_NAMES if joints is None else joints
    for j in js:
        for s in ("_T", "_mean", "_p90"):
            if j + s in g:
                g[j + s] = g[j + s] + d
    if joints is None:
        for c in ("dorsum_T", "hand_T"):
            if c in g:
                g[c] = g[c] + d
    return g


PATTERNS = {"single_joint_2C": ("single", 2.0), "all10_1C": (QIAO_JOINTS, 1.0), "all10_0.5C": (QIAO_JOINTS, 0.5),
            "MCP_1C": ([j for j in JOINT_NAMES if j.endswith("MCP")], 1.0),
            "PIP_1C": ([j for j in JOINT_NAMES if j.endswith(("PIP", "_IP"))], 1.0),
            "DIP_1C": ([j for j in JOINT_NAMES if j.endswith("DIP")], 1.0), "diffuse_2C": (None, 2.0)}


def warmed_features(frames, static_recs):
    """Features of every static recording under every simulated pattern (computed once)."""
    out = {}
    for rec in static_recs:
        g = frames[rec]
        out[(rec, "none", "")] = recording_features(g, "static")
        for name, (js, d) in PATTERNS.items():
            if js == "single":
                for j in JOINT_NAMES:
                    out[(rec, name, j)] = recording_features(warm_frames(g, [j], d), "static")
            else:
                out[(rec, name, "")] = recording_features(warm_frames(g, js, d), "static")
    return out


def ablation(F: pd.DataFrame, W: dict, models=("mahalanobis_ledoit_wolf", "robust_z_max")) -> pd.DataFrame:
    from src.anomaly_models import Pipeline, nested_threshold
    st = F[F.task == "static"].reset_index(drop=True)
    groups = st.subject.to_numpy()
    rows = []
    for set_name, cols in FEATURE_SETS.items():
        X = st[cols].to_numpy(float)
        keep = np.isfinite(X).mean(0) > 0.8                   # drop features missing in >20 % of hands
        X = X[:, keep]
        cols_k = [c for c, k in zip(cols, keep) if k]
        for model in models:
            res = {k: [] for k in ["none"] + list(PATTERNS)}
            for s in np.unique(groups):
                tr = groups != s
                thr = nested_threshold(X[tr], groups[tr], model)
                p = Pipeline(model).fit(X[tr])
                for rec in st.rec_id[~tr]:
                    vec = lambda key: np.array([[W[key].get(c, np.nan) for c in cols_k]])  # noqa: E731
                    res["none"].append(p.score(vec((rec, "none", "")))[0] <= thr)
                    for name, (js, _) in PATTERNS.items():
                        keys = [(rec, name, j) for j in JOINT_NAMES] if js == "single" else [(rec, name, "")]
                        res[name] += [p.score(vec(k))[0] > thr for k in keys]
            r = dict(feature_set=set_name, model=model, n_features=len(cols_k), specificity=float(np.mean(res["none"])))
            r.update({f"sens_{k}": float(np.mean(v)) for k, v in res.items() if k != "none"})
            r["selection_score"] = float(np.mean([r["specificity"], r["sens_all10_1C"], r["sens_single_joint_2C"]]))
            rows.append(r)
            print(f"{set_name:26s} {model:24s} n={len(cols_k):3d} spec {r['specificity']:.2f} "
                  f"all10+1 {r['sens_all10_1C']:.2f} single+2 {r['sens_single_joint_2C']:.2f} diffuse {r['sens_diffuse_2C']:.2f}")
    return pd.DataFrame(rows)


def stability(frames: dict, cols: list[str]) -> pd.DataFrame:
    """Split-half ICC(3,1) and MDC95 per feature on static recordings."""
    from src.analysis import icc31
    h1, h2 = [], []
    for rec, g in frames.items():
        if not rec.endswith("static"):
            continue
        g = usable_frames(g, "static")
        mid = g.t_s.median()
        h1.append(recording_features(g[g.t_s <= mid], "static"))
        h2.append(recording_features(g[g.t_s > mid], "static"))
    A, B = pd.DataFrame(h1), pd.DataFrame(h2)
    rows = []
    for c in cols:
        a, b = A[c].to_numpy(float), B[c].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        d = b[m] - a[m]
        rows.append(dict(feature=c, icc31=icc31(a[m], b[m]) if m.sum() >= 5 else np.nan,
                         mdc95=float(1.96 * d.std(ddof=1)) if m.sum() >= 5 else np.nan, n=int(m.sum())))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed_v6")
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    proc = Path(args.processed_dir)
    frames = load_frames(proc / "frames")
    reg = pd.read_csv(proc / "registration.csv") if (proc / "registration.csv").is_file() else None
    F = extract_dataset(frames, reg)
    F.to_csv(OUT / "features.csv", index=False)
    all_cols = sum(GROUPS.values(), [])
    S = stability(frames, all_cols)
    S["group"] = S.feature.map({c: g for g, cs in GROUPS.items() for c in cs})
    S.to_csv(OUT / "feature_stability.csv", index=False)
    summary = dict(n_recordings=len(F), groups={g: len(c) for g, c in GROUPS.items()},
                   stability_by_group=S.groupby("group")[["icc31", "mdc95"]].median().round(3).reset_index().to_dict("records"))
    if not args.no_ablation:
        static = [r for r in frames if r.endswith("static")]
        W = warmed_features(frames, static)
        A = ablation(F, W)
        A.to_csv(OUT / "ablation.csv", index=False)
        base = A[A.feature_set.str.startswith("S1")].set_index("model")
        decision = []
        for model, g in A.groupby("model"):
            b = base.loc[model, "selection_score"]
            better = g[(g.selection_score >= b + 0.02) & (g.specificity >= 0.90)].sort_values("selection_score", ascending=False)
            decision.append(dict(model=model, baseline_score=b,
                                 chosen=better.iloc[0].feature_set if len(better) else "S1_D2JTD (v3 baseline)",
                                 chosen_score=float(better.iloc[0].selection_score) if len(better) else b))
        summary["ablation_decision"] = decision
        summary["decision_rule"] = "replace D2JTD-only only if selection score >= baseline + 0.02 and specificity >= 0.90"
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2, default=float)
    print(json.dumps(summary, indent=1, default=float)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
