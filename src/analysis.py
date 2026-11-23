"""Stages 9-11: features, reliability, healthy reference ranges, open-close response, one-class test.

Reads data/processed/frames/*.csv and registration.csv (from src.run_pipeline) and writes
reports/results/: tables (*.csv), figures (*.png), summary.json and results.md.

Pass criteria are the ones fixed in docs/stage3_learning_objective.md before the data were analysed.

Usage:  .venv312\\Scripts\\python.exe -m src.analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

from src.io_utils import load_config  # noqa: E402
from src.landmarks import JOINT_NAMES, QIAO_JOINTS  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"            # categorical slots 1-2 (reference palette)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEED = 42
QIAO_HEALTHY = (1.5, 0.4)                      # healthy D2JTD mean, SD (Qiao et al. 2025)
QIAO_RA_GAP = 0.4                              # healthy 1.5 vs RA 1.1 deg C


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
        "legend.frameon": False,
    })


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_frames(proc: Path) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in sorted((proc / "frames").glob("*.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    parts = df["rec_id"].str.extract(r"^(S\d+)_(left|right)_(static|open_close)$")
    parts.columns = ["subject", "hand", "task"]
    d2 = pd.DataFrame({f"{j}_D2JTD": df["dorsum_T"] - df[f"{j}_T"] for j in JOINT_NAMES})
    df = pd.concat([df, parts, d2], axis=1)
    for c in ("detected", "ok", "frame_valid"):
        df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    return df.copy()


def open_threshold(df: pd.DataFrame) -> float:
    """Openness below which a frame counts as a (partial) fist: 5th percentile of static frames."""
    s = df.loc[(df.task == "static") & df.frame_valid, "openness"].dropna()
    return float(np.percentile(s, 5))


# --------------------------------------------------------------------------- #
# RQ1-RQ3: geometry and quality
# --------------------------------------------------------------------------- #
def rq1(df):
    g = df.groupby("task").agg(frames=("frame", "size"), detected=("detected", "mean"),
                               passed_checks=("ok", "mean"), valid=("frame_valid", "mean"))
    re = df[df.detected].groupby("task")[["reordered_index_little", "reordered_middle_ring",
                                          "mcp_collapse", "out_of_image"]].mean()
    per_rec = df.groupby(["rec_id", "task"])["frame_valid"].mean().rename("valid_frac").reset_index()
    t = g.join(re).reset_index()
    static_valid = float(g.loc["static", "valid"])
    return t, per_rec, dict(static_valid_frac=static_valid, pass_=static_valid >= 0.90,
                            worst_static_recording=per_rec[per_rec.task == "static"].nsmallest(1, "valid_frac").to_dict("records"))


def rq2(reg: pd.DataFrame, df: pd.DataFrame, ann_path: Path, native: int):
    d = reg["dice_heldout_median"]
    out = dict(dice_heldout_median=float(d.median()), dice_heldout_p10=float(d.quantile(0.1)),
               dice_heldout_min=float(d.min()), recordings_dice_ge_090=int((d >= 0.90).sum()),
               n_recordings=int(len(d)), dice_pass=bool(d.median() >= 0.90),
               scale_median=float(reg.scale.median()), scale_iqr=[float(reg.scale.quantile(.25)), float(reg.scale.quantile(.75))],
               tx_median=float(reg.tx.median()), ty_median=float(reg.ty.median()),
               tx_iqr=[float(reg.tx.quantile(.25)), float(reg.tx.quantile(.75))],
               ty_iqr=[float(reg.ty.quantile(.25)), float(reg.ty.quantile(.75))])
    err = None
    if ann_path.is_file():
        a = pd.read_csv(ann_path)
        lm_of = {n: i for n, i in zip(JOINT_NAMES, [2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19])}
        rows = []
        for r in a.itertuples():
            f = df[(df.rec_id == r.rec_id) & (df.frame == r.frame)]
            if f.empty or r.joint not in lm_of:
                continue
            j = lm_of[r.joint]
            px, py = f.iloc[0][f"lm{j}_x"], f.iloc[0][f"lm{j}_y"]
            rows.append(dict(rec_id=r.rec_id, frame=r.frame, joint=r.joint, err_px=float(np.hypot(px - r.x, py - r.y))))
        err = pd.DataFrame(rows)
        if len(err):
            planned = 2 * 6 * reg.shape[0]                          # 2 frames x 6 joints x 68 recordings
            enough = len(err) >= 0.8 * planned
            out.update(landmark_err_median_px=float(err.err_px.median()),
                       landmark_err_median_native=float(err.err_px.median() / native),
                       landmark_err_p90_px=float(err.err_px.quantile(0.9)), n_annotated_points=int(len(err)),
                       n_annotated_frames=int(err[["rec_id", "frame"]].drop_duplicates().shape[0]),
                       landmark_pass=bool(err.err_px.median() <= 8) if enough else None,
                       landmark_note=None if enough else
                       f"partial annotation ({len(err)} of {planned} planned points): no verdict yet")
    if err is None or not len(err):
        out.update(landmark_err_median_px=None, landmark_pass=None,
                   landmark_note="pending: run src.annotate to click joints on the reserved frames")
    corr_path = Path("reports/results/rq2_correction.json")
    if corr_path.is_file() and out.get("landmark_err_median_px") is not None:
        c = json.load(open(corr_path))
        out.update(landmark_err_raw_median_px=out["landmark_err_median_px"],
                   landmark_err_corrected_loso_median_px=c["loso_corrected_err_px"]["median"],
                   landmark_err_corrected_loso_p90_px=c["loso_corrected_err_px"]["p90"],
                   landmark_err_corrected_static_px=c["static_loso_err_px"],
                   landmark_err_corrected_open_close_px=c["open_close_loso_err_px"],
                   roi_temp_error_raw_median_C=c["abs_dT_raw_C"]["50%"],
                   roi_temp_error_corrected_median_C=c["abs_dT_corrected_C"]["50%"],
                   roi_temp_error_corrected_p90_C=c["abs_dT_corrected_C"]["90%"],
                   landmark_correction=c["correction"])
        if out.get("landmark_pass") is not None:
            out["landmark_pass"] = bool(c["loso_corrected_err_px"]["median"] <= 8)
            out["landmark_note"] = ("verdict uses the leave-one-subject-out corrected error "
                                    "(the pipeline applies the along-finger correction)")
    return out, err


def rq3(df, on_skin, native):
    v = df[df.frame_valid]
    rows = []
    for j in JOINT_NAMES:
        inn = v[f"{j}_in"]
        rows.append(dict(joint=j, on_skin_frac=float((inn >= on_skin).mean()), usable_frac=float(v[f"{j}_T"].notna().mean()),
                         median_in=float(inn.median()), median_core_px=float(v[f"{j}_npx"].median()),
                         median_native_px=float(v[f"{j}_npx"].median() / native ** 2)))
    t = pd.DataFrame(rows)
    allin = np.concatenate([v[f"{j}_in"].values for j in JOINT_NAMES])
    return t, dict(on_skin_frac_all=float(np.mean(allin >= on_skin)), pass_=bool(np.mean(allin >= on_skin) >= 0.95),
                   usable_frac_all=float(t.usable_frac.mean()), median_native_px=float(t.median_native_px.median()))


# --------------------------------------------------------------------------- #
# Stage 9: features per recording
# --------------------------------------------------------------------------- #
def slope_per_min(t, y):
    m = np.isfinite(y)
    if m.sum() < 10:
        return np.nan
    return float(stats.theilslopes(y[m], t[m])[0] * 60)


def features(df, open_thr):
    rows = []
    for rec, g in df.groupby("rec_id"):
        g = g[g.frame_valid]
        if rec.endswith("open_close"):                 # compare open-hand frames only
            g = g[g.openness >= open_thr]
        g = g.sort_values("t_s")
        r = dict(rec_id=rec, subject=g.subject.iloc[0] if len(g) else rec.split("_")[0],
                 hand=rec.split("_")[1], task="static" if rec.endswith("static") else "open_close",
                 n_frames_used=len(g), dorsum_T=g.dorsum_T.median())
        t = g.t_s.values
        r["dorsum_slope"] = slope_per_min(t, g.dorsum_T.values)
        for j in JOINT_NAMES:
            r[f"{j}_T"] = g[f"{j}_T"].median()
            r[f"{j}_D2JTD"] = g[f"{j}_D2JTD"].median()
            r[f"{j}_HDI"] = g[f"{j}_sd"].median()
        jt = g[[f"{j}_T" for j in JOINT_NAMES]].mean(axis=1).values
        dj = g[[f"{j}_D2JTD" for j in QIAO_JOINTS]].mean(axis=1).values
        r["jointmean_T"] = float(np.nanmedian(jt)) if len(jt) else np.nan
        r["jointmean_slope"] = slope_per_min(t, jt)
        r["D2JTD_mean10"] = float(np.nanmedian(dj)) if len(dj) else np.nan
        r["D2JTD_mean10_slope"] = slope_per_min(t, dj)
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# RQ4 repeatability
# --------------------------------------------------------------------------- #
def icc31(x1, x2):
    X = np.column_stack([x1, x2])
    X = X[np.isfinite(X).all(1)]
    n, k = X.shape
    if n < 5:
        return np.nan
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse))


def rq4(df, open_thr):
    s = df[(df.task == "static") & df.frame_valid]
    halves = []
    for rec, g in s.groupby("rec_id"):
        g = g.sort_values("t_s")
        mid = g.t_s.median()
        for h, gg in (("h1", g[g.t_s <= mid]), ("h2", g[g.t_s > mid])):
            r = dict(rec_id=rec, half=h)
            for j in JOINT_NAMES:
                r[j] = gg[f"{j}_D2JTD"].median()
            r["D2JTD_mean10"] = gg[[f"{j}_D2JTD" for j in QIAO_JOINTS]].mean(axis=1).median()
            halves.append(r)
    H = pd.DataFrame(halves)
    a, b = H[H.half == "h1"].set_index("rec_id"), H[H.half == "h2"].set_index("rec_id")
    rows = []
    for j in JOINT_NAMES + ["D2JTD_mean10"]:
        d = (b[j] - a[j]).dropna()
        sem = d.std(ddof=1) / np.sqrt(2)
        mdc = 1.96 * np.sqrt(2) * sem
        rows.append(dict(feature=j, n_hands=len(d), icc31=icc31(a[j].values, b[j].values),
                         mean_diff=float(d.mean()), sem=float(sem), mdc95=float(mdc), pass_mdc_lt_04=bool(mdc < QIAO_RA_GAP)))
    t = pd.DataFrame(rows)
    # frame-to-frame noise inside a recording (single-frame SD of the 10-joint mean D2JTD)
    fr = s.assign(m=s[[f"{j}_D2JTD" for j in QIAO_JOINTS]].mean(axis=1)).groupby("rec_id")["m"].std()
    joints = t[t.feature != "D2JTD_mean10"]
    summ = dict(joints_passing=int(joints.pass_mdc_lt_04.sum()), n_joints=len(joints),
                mean10_mdc95=float(t.loc[t.feature == "D2JTD_mean10", "mdc95"].iloc[0]),
                mean10_icc=float(t.loc[t.feature == "D2JTD_mean10", "icc31"].iloc[0]),
                median_joint_mdc95=float(joints.mdc95.median()),
                single_frame_sd_mean10_median=float(fr.median()))
    summ["pass_"] = bool(summ["mean10_mdc95"] < QIAO_RA_GAP)
    return t, (a, b), summ


# --------------------------------------------------------------------------- #
# RQ5 reference ranges
# --------------------------------------------------------------------------- #
def boot_ci(x, rng, n=5000):
    x = np.asarray(x)[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def rq5(F, rng):
    st = F[F.task == "static"]
    subj = st.groupby("subject")[[f"{j}_D2JTD" for j in JOINT_NAMES] + ["D2JTD_mean10"]].mean()
    rows = []
    for j in JOINT_NAMES + ["D2JTD_mean10"]:
        c = j if j == "D2JTD_mean10" else f"{j}_D2JTD"
        x = subj[c].dropna().values
        lo, hi = boot_ci(x, rng)
        w = stats.wilcoxon(x, alternative="greater") if len(x) >= 6 else None
        rows.append(dict(feature=j, n_subjects=len(x), mean=x.mean(), sd=x.std(ddof=1), ci_lo=lo, ci_hi=hi,
                         ref_lo=x.mean() - 1.96 * x.std(ddof=1), ref_hi=x.mean() + 1.96 * x.std(ddof=1),
                         frac_positive=float((x > 0).mean()), p_greater_0=float(w.pvalue) if w else np.nan))
    t = pd.DataFrame(rows)
    q = t.feature.isin(QIAO_JOINTS)
    t.loc[q, "p_holm"] = holm(t.loc[q, "p_greater_0"].values)
    h1_joints = int((t.loc[q, "p_holm"] < 0.05).sum())
    # left - right
    lr_rows = []
    piv = st.pivot_table(index="subject", columns="hand", values=[f"{j}_T" for j in JOINT_NAMES] + [f"{j}_D2JTD" for j in JOINT_NAMES])
    for j in JOINT_NAMES:
        dT = (piv[(f"{j}_T", "left")] - piv[(f"{j}_T", "right")]).dropna()
        dD = (piv[(f"{j}_D2JTD", "left")] - piv[(f"{j}_D2JTD", "right")]).dropna()
        lr_rows.append(dict(joint=j, n=len(dT), dT_mean=dT.mean(), dT_sd=dT.std(ddof=1), absdT_median=dT.abs().median(),
                            frac_absdT_lt_05=float((dT.abs() < 0.5).mean()), dD2JTD_mean=dD.mean(), absdD2JTD_median=dD.abs().median()))
    lr = pd.DataFrame(lr_rows)
    m10 = t[t.feature == "D2JTD_mean10"].iloc[0]
    summ = dict(mean10=float(m10["mean"]), mean10_sd=float(m10["sd"]), mean10_ci=[float(m10.ci_lo), float(m10.ci_hi)],
                qiao_healthy_in_ci=bool(m10.ci_lo <= QIAO_HEALTHY[0] <= m10.ci_hi),
                h1_joints_supported=h1_joints, h1_joints_tested=int(q.sum()),
                h2_joints_median_lt_05=int((lr.absdT_median < 0.5).sum()), h2_n_joints=len(lr),
                h2_supported=bool((lr.absdT_median < 0.5).sum() > len(lr) / 2),
                lr_frac_all_lt_05=float(np.nanmean(lr.frac_absdT_lt_05)))
    return t, lr, subj, summ


# --------------------------------------------------------------------------- #
# RQ6 open-close response
# --------------------------------------------------------------------------- #
def rq6(F):
    p = F.pivot_table(index=["subject", "hand"], columns="task",
                      values=["dorsum_slope", "jointmean_slope", "D2JTD_mean10_slope", "D2JTD_mean10", "dorsum_T", "jointmean_T"])
    rows = []
    for v in ["dorsum_slope", "jointmean_slope", "D2JTD_mean10_slope", "D2JTD_mean10", "dorsum_T", "jointmean_T"]:
        a, b = p[(v, "static")], p[(v, "open_close")]
        m = a.notna() & b.notna()
        d = (b - a)[m]
        w = stats.wilcoxon(d) if len(d) >= 6 else None
        rpos, rneg = (stats.rankdata(np.abs(d))[d > 0].sum(), stats.rankdata(np.abs(d))[d < 0].sum())
        rb = (rpos - rneg) / (rpos + rneg) if (rpos + rneg) else np.nan
        # subject-level sensitivity (average the two hands)
        ds = d.groupby(level="subject").mean()
        ws = stats.wilcoxon(ds) if len(ds) >= 6 else None
        rows.append(dict(variable=v, n_hands=int(m.sum()), static_median=float(a[m].median()), open_close_median=float(b[m].median()),
                         diff_median=float(d.median()), rank_biserial=float(rb), p_hands=float(w.pvalue) if w else np.nan,
                         p_subjects=float(ws.pvalue) if ws else np.nan))
    return pd.DataFrame(rows), p


# --------------------------------------------------------------------------- #
# RQ7 one-class (healthy-only) model with synthetic warming
# --------------------------------------------------------------------------- #
def rq7(F, rng, deltas=(0.25, 0.5, 1.0, 1.5, 2.0)):
    st = F[F.task == "static"].reset_index(drop=True)
    cols = [f"{j}_D2JTD" for j in JOINT_NAMES]
    X = st[cols].values.astype(float)
    subj = st.subject.values
    qidx = [JOINT_NAMES.index(j) for j in QIAO_JOINTS]
    res = {m: dict(normal=[], single={d: [] for d in deltas}, global_={d: [] for d in deltas}) for m in ("maxz", "mahal")}
    for s in np.unique(subj):
        tr, te = X[subj != s], X[subj == s]
        mu = np.nanmean(tr, 0)
        tr = np.where(np.isnan(tr), mu, tr)
        te = np.where(np.isnan(te), mu, te)
        sd = tr.std(0, ddof=1)
        lw = LedoitWolf().fit(tr)

        def score(Z, m):
            if m == "maxz":
                return np.max(np.abs((Z - mu) / sd), axis=1)
            return lw.mahalanobis(Z)
        for m in res:
            thr = np.percentile(score(tr, m), 95)
            res[m]["normal"] += list(score(te, m) <= thr)
            for d in deltas:
                for j in range(X.shape[1]):              # one joint warmer by d -> its D2JTD drops by d
                    Z = te.copy()
                    Z[:, j] -= d
                    res[m]["single"][d] += list(score(Z, m) > thr)
                Z = te.copy()
                Z[:, qidx] -= d                          # all 10 MCP/PIP joints warmer (polyarthritis-like)
                res[m]["global_"][d] += list(score(Z, m) > thr)
    rows = []
    for m, r in res.items():
        spec = float(np.mean(r["normal"]))
        for d in deltas:
            rows.append(dict(model=m, delta_C=d, specificity=spec, sens_single_joint=float(np.mean(r["single"][d])),
                             sens_all_10_joints=float(np.mean(r["global_"][d]))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# RGB <-> thermal synchronisation check (hand openness vs thermal hand area)
# --------------------------------------------------------------------------- #
def sync_check(df, stride, max_steps=20):
    rows = []
    for rec, g in df[df.task == "open_close"].groupby("rec_id"):
        g = g.sort_values("frame")
        a = g.openness.interpolate().values
        b = g.th_area.astype(float).interpolate().values
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        a, b = (a - a.mean()) / (a.std() + 1e-9), (b - b.mean()) / (b.std() + 1e-9)
        best = []
        for L in range(-max_steps, max_steps + 1):
            x = a[max(0, L):len(a) + min(0, L)]
            y = b[max(0, -L):len(b) - max(0, L)]
            best.append((float(np.corrcoef(x, y)[0, 1]), L))
        best.sort(reverse=True)
        (c0, L0), (c1, _) = best[0], best[1]
        rows.append(dict(rec_id=rec, best_lag_frames=L0 * stride, corr=c0, margin=c0 - c1,
                         status=("weak signal" if c0 <= 0.3 else "aligned" if abs(L0) <= 1
                                 else "ambiguous (tie with other lags)" if c0 - c1 < 0.02 else f"offset {L0 * stride:+d} frames")))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_reference(t5, subj, path):
    t = t5[t5.feature != "D2JTD_mean10"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    y = np.arange(len(t))[::-1]
    for yi, j in zip(y, t.feature):
        x = subj[f"{j}_D2JTD"].dropna().values
        ax.scatter(x, yi + (np.random.default_rng(1).random(len(x)) - 0.5) * 0.35, s=14, color=BLUE, alpha=0.45, lw=0)
    ax.errorbar(t["mean"], y, xerr=[t["mean"] - t.ci_lo, t.ci_hi - t["mean"]], fmt="o", color=INK, ms=4, lw=1.2, capsize=0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axvline(QIAO_HEALTHY[0], color=ORANGE, lw=1.2, ls="--")
    ax.text(QIAO_HEALTHY[0], -0.9, " Qiao et al. healthy mean (1.5 °C)", color=INK2, fontsize=8, va="center")
    ax.set_ylim(-1.3, len(t) - 0.4)
    ax.set_yticks(y, [j.replace("_", " ") for j in t.feature])
    ax.set_xlabel("D2JTD = dorsum − joint temperature (°C)   ·   > 0 means joint cooler than back of hand")
    ax.set_title("Healthy reference: per-joint D2JTD, static recordings (dots = subjects, black = mean ± 95% CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_repeatability(halves, path):
    a, b = halves
    x1, x2 = a["D2JTD_mean10"], b["D2JTD_mean10"]
    m = (x1 + x2) / 2
    d = x2 - x1
    md, sd = d.mean(), d.std(ddof=1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(m, d, s=26, color=BLUE, edgecolor=SURF, lw=1)
    for v, lab in ((md, "mean"), (md + 1.96 * sd, "+1.96 SD"), (md - 1.96 * sd, "−1.96 SD")):
        ax.axhline(v, color=INK2 if lab == "mean" else ORANGE, lw=1, ls="-" if lab == "mean" else "--")
        ax.text(ax.get_xlim()[1], v, f" {lab} {v:+.2f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("mean of halves, 10-joint D2JTD (°C)")
    ax.set_ylabel("2nd half − 1st half (°C)")
    ax.set_title("Within-recording repeatability (static, one dot per hand)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_open_close(p, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), sharey=True)
    for ax, v, title in zip(axes, ["dorsum_slope", "jointmean_slope"], ["Back of hand", "Mean of 14 joints"]):
        a, b = p[(v, "static")], p[(v, "open_close")]
        for ai, bi in zip(a, b):
            if np.isfinite(ai) and np.isfinite(bi):
                ax.plot([0, 1], [ai, bi], color=GRID, lw=1, zorder=1)
        ax.scatter(np.zeros(len(a)), a, s=22, color=BLUE, zorder=2, edgecolor=SURF, lw=1, label="static")
        ax.scatter(np.ones(len(b)), b, s=22, color=ORANGE, zorder=2, edgecolor=SURF, lw=1, label="open–close")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks([0, 1], ["static", "open–close"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(title)
    axes[0].set_ylabel("temperature trend (°C per minute)")
    fig.suptitle("Temperature change during the recording, per hand (lines join the same hand)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_anomaly(t7, path):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for m, col, lab in (("maxz", BLUE, "max |z|"), ("mahal", ORANGE, "Mahalanobis")):
        g = t7[t7.model == m]
        ax.plot(g.delta_C, g.sens_all_10_joints, color=col, lw=2, marker="o", ms=8, label=f"{lab}: all 10 MCP/PIP warmer")
        ax.plot(g.delta_C, g.sens_single_joint, color=col, lw=2, ls="--", marker="s", ms=7, label=f"{lab}: one joint warmer")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("synthetic warming added to joint(s) (°C)")
    ax.set_ylabel("sensitivity (fraction flagged)")
    spec = t7.groupby("model").specificity.first()
    ax.set_title(f"Healthy-only model, leave-one-subject-out (specificity: max|z| {spec['maxz']:.2f}, Mahalanobis {spec['mahal']:.2f})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def md_table(df: pd.DataFrame, floatfmt=".2f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stages 9-11 analysis")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config))
    pc = cfg["pipeline"]
    proc, out = Path(pc["processed_dir"]), Path(pc["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    style()
    native = int(pc["native_px_per_axis"])

    df = load_frames(proc)
    reg = pd.read_csv(proc / "registration.csv")
    thr = open_threshold(df)

    t1, per_rec1, s1 = rq1(df)
    s2, err2 = rq2(reg, df, Path("data/annotations/joints_clicked.csv"), native)
    t3, s3 = rq3(df, pc["roi"]["on_skin_frac"], native)
    F = features(df, thr)
    t4, halves, s4 = rq4(df, thr)
    t5, lr5, subj5, s5 = rq5(F, rng)
    t6, p6 = rq6(F)
    t7 = rq7(F, rng)
    sync = sync_check(df, int(pc["frame_stride"]))

    for name, t in [("rq1_landmarks.csv", t1), ("rq1_per_recording.csv", per_rec1), ("rq3_rois.csv", t3),
                    ("features_per_recording.csv", F), ("rq4_repeatability.csv", t4), ("rq5_reference.csv", t5),
                    ("rq5_left_right.csv", lr5), ("rq6_open_close.csv", t6), ("rq7_one_class.csv", t7),
                    ("sync_rgb_thermal.csv", sync), ("registration.csv", reg)]:
        t.to_csv(out / name, index=False)
    if err2 is not None and len(err2):
        err2.to_csv(out / "rq2_landmark_errors.csv", index=False)

    fig_reference(t5, subj5, out / "fig_reference_d2jtd.png")
    fig_repeatability(halves, out / "fig_repeatability.png")
    fig_open_close(p6, out / "fig_open_close.png")
    fig_anomaly(t7, out / "fig_one_class.png")

    # healthy reference for the edge demo
    ref = {j: dict(mean=float(r["mean"]), sd=float(r["sd"])) for j, r in t5.set_index("feature").iterrows()}
    Path("models").mkdir(exist_ok=True)
    json.dump(dict(feature="D2JTD (dorsum - joint, deg C), static, healthy volunteers", n_subjects=int(t5.n_subjects.max()),
                   joints=ref), open("models/healthy_reference.json", "w"), indent=2)

    summary = dict(open_threshold=thr, RQ1=s1, RQ2=s2, RQ3=s3, RQ4=s4, RQ5=s5,
                   RQ6=t6.to_dict("records"), RQ7=t7.to_dict("records"),
                   sync=dict(counts=sync.status.value_counts().to_dict(), median_corr=float(sync["corr"].median())))
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    md = ["# Results tables (generated by src.analysis)", "",
          f"Open-hand threshold (openness, 5th pct of static frames): {thr:.2f}", "",
          "## RQ1 landmarks", md_table(t1), "",
          "## RQ2 registration", "```", json.dumps(s2, indent=2), "```", "",
          "## RQ3 ROIs", md_table(t3), "",
          "## RQ4 repeatability (static, half vs half)", md_table(t4, ".3f"), "",
          "## RQ5 healthy reference (subject means, static)", md_table(t5, ".2f"), "",
          "## RQ5 left − right (static)", md_table(lr5, ".2f"), "",
          "## RQ6 static vs open–close", md_table(t6, ".3f"), "",
          "## RQ7 one-class", md_table(t7, ".2f"), "",
          "## RGB ↔ thermal sync (open–close)", md_table(sync, ".2f"), ""]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
