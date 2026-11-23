"""v7 — Healthy-reference anomaly models with leakage-free, subject-independent evaluation.

Goal: perform well on UNSEEN people, not on these 17. Rules enforced in code:
  * Unit of independence = subject. Outer loop: leave-one-subject-out (LOSO).
  * Everything learned — imputation values, scaling, covariance, the model and the DECISION
    THRESHOLD — is learned from training subjects only. The threshold comes from an inner LOSO
    over the training subjects (nested), at the 95th percentile of their held-out scores, so the
    target specificity (0.95) is estimated on data the model did not see (v3 set it in-sample and
    got 0.88 instead of 0.95 for Mahalanobis).
  * Small models: 17 subjects cannot support a full 14x14 covariance (105 parameters), so the
    covariance models use Ledoit-Wolf shrinkage, a diagonal, or a robust MCD estimate.
  * Uncertainty: 95 % confidence intervals by resampling SUBJECTS (bootstrap), not frames.

Anomalies are SIMULATED (healthy hands only, no patients): a joint warmer by d degC lowers its
D2JTD by d. Patterns: single joint, MCP-only, PIP-only, DIP-only, all 10 MCP/PIP, diffuse (whole
hand incl. dorsum warmer -> D2JTD unchanged: relative features are blind to it by design).
None of this is clinical performance.

Outputs: reports/v7_models/ and a versioned model bundle in models/registry/<model_id>/.

Usage:  .venv312\\Scripts\\python.exe -m src.anomaly_models [--features reports/results/features_per_recording.csv]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.landmarks import JOINT_NAMES, QIAO_JOINTS

OUT = Path("reports/v7_models")
REGISTRY = Path("models/registry")
SEED = 42
TARGET_SPEC = 0.95
DELTAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
FEATURES = [f"{j}_D2JTD" for j in JOINT_NAMES]
PATTERNS = {
    "single_joint": None,                                                     # each joint in turn
    "MCP_only": [JOINT_NAMES.index(j) for j in JOINT_NAMES if j.endswith("MCP")],
    "PIP_only": [JOINT_NAMES.index(j) for j in JOINT_NAMES if j.endswith(("PIP", "_IP"))],
    "DIP_only": [JOINT_NAMES.index(j) for j in JOINT_NAMES if j.endswith("DIP")],
    "all_10_MCP_PIP": [JOINT_NAMES.index(j) for j in QIAO_JOINTS],
    "diffuse_whole_hand": [],                                                 # dorsum warms too: D2JTD unchanged
}


# --------------------------------------------------------------------------- #
# Models: fit(X) -> self, score(X) -> higher = more abnormal
# --------------------------------------------------------------------------- #
class Imputer:
    """Median imputation learned on training data only."""

    def fit(self, X):
        self.med_ = np.nanmedian(X, axis=0)
        self.med_ = np.where(np.isfinite(self.med_), self.med_, 0.0)
        return self

    def transform(self, X):
        return np.where(np.isnan(X), self.med_, X)


class RobustZ:
    """Max over joints of |x - median| / (1.4826 * MAD). Two numbers per joint."""
    name = "robust_z_max"

    def fit(self, X):
        self.c_ = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.abs(X - self.c_), axis=0)
        self.s_ = np.maximum(mad, 0.05)                          # floor: 0.05 degC
        return self

    def score(self, X):
        return np.max(np.abs((X - self.c_) / self.s_), axis=1)

    def explain(self, x):
        return (x - self.c_) / self.s_                          # per-joint robust z


class DiagMahal:
    """Mahalanobis with a diagonal covariance (joints treated as independent)."""
    name = "mahalanobis_diag"

    def fit(self, X):
        self.c_ = X.mean(0)
        self.s_ = np.maximum(X.std(0, ddof=1), 0.05)
        return self

    def score(self, X):
        return np.sqrt(np.sum(((X - self.c_) / self.s_) ** 2, axis=1))

    def explain(self, x):
        return (x - self.c_) / self.s_


class ShrinkMahal:
    """Mahalanobis with Ledoit-Wolf shrinkage covariance."""
    name = "mahalanobis_ledoit_wolf"

    def fit(self, X):
        self.m_ = LedoitWolf().fit(X)
        return self

    def score(self, X):
        return np.sqrt(self.m_.mahalanobis(X))

    def explain(self, x):
        return (x - self.m_.location_) / np.sqrt(np.diag(self.m_.covariance_))


class RobustCov:
    """Mahalanobis with Minimum Covariance Determinant (robust to outlying training hands)."""
    name = "robust_covariance_mcd"

    def fit(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m_ = MinCovDet(random_state=SEED, support_fraction=0.8).fit(X)
        return self

    def score(self, X):
        return np.sqrt(self.m_.mahalanobis(X))

    def explain(self, x):
        return (x - self.m_.location_) / np.sqrt(np.diag(self.m_.covariance_))


class IForest:
    name = "isolation_forest"

    def fit(self, X):
        self.m_ = IsolationForest(n_estimators=200, random_state=SEED).fit(X)
        self.c_, self.s_ = np.median(X, 0), np.maximum(X.std(0, ddof=1), 0.05)
        return self

    def score(self, X):
        return -self.m_.score_samples(X)

    def explain(self, x):
        return (x - self.c_) / self.s_


class OCSVM:
    name = "one_class_svm"

    def fit(self, X):
        self.sc_ = StandardScaler().fit(X)
        self.m_ = OneClassSVM(nu=0.1, gamma="scale").fit(self.sc_.transform(X))
        return self

    def score(self, X):
        return -self.m_.decision_function(self.sc_.transform(X))

    def explain(self, x):
        return self.sc_.transform(x[None])[0]


MODELS = {c.name: c for c in (RobustZ, DiagMahal, ShrinkMahal, RobustCov, IForest, OCSVM)}


class Pipeline:
    """Imputer + model + threshold, fitted on training subjects only."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def fit(self, X):
        self.imp_ = Imputer().fit(X)
        Z = self.imp_.transform(X)
        self.model_ = MODELS[self.model_name]().fit(Z)
        lo, hi = np.nanmin(X, 0), np.nanmax(X, 0)
        pad = 0.25 * (hi - lo)
        self.range_ = (lo - pad, hi + pad)                     # "model experience" box
        return self

    def score(self, X):
        return self.model_.score(self.imp_.transform(X))

    def out_of_range(self, X):
        """True where any feature lies outside the training range (+25 % margin)."""
        lo, hi = self.range_
        with np.errstate(invalid="ignore"):
            return np.any((X < lo) | (X > hi), axis=1)


def nested_threshold(X, groups, model_name, q=TARGET_SPEC):
    """Threshold from an inner LOSO over the given (training) subjects only."""
    scores = []
    for u in np.unique(groups):
        tr = groups != u
        p = Pipeline(model_name).fit(X[tr])
        scores.append(p.score(X[~tr]))
    return float(np.quantile(np.concatenate(scores), q))


# --------------------------------------------------------------------------- #
# Synthetic anomalies
# --------------------------------------------------------------------------- #
def warm(X, joints, d):
    Z = X.copy()
    if joints:
        Z[:, joints] -= d                                      # joint warmer -> D2JTD lower
    return Z


def evaluate_model(X, groups, model_name):
    """Outer LOSO with nested thresholds. Returns per-hand rows (no leakage)."""
    rows, thresholds, fit_ms, score_us = [], [], [], []
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        t0 = time.perf_counter()
        thr = nested_threshold(X[tr], groups[tr], model_name)
        p = Pipeline(model_name).fit(X[tr])
        fit_ms.append(1000 * (time.perf_counter() - t0))
        thresholds.append(thr)
        Xt = X[te]
        t1 = time.perf_counter()
        sc = p.score(Xt)
        score_us.append(1e6 * (time.perf_counter() - t1) / len(Xt))
        oor = p.out_of_range(Xt)
        for i in range(len(Xt)):
            rows.append(dict(subject=s, pattern="none", delta=0.0, joint=-1, score=sc[i], flagged=sc[i] > thr,
                             out_of_range=bool(oor[i])))
        for pat, js in PATTERNS.items():
            for d in DELTAS:
                sets = [[j] for j in range(X.shape[1])] if js is None else [js]
                for jj in sets:
                    sc2 = p.score(warm(Xt, jj, d))
                    for i in range(len(Xt)):
                        rows.append(dict(subject=s, pattern=pat, delta=d, joint=jj[0] if len(jj) == 1 else -1,
                                         score=sc2[i], flagged=sc2[i] > thr))
    R = pd.DataFrame(rows)
    cost = dict(train_ms_per_fold_incl_threshold=float(np.median(fit_ms)), score_us_per_hand=float(np.median(score_us)),
                threshold_cv=float(np.std(thresholds) / np.mean(thresholds)) if np.mean(thresholds) else np.nan)
    return R, cost


def subject_bootstrap(R, col, n=2000, seed=SEED):
    """95 % CI of a per-row mean, resampling subjects."""
    rng = np.random.default_rng(seed)
    subs = R.subject.unique()
    per = R.groupby("subject")[col].agg(["sum", "count"])
    vals = []
    for _ in range(n):
        b = rng.choice(subs, len(subs), replace=True)
        s = per.loc[b]
        vals.append(s["sum"].sum() / s["count"].sum())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarise(R, model_name, cost):
    n = R[R.pattern == "none"]
    spec = 1 - n.flagged.mean()
    lo, hi = subject_bootstrap(n.assign(ok=~n.flagged), "ok")
    out = dict(model=model_name, specificity=spec, specificity_ci=[lo, hi],
               false_out_of_range_rate=float(n.out_of_range.mean()), **cost)
    for pat in PATTERNS:
        for d in (0.5, 1.0, 2.0):
            g = R[(R.pattern == pat) & (R.delta == d)]
            out[f"sens_{pat}_{d:g}C"] = float(g.flagged.mean())
    g1 = R[(R.pattern == "all_10_MCP_PIP") & (R.delta == 1.0)]
    out["sens_all10_1C_ci"] = list(subject_bootstrap(g1.assign(f=g1.flagged.astype(float)), "f"))
    # pre-declared selection criterion: mean of specificity and the two headline sensitivities
    out["selection_score"] = float(np.mean([spec, out["sens_all_10_MCP_PIP_1C"], out["sens_single_joint_2C"]]))
    return out


def learning_curve(X, groups, model_name, sizes=(4, 6, 8, 10, 12, 14, 16), reps=10):
    """Performance on held-out subjects vs number of training subjects (nested thresholds)."""
    rng = np.random.default_rng(SEED)
    subs = np.unique(groups)
    rows = []
    for s in subs:
        pool = subs[subs != s]
        for n in sizes:
            for r in range(reps if n < len(pool) else 1):
                chosen = rng.choice(pool, n, replace=False)
                m = np.isin(groups, chosen)
                thr = nested_threshold(X[m], groups[m], model_name)
                p = Pipeline(model_name).fit(X[m])
                Xt = X[groups == s]
                spec = np.mean(p.score(Xt) <= thr)
                sens = np.mean(p.score(warm(Xt, PATTERNS["all_10_MCP_PIP"], 1.0)) > thr)
                rows.append(dict(test_subject=s, n_train=n, rep=r, specificity=spec, sens_all10_1C=sens))
    return pd.DataFrame(rows)


def save_bundle(X, groups, subjects_ids, model_name, summary, features_path: Path, cfg_hash: str,
                features: list[str] | None = None, training_dataset: str | None = None) -> Path:
    """Final model on ALL subjects, threshold from LOSO over all subjects; versioned, never overwritten."""
    thr = nested_threshold(X, groups, model_name)
    p = Pipeline(model_name).fit(X)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"v7_{model_name}_{stamp}"
    d = REGISTRY / model_id
    d.mkdir(parents=True, exist_ok=False)
    features = features or FEATURES
    joblib.dump(dict(pipeline=p, threshold=thr, features=features, kind="anomaly"), d / "bundle.joblib")
    meta = dict(
        model_id=model_id, model_type=model_name, created=dt.datetime.now().isoformat(timespec="seconds"),
        task="healthy-reference anomaly detection (NOT an arthritis classifier)",
        training_dataset=training_dataset or "FLIROne (17 healthy volunteers), static recordings, one row per hand",
        training_subjects=sorted(map(str, set(subjects_ids))), n_training_hands=int(len(X)),
        features=features, feature_source=str(features_path),
        feature_sha256=hashlib.sha256(features_path.read_bytes()).hexdigest(),
        preprocessing=dict(imputation="median of training hands", scaling="model-internal",
                           out_of_range_margin="25 % of training range per feature"),
        threshold=dict(value=thr, method=f"{TARGET_SPEC:.2f} quantile of leave-one-subject-out scores of training subjects"),
        calibration_version=cfg_hash, validation=summary,
        software=dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__),
        limitations="Healthy subjects only; validation uses simulated warming; scores are not probabilities.")
    json.dump(meta, open(d / "metadata.json", "w"), indent=2, default=float)
    return d


def load_joblib(path: Path):
    """joblib.load that also works for bundles saved while this module ran as a script (classes pickled as
    __main__.Pipeline etc.): those names are resolved from this module, whatever program is running."""
    import sys
    main = sys.modules["__main__"]
    names = [n for n in ("Pipeline", "Imputer", "RobustZ", "DiagMahal", "ShrinkMahal", "RobustCov", "IForest", "OCSVM")
             if not hasattr(main, n)]
    for n in names:
        setattr(main, n, globals()[n])
    try:
        return joblib.load(path)
    finally:
        for n in names:
            delattr(main, n)


def load_bundle(path: Path) -> dict:
    b = load_joblib(Path(path) / "bundle.joblib")
    b["metadata"] = json.load(open(Path(path) / "metadata.json"))
    return b


def figures(C, LC, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.analysis import BLUE, ORANGE, style
    style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(C))
    ax.barh(y + 0.2, C.specificity, height=0.38, color=BLUE, label="specificity (real held-out hands)")
    ax.barh(y - 0.2, C["sens_all_10_MCP_PIP_1C"], height=0.38, color=ORANGE, label="sensitivity: all 10 MCP/PIP +1 °C (simulated)")
    ax.set_yticks(y, C.model.str.replace("_", " "))
    ax.axvline(TARGET_SPEC, color="#52514e", lw=1, ls="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction (leave-one-subject-out, nested threshold)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    ax.set_title("Anomaly models: unseen-subject performance")
    fig.tight_layout()
    fig.savefig(out / "fig_model_comparison.png", dpi=130)
    plt.close(fig)
    g = LC.groupby("n_train")[["specificity", "sens_all10_1C"]].mean()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(g.index, g.specificity, "-o", color=BLUE, lw=2, ms=8, label="specificity")
    ax.plot(g.index, g.sens_all10_1C, "-s", color=ORANGE, lw=2, ms=8, label="sensitivity, all 10 joints +1 °C")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("number of training subjects")
    ax.set_ylabel("held-out subject performance")
    ax.set_title("Learning curve (selected model): would more subjects help?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_learning_curve.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="reports/results/features_per_recording.csv")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-save", action="store_true", help="do not write a model bundle")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    fpath = Path(args.features)
    F = pd.read_csv(fpath)
    F = F[F.task == "static"].reset_index(drop=True)
    X = F[FEATURES].to_numpy(float)
    groups = F.subject.to_numpy()
    cfg_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()[:12]

    results, allR = [], []
    for name in MODELS:
        t0 = time.perf_counter()
        R, cost = evaluate_model(X, groups, name)
        s = summarise(R, name, cost)
        s["eval_seconds"] = round(time.perf_counter() - t0, 1)
        results.append(s)
        allR.append(R.assign(model=name))
        print(f"{name:26s} spec {s['specificity']:.2f} [{s['specificity_ci'][0]:.2f}-{s['specificity_ci'][1]:.2f}]"
              f"  all10+1C {s['sens_all_10_MCP_PIP_1C']:.2f}  single+2C {s['sens_single_joint_2C']:.2f}"
              f"  OOR {s['false_out_of_range_rate']:.2f}  thrCV {s['threshold_cv']:.2f}")
    C = pd.DataFrame(results).sort_values("selection_score", ascending=False)
    R = pd.concat(allR)
    C.to_csv(OUT / "model_comparison.csv", index=False)
    sens = R[R.pattern != "none"].groupby(["model", "pattern", "delta"]).flagged.mean().unstack("delta").reset_index()
    sens.to_csv(OUT / "synthetic_sensitivity.csv", index=False)
    best = C.iloc[0].model
    LC = learning_curve(X, groups, best)
    LC.to_csv(OUT / "learning_curve.csv", index=False)
    figures(C, LC, OUT)
    summary = dict(selected_model=best, selection_rule="max mean(specificity, sens all-10 +1C, sens single-joint +2C)",
                   n_subjects=int(len(np.unique(groups))), n_hands=int(len(X)),
                   comparison=C.to_dict("records"),
                   learning_curve=LC.groupby("n_train")[["specificity", "sens_all10_1C"]].mean().round(3).reset_index().to_dict("records"),
                   v3_reference=dict(note="v3 thresholds were set in-sample", maxz_specificity=0.94, mahalanobis_specificity=0.88))
    if not args.no_save:
        d = save_bundle(X, groups, groups, best, C.iloc[0].to_dict(), fpath, cfg_hash)
        summary["saved_bundle"] = str(d)
        print("saved", d)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2, default=float)
    print("selected:", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
