"""v8 — Ground-truth labels, evaluation metrics, model training wrappers and the locked
independent-test workflow.

Scientific guards built in:
  * Labels are only read from files; they are never inferred from temperatures.
  * Metrics that are undefined for the available labels are not computed (e.g. ROC with one class);
    the reason is reported instead.
  * Measurement error (pixels, deg C) is reported separately from classification metrics.
  * Independent tests never refit, re-threshold or re-scale: the saved bundle is applied as is.
    Every run gets its own folder; inputs are fingerprinted in lock.json.
  * Anomaly flags are compared with labels only as "screening flag vs label": the anomaly score is
    not a probability of arthritis.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics as skm

TESTS = Path("results/independent_tests")
LABEL_MAP = {"healthy": "healthy", "normal": "healthy", "control": "healthy", "hc": "healthy", "0": "healthy",
             "arthritis": "arthritis", "ra": "arthritis", "oa": "arthritis", "psa": "arthritis", "patient": "arthritis",
             "1": "arthritis", "other": "other", "other condition": "other", "unknown": "unknown", "": "unlabelled",
             "nan": "unlabelled", "unlabelled": "unlabelled", "unlabeled": "unlabelled"}
CATEGORIES = ("Normal measurement", "Potential anomaly", "Low-confidence measurement", "Invalid measurement",
              "Insufficient data", "Outside model experience", "Not applicable")


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def load_labels(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    L = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
    L.columns = [c.strip().lower().replace(" ", "_") for c in L.columns]
    if "subject" not in L.columns and "rec_id" not in L.columns:
        raise ValueError("label file needs a 'subject' or 'rec_id' column")
    if "label" not in L.columns:
        raise ValueError("label file needs a 'label' column")
    L["label_raw"] = L["label"].astype(str)
    L["label"] = L["label"].astype(str).str.strip().str.lower().map(LABEL_MAP).fillna("other")
    for c in ("subject", "hand", "rec_id", "joint", "source"):
        if c not in L.columns:
            L[c] = np.nan
    L["subject"] = L["subject"].astype(str).str.upper().replace("NAN", np.nan)
    L["hand"] = L["hand"].astype(str).str.lower().map({"l": "left", "r": "right", "left": "left", "right": "right"})
    return L


def match_labels(L: pd.DataFrame, recordings: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Recording-level label: rec_id row > subject+hand row > subject row. Joint rows are kept apart."""
    rec_rows = L[L.rec_id.notna() & L.joint.isna()]
    hand_rows = L[L.rec_id.isna() & L.hand.notna() & L.joint.isna()]
    subj_rows = L[L.rec_id.isna() & L.hand.isna() & L.joint.isna()]
    out = []
    for r in recordings:
        lab, src, lvl = "unlabelled", None, None
        m = rec_rows[rec_rows.rec_id == r["rec_id"]]
        if len(m):
            lab, src, lvl = m.iloc[0].label, m.iloc[0].source, "recording"
        else:
            m = hand_rows[(hand_rows.subject == r["subject"]) & (hand_rows.hand == r["hand"])]
            if len(m):
                lab, src, lvl = m.iloc[0].label, m.iloc[0].source, "hand"
            else:
                m = subj_rows[subj_rows.subject == r["subject"]]
                if len(m):
                    lab, src, lvl = m.iloc[0].label, m.iloc[0].source, "subject"
        out.append(dict(rec_id=r["rec_id"], subject=r["subject"], hand=r["hand"], task=r["task"], label=lab,
                        label_level=lvl, label_source=src))
    M = pd.DataFrame(out)
    known_subj = {r["subject"] for r in recordings}
    known_rec = {r["rec_id"] for r in recordings}
    report = dict(
        unmatched_label_rows=L[(L.rec_id.notna() & ~L.rec_id.isin(known_rec)) |
                               (L.rec_id.isna() & L.subject.notna() & ~L.subject.isin(known_subj))].to_dict("records"),
        recordings_without_label=M[M.label == "unlabelled"].rec_id.tolist(),
        label_counts=M.label.value_counts().to_dict(),
        missing_source=int(L[L.label.isin(["healthy", "arthritis"]) & L.source.isna()].shape[0]),
        joint_level_rows=int(L.joint.notna().sum()))
    return M, report


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, y_pred, scores=None, positive_name="arthritis") -> dict:
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    out = dict(n=int(len(y_true)), n_positive=n_pos, n_negative=n_neg, positive_class=positive_name)
    if len(y_true) == 0:
        return dict(out, note="no labelled samples")
    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["confusion_matrix"] = dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    if n_neg:
        out.update(specificity=tn / n_neg, false_positive_rate=fp / n_neg)
    if n_pos:
        out.update(sensitivity=tp / n_pos, recall=tp / n_pos, false_negative_rate=fn / n_pos)
    if n_pos and n_neg:
        out.update(accuracy=(tp + tn) / len(y_true), balanced_accuracy=(tp / n_pos + tn / n_neg) / 2,
                   precision=tp / (tp + fp) if tp + fp else np.nan,
                   f1=skm.f1_score(y_true, y_pred, zero_division=0))
        if scores is not None and np.isfinite(scores).all():
            out.update(roc_auc=skm.roc_auc_score(y_true, scores), average_precision=skm.average_precision_score(y_true, scores))
            fpr, tpr, _ = skm.roc_curve(y_true, scores)
            pr, rc, _ = skm.precision_recall_curve(y_true, scores)
            out["roc_curve"] = dict(fpr=fpr.tolist(), tpr=tpr.tolist())
            out["pr_curve"] = dict(precision=pr.tolist(), recall=rc.tolist())
        if min(n_pos, n_neg) < 10:
            out["small_sample_warning"] = (f"only {n_pos} positive and {n_neg} negative samples: estimates are "
                                           "very uncertain; do not report them without confidence intervals")
    else:
        out["note"] = (f"only one class present ({'positive' if n_pos else 'negative'}): accuracy, precision, F1, "
                       "ROC and PR are not meaningful and were not computed")
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def multiclass_metrics(y_true, y_pred) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    rep = skm.classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return dict(report=rep, confusion_matrix=skm.confusion_matrix(y_true, y_pred, labels=labels).tolist(), labels=labels)


def measurement_metrics(pred, ref, unit="") -> dict:
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    m = np.isfinite(pred) & np.isfinite(ref)
    e = pred[m] - ref[m]
    if not m.any():
        return dict(n=0, unit=unit, note="no paired measurements")
    return dict(n=int(m.sum()), unit=unit, mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                median_abs_error=float(np.median(np.abs(e))), p95_abs_error=float(np.percentile(np.abs(e), 95)),
                mean_signed_error=float(np.mean(e)))


# --------------------------------------------------------------------------- #
# Training (healthy reference) and supervised CV (only with real labels)
# --------------------------------------------------------------------------- #
def train_reference_model(F: pd.DataFrame, features: list[str], model_name: str, dataset_desc: str,
                          features_path: Path, labels: pd.DataFrame | None = None) -> tuple[Path, dict]:
    """Healthy-reference anomaly model: trained on static recordings labelled healthy (or all, if no
    labels, which is recorded in the metadata). Validation = nested LOSO specificity (+ simulated
    warming when the features are the 14 D2JTD values)."""
    from src import anomaly_models as AM
    st = F[F.task == "static"].copy()
    note = "no labels supplied: all static recordings assumed healthy"
    if labels is not None:
        st = st.merge(labels[["rec_id", "label"]], on="rec_id", how="left")
        st = st[st.label == "healthy"]
        note = "trained on recordings labelled healthy only"
    X = st[features].to_numpy(float)
    groups = st.subject.to_numpy()
    if len(np.unique(groups)) < 4:
        raise ValueError("need at least 4 healthy subjects to train and validate a reference model")
    if features == AM.FEATURES:
        R, cost = AM.evaluate_model(X, groups, model_name)
        summary = AM.summarise(R, model_name, cost)
    else:
        spec = []
        for s in np.unique(groups):
            tr = groups != s
            thr = AM.nested_threshold(X[tr], groups[tr], model_name)
            spec += list(AM.Pipeline(model_name).fit(X[tr]).score(X[~tr]) <= thr)
        summary = dict(model=model_name, specificity=float(np.mean(spec)), note="simulated warming only defined for D2JTD features")
    summary["training_note"] = note
    cfg_hash = hashlib.sha256(Path("config.yaml").read_bytes()).hexdigest()[:12]
    d = AM.save_bundle(X, groups, groups, model_name, summary, Path(features_path), cfg_hash,
                       features=features, training_dataset=dataset_desc)
    return d, summary


def supervised_cv(F: pd.DataFrame, labels: pd.DataFrame, features: list[str], model: str = "logistic_regression") -> dict:
    """Subject-grouped LOSO for a supervised healthy-vs-arthritis model. Refuses without both classes."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    D = F.merge(labels[["rec_id", "label"]], on="rec_id")
    D = D[D.label.isin(["healthy", "arthritis"]) & (D.task == "static")]
    y = (D.label == "arthritis").astype(int).to_numpy()
    if len(set(y)) < 2 or D.subject.nunique() < 6:
        raise ValueError("supervised training needs both healthy and arthritis labels from at least 6 subjects")
    est = {"logistic_regression": LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced"),
           "random_forest": RandomForestClassifier(n_estimators=300, max_depth=3, class_weight="balanced", random_state=42)}[model]
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)   # all fitted inside each fold
    X, g = D[features].to_numpy(float), D.subject.to_numpy()
    prob = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        pipe.fit(X[tr], y[tr])
        prob[te] = pipe.predict_proba(X[te])[:, 1]
    return dict(model=model, cv="leave-one-subject-out", **classification_metrics(y, prob >= 0.5, prob))


# --------------------------------------------------------------------------- #
# Independent test (locked)
# --------------------------------------------------------------------------- #
def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def categorise(row: pd.Series, flagged: bool, oor: bool, task_ok: bool) -> str:
    if not task_ok:
        return "Not applicable"
    if row.get("q_frames_used", 0) < 10:
        return "Insufficient data"
    if row.get("q_roi_usable", 0) < 0.5 or row.get("q_contaminated_rois", 0) >= 4:
        return "Invalid measurement"
    if row.get("q_roi_usable", 0) < 0.8 or (np.isfinite(row.get("q_registration_dice", np.nan)) and row["q_registration_dice"] < 0.7):
        return "Low-confidence measurement"
    if oor:
        return "Outside model experience"
    return "Potential anomaly" if flagged else "Normal measurement"


def run_independent_test(dataset_id: str, model_dir: str | Path, labels_path: str | Path | None = None,
                         test_name: str | None = None) -> Path:
    from src.dataset_manager import DATASETS, load_manifest
    m = load_manifest(dataset_id)
    fpath = DATASETS / dataset_id / "processed" / "features.csv"
    if not fpath.is_file():
        raise FileNotFoundError("process the dataset first (features.csv missing)")
    model_dir = Path(model_dir)
    from src.anomaly_models import load_joblib
    bundle = load_joblib(model_dir / "bundle.joblib")
    meta = json.load(open(model_dir / "metadata.json"))
    train_subjects = set(meta.get("training_subjects", []))
    F = pd.read_csv(fpath)
    run_id = f"{test_name or dataset_id}__{meta['model_id']}__{dt.datetime.now():%Y%m%d_%H%M%S}"
    out = TESTS / run_id
    out.mkdir(parents=True, exist_ok=False)
    previous = sorted(p.name for p in TESTS.glob(f"*{dataset_id}*__{meta['model_id']}__*") if p.name != run_id)
    lock = dict(run_id=run_id, dataset_id=dataset_id, model_id=meta["model_id"], created=dt.datetime.now().isoformat(timespec="seconds"),
                features_sha256=_sha(fpath), manifest_sha256=_sha(DATASETS / dataset_id / "manifest.json"),
                model_sha256=_sha(model_dir / "bundle.joblib"), labels_sha256=_sha(labels_path) if labels_path else None,
                rule="model applied as saved: no refitting, rescaling or threshold changes",
                previous_runs_same_dataset_and_model=previous)
    overlap = sorted(train_subjects & set(F.subject))
    lock["subjects_overlapping_training"] = overlap
    json.dump(lock, open(out / "lock.json", "w"), indent=2)
    p, thr, feats = bundle["pipeline"], bundle["threshold"], bundle["features"]
    rows = []
    for _, r in F.iterrows():
        x = np.array([[r.get(c, np.nan) for c in feats]], float)
        task_ok = r.task == "static"
        score = float(p.score(x)[0])
        oor = bool(p.out_of_range(x)[0])
        z = p.model_.explain(p.imp_.transform(x)[0])
        order = np.argsort(-np.abs(z))[:3]
        cat = categorise(r, score > thr, oor, task_ok)
        rows.append(dict(rec_id=r.rec_id, subject=r.subject, hand=r.hand, task=r.task, score=score, threshold=thr,
                         category=cat, out_of_range=oor,
                         top_features=", ".join(f"{feats[i]} ({z[i]:+.1f})" for i in order),
                         subject_in_training=r.subject in train_subjects,
                         **{f"z_{feats[i]}": float(z[i]) for i in range(len(feats))}))
    P = pd.DataFrame(rows)
    result = dict(lock=lock, n_recordings=len(P), category_counts=P.category.value_counts().to_dict())
    if labels_path:
        L = load_labels(labels_path)
        M, rep = match_labels(L, m["recordings"])
        P = P.merge(M[["rec_id", "label", "label_level", "label_source"]], on="rec_id", how="left")
        result["label_report"] = rep
        ev = P[P.label.isin(["healthy", "arthritis"]) & P.category.isin(["Normal measurement", "Potential anomaly"])]
        result["excluded_from_metrics"] = dict(
            unlabelled_or_other=int((~P.label.isin(["healthy", "arthritis"])).sum()),
            not_scored_quality_or_task=int((P.label.isin(["healthy", "arthritis"]) & ~P.category.isin(["Normal measurement", "Potential anomaly"])).sum()))
        result["metrics_flag_vs_label"] = classification_metrics((ev.label == "arthritis").astype(int),
                                                                 (ev.category == "Potential anomaly").astype(int),
                                                                 ev.score.to_numpy())
        result["metrics_flag_vs_label"]["interpretation"] = ("positive = flagged 'Potential anomaly'; the anomaly score is "
                                                             "a distance from the healthy reference, not a probability")
        result["n_labelled_samples"] = int(len(ev))
    if overlap:
        result["warning"] = f"{len(overlap)} test subject(s) were also in the model's training data: not an independent test"
    P.to_csv(out / "predictions.csv", index=False)
    json.dump(result, open(out / "results.json", "w"), indent=2, default=float)
    return out
