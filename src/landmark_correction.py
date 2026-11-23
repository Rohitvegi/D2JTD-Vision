"""RQ2 follow-up: learn a joint-position correction from the hand-clicked thermal joints.

Finding (annotations, 777 points / 136 frames): mapped RGB joints sit ~10 px towards the wrist of the
clicked joints, along the finger, for every joint type; error across the finger is small. Hand-mask
registration cannot see this (sliding along a finger barely changes mask overlap).

Correction model (scale- and rotation-invariant): move each joint along its own finger direction
(joint -> next distal landmark) by  f_kind x |MCP - PIP|  of that finger.
  * f_MCP, f_PIP are the median along-finger errors / bone length, learned from the clicks.
  * Only MCP and PIP were annotated. Thumb MCP uses f_MCP; thumb IP and the DIP joints use f_PIP
    (extrapolation, not validated).
  * No across-finger term: it gained 0.3 px and its sign would depend on left/right hand.

Honest evaluation: leave-one-subject-out (the correction for a subject is learned without that
subject's clicks). Also reports how much the joint misplacement changes the ROI temperature.

Writes models/landmark_correction.json (used by run_pipeline / edge_demo when present) and
reports/results/rq2_correction.json + rq2_correction_points.csv.

Usage:  .venv312\\Scripts\\python.exe -m src.landmark_correction      (after src.annotate, before
        rerunning src.run_pipeline)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.io_utils import load_config, load_recordings, open_npy
from src.landmarks import JOINTS, RADIUS_PAIR
from src.segmentation import thermal_mask

ANN = Path("data/annotations/joints_clicked.csv")
OUT_MODEL = Path("models/landmark_correction.json")
ANN_LM = {"index_MCP": 5, "index_PIP": 6, "middle_MCP": 9, "middle_PIP": 10, "ring_MCP": 13, "ring_PIP": 14}
CORE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def kind_of(name: str) -> str:
    k = name.split("_")[1]
    return "MCP" if k == "MCP" else "PIP"          # PIP rule also used for IP and DIP (extrapolated)


def load(path: Path = OUT_MODEL) -> dict | None:
    return json.load(open(path)) if path.is_file() else None


def apply(pts: np.ndarray, corr: dict | None) -> np.ndarray:
    """Return a copy of the 21 mapped landmarks with the 14 joint centres moved along their finger."""
    if not corr:
        return pts
    out = pts.copy()
    f = corr["along_frac"]
    for name, lm, finger, _ in JOINTS:
        u = pts[lm + 1] - pts[lm]
        n = np.linalg.norm(u)
        if n < 1e-6:
            continue
        a, b = RADIUS_PAIR[finger]
        seg = np.linalg.norm(pts[b] - pts[a])
        out[lm] = pts[lm] + f[kind_of(name)] * seg * (u / n)
    return out


def _errors(a: pd.DataFrame, proc: Path) -> pd.DataFrame:
    """Signed error of each clicked joint in its finger frame (raw mapped landmarks)."""
    rows = []
    for rec, g in a.groupby("rec_id"):
        d = pd.read_csv(proc / "frames" / f"{rec}.csv").set_index("frame")
        for r in g.itertuples():
            f = d.loc[r.frame]
            lm = ANN_LM[r.joint]
            P = np.array([f[f"lm{lm}_x"], f[f"lm{lm}_y"]])
            Q = np.array([f[f"lm{lm + 1}_x"], f[f"lm{lm + 1}_y"]])
            u = (Q - P) / np.linalg.norm(Q - P)
            v = np.array([-u[1], u[0]])
            a_, b_ = RADIUS_PAIR[r.joint.split("_")[0]]
            seg = float(np.hypot(f[f"lm{b_}_x"] - f[f"lm{a_}_x"], f[f"lm{b_}_y"] - f[f"lm{a_}_y"]))
            e = np.array([r.x, r.y]) - P
            rows.append(dict(rec_id=rec, subject=rec.split("_")[0], frame=int(r.frame), joint=r.joint,
                             kind=kind_of(r.joint), task="static" if rec.endswith("static") else "open_close",
                             x=r.x, y=r.y, px=P[0], py=P[1], ux=u[0], uy=u[1], seg=seg,
                             along=float(e @ u), across=float(e @ v), err_raw=float(np.linalg.norm(e)),
                             radius=f.get(f"{r.joint}_r", np.nan)))
    E = pd.DataFrame(rows)
    E["frac"] = E.along / E.seg
    return E


def loso_corrected(E: pd.DataFrame) -> pd.DataFrame:
    """Add leave-one-subject-out corrected positions (cx, cy) and their error (err_loso).

    The correction for each subject is the median along-finger fraction of the OTHER subjects.
    """
    E = E.copy()
    E["err_loso"] = np.nan
    E["cx"], E["cy"] = np.nan, np.nan
    for s in E.subject.unique():
        tr = E[E.subject != s]
        fk = tr.groupby("kind").frac.median()
        m = E.subject == s
        shift = E.loc[m, "kind"].map(fk).values * E.loc[m, "seg"].values
        cx = E.loc[m, "px"].values + shift * E.loc[m, "ux"].values
        cy = E.loc[m, "py"].values + shift * E.loc[m, "uy"].values
        E.loc[m, "cx"], E.loc[m, "cy"] = cx, cy
        E.loc[m, "err_loso"] = np.hypot(E.loc[m, "x"].values - cx, E.loc[m, "y"].values - cy)
    return E


def _roi_T(T, core, c, r):
    h, w = T.shape
    x0, x1 = max(0, int(c[0] - r)), min(w, int(c[0] + r) + 1)
    y0, y1 = max(0, int(c[1] - r)), min(h, int(c[1] + r) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.nan
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = ((xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r) & core[y0:y1, x0:x1]
    return float(np.median(T[y0:y1, x0:x1][m])) if m.sum() >= 32 else np.nan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config))
    pc = cfg["pipeline"]
    proc, res = Path(pc["processed_dir"]), Path(pc["results_dir"])
    if not ANN.is_file():
        raise SystemExit(f"{ANN} not found: run src.annotate first")
    a = pd.read_csv(ANN)
    E = _errors(a, proc)
    native = int(pc["native_px_per_axis"])

    E = loso_corrected(E)

    # temperature impact of the misplacement (same ROI radius & core mask as the pipeline)
    infos = {i.rec_id: i for i in load_recordings(cfg)}
    dT_raw, dT_cor = [], []
    for (rec, f), g in E.groupby(["rec_id", "frame"]):
        T = np.asarray(open_npy(infos[rec])[f], dtype=np.float32)
        hand, _ = thermal_mask(T)
        core = cv2.erode(hand.astype(np.uint8), CORE) > 0
        for i, r in g.iterrows():
            rad = r.radius if np.isfinite(r.radius) else 0.3 * r.seg
            t_true = _roi_T(T, core, (r.x, r.y), rad)
            E.loc[i, "dT_raw"] = _roi_T(T, core, (r.px, r.py), rad) - t_true
            E.loc[i, "dT_loso"] = _roi_T(T, core, (r.cx, r.cy), rad) - t_true

    def summ(col):
        x = E[col].dropna()
        return dict(median=float(x.median()), p90=float(x.quantile(0.9)))

    fk_all = E.groupby("kind").frac.median().to_dict()
    model = dict(description="Along-finger joint correction learned from hand-clicked thermal joints "
                             "(fraction of MCP-PIP bone length; positive = towards fingertip). "
                             "PIP value also applied to thumb IP and DIP joints (extrapolated).",
                 along_frac={"MCP": float(fk_all["MCP"]), "PIP": float(fk_all["PIP"])},
                 n_points=int(len(E)), n_frames=int(E[["rec_id", "frame"]].drop_duplicates().shape[0]))
    OUT_MODEL.parent.mkdir(exist_ok=True)
    json.dump(model, open(OUT_MODEL, "w"), indent=2)

    report = dict(
        n_points=int(len(E)),
        raw_err_px=summ("err_raw"), loso_corrected_err_px=summ("err_loso"),
        loso_corrected_err_native=float(E.err_loso.median() / native),
        static_loso_err_px=float(E[E.task == "static"].err_loso.median()),
        open_close_loso_err_px=float(E[E.task == "open_close"].err_loso.median()),
        along_bias_px_raw=float(E.along.median()), across_bias_px_raw=float(E.across.median()),
        pass_le_8px=bool(E.err_loso.median() <= 8),
        abs_dT_raw_C={k: float(v) for k, v in E.dT_raw.abs().describe(percentiles=[.5, .9])[["50%", "90%"]].items()},
        abs_dT_corrected_C={k: float(v) for k, v in E.dT_loso.abs().describe(percentiles=[.5, .9])[["50%", "90%"]].items()},
        correction=model["along_frac"])
    res.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(res / "rq2_correction.json", "w"), indent=2)
    E.to_csv(res / "rq2_correction_points.csv", index=False)
    print(json.dumps(report, indent=1))
    print(f"Wrote {OUT_MODEL}. Rerun src.run_pipeline to apply it, then src.analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
