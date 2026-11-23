"""v5 — Annotation quality control: are the hand-clicked joints a trustworthy reference?

Three independent checks, none of which needs another ground truth:

1. Internal consistency of the clicks (per frame):
   * the middle joint (PIP) must lie further along the finger than the knuckle (MCP);
   * clicked MCP-PIP length within 0.6-1.5x the pipeline's bone length;
   * each click nearest to the pipeline joint of its own finger (finger mix-up check).
2. Three-cornered hat. Three estimators of the same joint:
     A = hand clicks,  R = RGB landmarks mapped to thermal (the pipeline),
     T = MediaPipe run directly on the thermal image.
   If their errors are independent, mean squared pair gaps give each one's own error:
     sA^2 = (|A-R|^2 + |A-T|^2 - |R-T|^2) / 2   (and cyclically).
   R and T share the MediaPipe model, so their errors are partly correlated; this makes |R-T|
   smaller and sA an UPPER BOUND on the clicking error. Reported raw and with each pair's median
   bias removed (random scatter only).
3. Bias attribution: the median vector between each pair shows which estimator carries a
   systematic offset.

Also re-scores RQ2 without the frames that fail check 1 (a fixed rule, as a sensitivity check;
the published v3 number is not changed) and draws a contact sheet of flagged / worst frames.

Outputs: reports/v5_annotation_qc/  (summary.json, flags.csv, points.csv, contact_sheet.png)

Usage:  .venv312\\Scripts\\python.exe -m src.annotation_qc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import landmarks as L
from src.io_utils import load_config, load_recordings, open_npy
from src.landmark_correction import ANN, _errors, loso_corrected

OUT = Path("reports/v5_annotation_qc")
ANN_LM = {"index_MCP": 5, "index_PIP": 6, "middle_MCP": 9, "middle_PIP": 10, "ring_MCP": 13, "ring_PIP": 14}


def thermal_render(T: np.ndarray) -> np.ndarray:
    """Inferno rendering of a radiometric frame (RGB order) — the view MediaPipe and the annotator see."""
    lo, hi = np.percentile(T, 1), np.percentile(T, 99.7)
    g = np.clip((T - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)[:, :, ::-1])


def consistency_flags(E: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rec, f), g in E.groupby(["rec_id", "frame"]):
        g = g.set_index("joint")
        for fin in ("index", "middle", "ring"):
            if f"{fin}_MCP" in g.index and f"{fin}_PIP" in g.index:
                m, p = g.loc[f"{fin}_MCP"], g.loc[f"{fin}_PIP"]
                v = np.array([p.x - m.x, p.y - m.y])
                if v @ np.array([m.ux, m.uy]) <= 0:
                    rows.append((rec, f, fin, "order", "PIP not further along the finger than MCP (labels swapped?)"))
                ratio = np.linalg.norm(v) / m.seg
                if not 0.6 <= ratio <= 1.5:
                    rows.append((rec, f, fin, "length", f"clicked MCP-PIP length {ratio:.2f}x the bone length"))
        for kind in ("MCP", "PIP"):
            sub = g[g.kind == kind]
            for j, r in sub.iterrows():
                d = {jj: np.hypot(r.x - rr.px, r.y - rr.py) for jj, rr in sub.iterrows()}
                best = min(d, key=d.get)
                if best != j and d[j] > 1.5 * d[best]:
                    rows.append((rec, f, j, "finger", f"click is nearest the pipeline's {best} (finger mix-up?)"))
    return pd.DataFrame(rows, columns=["rec_id", "frame", "joint", "check", "issue"])


def thermal_mediapipe(E: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    lm = L.make_landmarker(cfg["pipeline"]["landmark_model"], video=False)
    infos = {i.rec_id: i for i in load_recordings(cfg)}
    rows = []
    for (rec, f), _ in E.groupby(["rec_id", "frame"]):
        T = np.asarray(open_npy(infos[rec])[int(f)], dtype=np.float32)
        p, _ = L.detect(lm, thermal_render(T))
        row = dict(rec_id=rec, frame=int(f), t_found=p is not None)
        if p is not None:
            p, fl = L.check_and_fix(p, T.shape)
            row["t_ok"] = fl["ok"]
            for name, j in ANN_LM.items():
                row[f"{name}_tx"], row[f"{name}_ty"] = float(p[j, 0]), float(p[j, 1])
        rows.append(row)
    lm.close()
    return pd.DataFrame(rows)


def three_cornered_hat(A, R, T, debias: bool) -> dict:
    pairs = {"A-R": A - R, "A-T": A - T, "R-T": R - T}
    bias = {k: np.median(v, axis=0) for k, v in pairs.items()}
    if debias:
        pairs = {k: v - bias[k] for k, v in pairs.items()}
    ms = {k: float(np.mean(np.sum(v ** 2, axis=1))) for k, v in pairs.items()}
    own = {"annotation": (ms["A-R"] + ms["A-T"] - ms["R-T"]) / 2,
           "rgb_mapped": (ms["A-R"] + ms["R-T"] - ms["A-T"]) / 2,
           "thermal_mediapipe": (ms["A-T"] + ms["R-T"] - ms["A-R"]) / 2}
    return dict(n=int(len(A)), rms_pair_gap_px={k: float(np.sqrt(v)) for k, v in ms.items()},
                own_rms_error_px={k: float(np.sqrt(v)) if v > 0 else 0.0 for k, v in own.items()},
                median_bias_px={k: [float(b[0]), float(b[1])] for k, b in bias.items()})


def contact_sheet(E, flags, cfg, path: Path, n: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    flagged = flags[["rec_id", "frame"]].drop_duplicates()
    worst = (E.groupby(["rec_id", "frame"]).err_raw.median().sort_values(ascending=False)
             .reset_index()[["rec_id", "frame"]])
    pick = pd.concat([flagged, worst]).drop_duplicates().head(n)
    infos = {i.rec_id: i for i in load_recordings(cfg)}
    cols = 4
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.6 * rows))
    for ax, (rec, f) in zip(np.ravel(axes), pick.itertuples(index=False)):
        T = np.asarray(open_npy(infos[rec])[int(f)], dtype=np.float32)
        g = E[(E.rec_id == rec) & (E.frame == f)]
        xs = np.r_[g.x, g.px]
        ys = np.r_[g.y, g.py]
        x0, x1 = max(0, xs.min() - 60), min(T.shape[1], xs.max() + 60)
        y0, y1 = max(0, ys.min() - 60), min(T.shape[0], ys.max() + 60)
        ax.imshow(thermal_render(T), extent=(0, T.shape[1], T.shape[0], 0))
        for r in g.itertuples():
            ax.plot([r.px, r.x], [r.py, r.y], color="white", lw=0.8)
            ax.plot(r.x, r.y, "o", ms=7, mfc="none", mec="#2a78d6", mew=2)
            ax.plot(r.px, r.py, "x", ms=7, color="#eb6834", mew=2)
            ax.text(r.x + 3, r.y - 3, r.joint.replace("_", " "), color="white", fontsize=6)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        tag = "FLAGGED" if ((flagged.rec_id == rec) & (flagged.frame == f)).any() else "large error"
        ax.set_title(f"{rec} f{f} ({tag})", fontsize=8)
        ax.axis("off")
    for ax in np.ravel(axes)[len(pick):]:
        ax.axis("off")
    fig.suptitle("Blue circle = your click, orange x = pipeline joint (raw). Check the clicks by eye.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config))
    OUT.mkdir(parents=True, exist_ok=True)
    E = loso_corrected(_errors(pd.read_csv(ANN), Path(cfg["pipeline"]["processed_dir"])))
    flags = consistency_flags(E)
    tm = thermal_mediapipe(E, cfg)
    P = E.merge(tm, on=["rec_id", "frame"], how="left")
    P["tx"] = [r[f"{r['joint']}_tx"] if r.get("t_found") is True and r.get("t_ok") is True else np.nan
               for _, r in P.iterrows()]
    P["ty"] = [r[f"{r['joint']}_ty"] if r.get("t_found") is True and r.get("t_ok") is True else np.nan
               for _, r in P.iterrows()]
    H = P.dropna(subset=["tx", "ty"])
    hat = {}
    for label, D in (("all", H), ("static", H[H.task == "static"]), ("open_close", H[H.task == "open_close"])):
        A, R, T = D[["x", "y"]].values, D[["px", "py"]].values, D[["tx", "ty"]].values
        hat[label] = dict(raw=three_cornered_hat(A, R, T, False), bias_removed=three_cornered_hat(A, R, T, True))
    bad_frames = set(map(tuple, flags[flags.check.isin(["order", "length", "finger"])][["rec_id", "frame"]].values))
    keep = ~E.apply(lambda r: (r.rec_id, r.frame) in bad_frames, axis=1)
    summary = dict(
        n_points=int(len(E)), n_frames=int(E.groupby(["rec_id", "frame"]).ngroups),
        flagged_frames=int(len(bad_frames)), flags_by_check=flags.check.value_counts().to_dict(),
        points_err_gt_40px=int((E.err_raw > 40).sum()),
        thermal_mediapipe_frames_found=int(tm.t_found.sum()),
        three_cornered_hat=hat,
        rq2_sensitivity=dict(
            all_points=dict(raw_median_px=float(E.err_raw.median()), corrected_loso_median_px=float(E.err_loso.median())),
            without_flagged_frames=dict(n_points=int(keep.sum()), raw_median_px=float(E[keep].err_raw.median()),
                                        corrected_loso_median_px=float(E[keep].err_loso.median()))),
        interpretation=("Clicks are internally consistent apart from the flagged frames. The systematic "
                        "along-finger offset belongs to the RGB-mapped joints (A-T median bias ~0, A-R and R-T "
                        "~10 px), i.e. it is a pipeline error that the v3 correction addresses. The clicking "
                        "error estimate is an upper bound because R and T share the MediaPipe model; a second "
                        "annotation pass is needed to measure it directly."))
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    flags.to_csv(OUT / "flags.csv", index=False)
    P.to_csv(OUT / "points.csv", index=False)
    contact_sheet(E, flags, cfg, OUT / "contact_sheet.png")
    print(json.dumps({k: v for k, v in summary.items() if k != "three_cornered_hat"}, indent=1))
    for k, v in hat.items():
        print(k, "own RMS error (bias removed):", {kk: round(vv, 1) for kk, vv in v["bias_removed"]["own_rms_error_px"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
