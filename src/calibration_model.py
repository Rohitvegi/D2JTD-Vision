"""Learned RGB -> thermal calibration (v9).

Why: the two FLIR ONE lenses sit side by side, so the shift between the RGB and thermal image (parallax)
grows when the hand is closer to the camera. v3 used ONE shift per recording, which cannot follow the hand
moving in depth (open-close error 13.2 px vs static 8.8 px). Physics: disparity ~ 1/depth and apparent
hand size ~ 1/depth, so the shift is (close to) linear in the hand size seen in RGB.

This module
  1. collects training frames from every recording: MediaPipe landmarks, RGB + thermal hand masks and the
     best shift for that single frame (fit_frame, scale fixed at the rig value) -> data/calibration/,
  2. trains a calibration model  t(frame) = a + B . [hand_size, centre_x, centre_y]  (Huber regression,
     robust to frames where the masks disagree),
  3. evaluates it leave-one-subject-out against the v3 methods with held-out Dice and with the 777
     hand-clicked joints (along-finger correction re-learned per method, also leave-one-subject-out),
  4. auto-calibrates a new camera/dataset: finds the image box, the scale and the shift model from the
     data itself and writes a calibration profile that the pipeline and the upload module apply.

Calibration modes (config pipeline.registration.mode):
  per_recording  v3: one shift per recording (median of 16 frames)
  learned        shift predicted per frame from the hand size (no fitting needed; works for any video)
  hybrid         per-recording shift + learned per-frame change with hand size  (recommended)

Usage:
  .venv312\\Scripts\\python.exe -m src.calibration_model collect --workers 6
  .venv312\\Scripts\\python.exe -m src.calibration_model train
  .venv312\\Scripts\\python.exe -m src.calibration_model auto --dataset <uploaded dataset id>
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from src import landmarks as L
from src import registration as R
from src.io_utils import crop_to_npy_scale, iter_frames, load_config, load_recordings
from src.segmentation import dice, hand_size, rgb_mask, thermal_mask

CAL_DIR = Path("data/calibration")
MODEL_PATH = Path("models/calibration/flirone_one_pro_v9_learned.json")
REPORT_DIR = Path("reports/v9_calibration")
FEATS = ["hand_size", "cx", "cy"]
N_SAMPLE = 24


# --------------------------------------------------------------------------- model
class ShiftModel:
    """t = intercept + coef . features, one robust linear model per axis."""

    def __init__(self, coef=None, intercept=None, feats=FEATS):
        self.feats = list(feats)
        self.coef = None if coef is None else np.asarray(coef, float)
        self.intercept = None if intercept is None else np.asarray(intercept, float)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        co, ic = [], []
        for k in range(2):
            m = HuberRegressor(epsilon=1.5, alpha=1e-3, max_iter=500).fit(X, Y[:, k])
            co.append(m.coef_)
            ic.append(m.intercept_)
        self.coef, self.intercept = np.array(co), np.array(ic)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.atleast_2d(X) @ self.coef.T + self.intercept

    def to_dict(self):
        return dict(features=self.feats, coef=self.coef.tolist(), intercept=self.intercept.tolist())

    @classmethod
    def from_dict(cls, d):
        return cls(d["coef"], d["intercept"], d["features"])


def frame_features(pts_rgb: np.ndarray) -> np.ndarray:
    """Calibration features from the 21 RGB landmarks (crop pixels)."""
    return np.array([hand_size(pts_rgb), float(pts_rgb[:, 0].mean()), float(pts_rgb[:, 1].mean())])


def load_model(path: Path = MODEL_PATH) -> dict | None:
    return json.load(open(path)) if path.is_file() else None


def frame_matrix(cal: dict, k: float, pts_rgb: np.ndarray, rec_shift: tuple[float, float] | None = None,
                 rec_size: float | None = None) -> np.ndarray:
    """RGB-crop -> .npy affine matrix for ONE frame.

    learned: shift predicted from this frame's landmarks.
    hybrid : recording shift + the model's predicted change between this frame and the recording's median hand.
    """
    sm = ShiftModel.from_dict(cal["shift_model"])
    x = frame_features(pts_rgb)
    t = sm.predict(x)[0]
    if cal.get("mode", "learned") == "hybrid" and rec_shift is not None and rec_size is not None:
        x0 = x.copy()
        x0[0] = rec_size
        t = np.asarray(rec_shift) + (t - sm.predict(x0)[0])
    return R.matrix(k, cal["scale"], float(t[0]), float(t[1]), float(cal.get("angle_deg", 0.0)))


# --------------------------------------------------------------------------- 1. collect
def _collect_one(args):
    info, cfg, frames_wanted = args
    pc = cfg["pipeline"]
    box = tuple(pc["video_box"])
    k = crop_to_npy_scale(box)
    s = float(pc["registration"]["fixed_scale"])
    cv2.setRNGSeed(0)
    lm = L.make_landmarker(pc["landmark_model"], video=False, min_conf=pc["min_detection_confidence"])
    rows, rgbm, thm = [], [], []
    for i, rgb, T in iter_frames(info, frames_wanted, box):
        pts, _ = L.detect(lm, rgb)
        if pts is None:
            continue
        pts, fl = L.check_and_fix(pts, rgb.shape[:2])
        if not fl["ok"]:
            continue
        th, _ = thermal_mask(T)
        rm = rgb_mask(rgb, pts)
        best = R.fit_frame(rm, th, k, np.array([s]))
        x = frame_features(pts)
        row = dict(rec_id=info.rec_id, subject=info.rec.subject, hand=info.rec.hand, task=info.rec.task, frame=int(i),
                   hand_size=x[0], cx=x[1], cy=x[2], tx=best["tx"], ty=best["ty"], dice_best=best["dice"],
                   openness=L.openness(pts), rgb_area=float(rm.mean()), th_area=float(th.mean()))
        for j in range(21):
            row[f"lm{j}_x"], row[f"lm{j}_y"] = float(pts[j, 0]), float(pts[j, 1])
        rows.append(row)
        rgbm.append(np.packbits(rm))
        thm.append(np.packbits(th))
    lm.close()
    if rows:
        np.savez_compressed(CAL_DIR / "masks" / f"{info.rec_id}.npz", frames=np.array([r["frame"] for r in rows]),
                            rgb=np.stack(rgbm), th=np.stack(thm), rgb_shape=rm.shape, th_shape=th.shape)
    return rows


def collect(cfg: dict, workers: int) -> pd.DataFrame:
    (CAL_DIR / "masks").mkdir(parents=True, exist_ok=True)
    ann = pd.read_csv("data/annotations/joints_clicked.csv")
    infos = load_recordings(cfg)
    jobs = []
    for info in infos:
        frames = set(np.linspace(0, info.n_frames - 1, N_SAMPLE).round().astype(int).tolist())
        frames |= set(ann[ann.rec_id == info.rec_id].frame.astype(int).tolist())
        jobs.append((info, cfg, sorted(frames)))
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, r in enumerate(ex.map(_collect_one, jobs), 1):
            rows += r
            print(f"[{n}/{len(jobs)}] {len(rows)} frames, {time.time() - t0:.0f} s", flush=True)
    F = pd.DataFrame(rows)
    F.to_csv(CAL_DIR / "frames.csv", index=False)
    return F


def _masks(rec_id: str):
    z = np.load(CAL_DIR / "masks" / f"{rec_id}.npz")
    rs, ts = tuple(z["rgb_shape"]), tuple(z["th_shape"])
    out = {}
    for f, a, b in zip(z["frames"], z["rgb"], z["th"]):
        out[int(f)] = (np.unpackbits(a)[: rs[0] * rs[1]].reshape(rs).astype(bool),
                       np.unpackbits(b)[: ts[0] * ts[1]].reshape(ts).astype(bool))
    return out


# --------------------------------------------------------------------------- 2+3. train and evaluate
def _shifts(F: pd.DataFrame, reg: pd.DataFrame, rig) -> dict[str, np.ndarray]:
    """Shift (tx, ty) of every frame under each method, all leave-one-subject-out where learned."""
    good = F.dice_best >= 0.75                                          # training frames: masks agree
    out = {m: np.full((len(F), 2), np.nan) for m in ("rig", "per_recording", "learned", "hybrid", "oracle")}
    out["rig"][:] = rig[1:]
    rr = reg.set_index("rec_id")
    out["per_recording"] = np.c_[F.rec_id.map(rr.tx), F.rec_id.map(rr.ty)]
    out["oracle"] = F[["tx", "ty"]].to_numpy()
    med_size = F.groupby("rec_id").hand_size.transform("median").to_numpy()
    X = F[FEATS].to_numpy()
    for s in F.subject.unique():
        tr, te = (F.subject != s).to_numpy() & good.to_numpy(), (F.subject == s).to_numpy()
        m = ShiftModel().fit(X[tr], F.loc[tr, ["tx", "ty"]].to_numpy())
        out["learned"][te] = m.predict(X[te])
        X0 = X[te].copy()
        X0[:, 0] = med_size[te]
        out["hybrid"][te] = out["per_recording"][te] + m.predict(X[te]) - m.predict(X0)
    return out


def _dice_eval(F, shifts, k, s, test_mask):
    rows = []
    for rec, g in F[test_mask].groupby("rec_id"):
        masks = _masks(rec)
        for i in g.index:
            rm, th = masks[int(F.frame[i])]
            r = dict(rec_id=rec, task=F.task[i], frame=int(F.frame[i]))
            for meth, S in shifts.items():
                r[meth] = dice(R.warp_mask(rm, R.matrix(k, s, *S[i])), th)
            rows.append(r)
    return pd.DataFrame(rows)


def _joint_eval(F, shifts, k, s):
    """Euclidean error of mapped (and LOSO-corrected) joints vs the 777 clicked joints, per method."""
    from src.landmark_correction import ANN_LM, kind_of, loso_corrected
    ann = pd.read_csv("data/annotations/joints_clicked.csv")
    Fi = F.reset_index().set_index(["rec_id", "frame"])
    res = {}
    for meth, S in shifts.items():
        rows = []
        for r in ann.itertuples():
            key = (r.rec_id, int(r.frame))
            if key not in Fi.index:
                continue
            row = Fi.loc[key]
            i = int(row["index"])
            pts = np.array([[row[f"lm{j}_x"], row[f"lm{j}_y"]] for j in range(21)])
            P21 = R.map_points(pts, R.matrix(k, s, *S[i]))
            lm = ANN_LM[r.joint]
            P, Q = P21[lm], P21[lm + 1]
            u = (Q - P) / np.linalg.norm(Q - P)
            a_, b_ = L.RADIUS_PAIR[r.joint.split("_")[0]]
            seg = float(np.linalg.norm(P21[b_] - P21[a_]))
            e = np.array([r.x, r.y]) - P
            rows.append(dict(rec_id=r.rec_id, subject=r.rec_id.split("_")[0], frame=int(r.frame), joint=r.joint,
                             kind=kind_of(r.joint), task="static" if r.rec_id.endswith("static") else "open_close",
                             x=r.x, y=r.y, px=P[0], py=P[1], ux=u[0], uy=u[1], seg=seg, along=float(e @ u),
                             err_raw=float(np.linalg.norm(e))))
        E = pd.DataFrame(rows)
        E["frac"] = E.along / E.seg
        res[meth] = loso_corrected(E)
    return res


def train(cfg: dict) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    F = pd.read_csv(CAL_DIR / "frames.csv").reset_index(drop=True)
    reg = pd.read_csv("data/processed_v8/registration.csv")
    pc = cfg["pipeline"]
    k, s, rig = crop_to_npy_scale(tuple(pc["video_box"])), float(pc["registration"]["fixed_scale"]), pc["registration"]["rig_default"]
    ann = pd.read_csv("data/annotations/joints_clicked.csv")
    ann_keys = set(zip(ann.rec_id, ann.frame.astype(int)))
    is_ann = np.array([(r, int(f)) in ann_keys for r, f in zip(F.rec_id, F.frame)])

    # physics check: does the per-frame shift follow the hand size?
    corr = {ax: float(np.corrcoef(F.hand_size[F.dice_best >= 0.75], F[ax][F.dice_best >= 0.75])[0, 1]) for ax in ("tx", "ty")}
    within = F[F.dice_best >= 0.75].groupby("rec_id").apply(
        lambda g: pd.Series({ax: np.corrcoef(g.hand_size, g[ax])[0, 1] if g.hand_size.std() > 1 else np.nan for ax in ("tx", "ty")}),
        include_groups=False)

    shifts = _shifts(F, reg, rig)
    D = _dice_eval(F, shifts, k, s, ~is_ann)                      # Dice on frames not used for joint scoring
    J = _joint_eval(F, shifts, k, s)
    meths = ["rig", "per_recording", "learned", "hybrid", "oracle"]
    summ = []
    for m in meths:
        E = J[m]
        summ.append(dict(method=m,
                         dice_median=float(D[m].median()), dice_p10=float(D[m].quantile(0.1)),
                         dice_median_static=float(D[D.task == "static"][m].median()),
                         dice_median_open_close=float(D[D.task == "open_close"][m].median()),
                         joint_err_raw_median=float(E.err_raw.median()),
                         joint_err_corrected_median=float(E.err_loso.median()),
                         joint_err_corrected_static=float(E[E.task == "static"].err_loso.median()),
                         joint_err_corrected_open_close=float(E[E.task == "open_close"].err_loso.median()),
                         joint_err_corrected_p90=float(E.err_loso.quantile(0.9)), n_points=int(len(E))))
    S = pd.DataFrame(summ)
    S.to_csv(REPORT_DIR / "method_comparison.csv", index=False)
    D.to_csv(REPORT_DIR / "dice_per_frame.csv", index=False)
    # paired test: hybrid vs per_recording on the same clicked joints
    from scipy.stats import wilcoxon
    a, b = J["per_recording"].err_loso.to_numpy(), J["hybrid"].err_loso.to_numpy()
    a2, b2 = J["per_recording"].err_loso.to_numpy(), J["learned"].err_loso.to_numpy()
    tests = dict(hybrid_vs_per_recording_p=float(wilcoxon(a, b).pvalue), learned_vs_per_recording_p=float(wilcoxon(a2, b2).pvalue),
                 hybrid_better_share=float(np.mean(b < a)))

    # final model on all subjects
    good = F.dice_best >= 0.75
    final = ShiftModel().fit(F.loc[good, FEATS].to_numpy(), F.loc[good, ["tx", "ty"]].to_numpy())
    best = S.set_index("method").loc[["per_recording", "learned", "hybrid"]].joint_err_corrected_median.idxmin()
    # correction fraction for the chosen mode, learned on all subjects
    Eb = J[best]
    frac = {kk: float(v) for kk, v in Eb.groupby("kind").frac.median().items()}
    cal = dict(name="flirone_one_pro_v9_learned", created=time.strftime("%Y-%m-%dT%H:%M:%S"),
               camera="FLIR ONE Pro (smartphone add-on)", mode=best, scale=s, fixed_scale=s, video_box=pc["video_box"],
               video_hw=[480, 640], npy_hw=[640, 480], rig_default=rig, shift_model=final.to_dict(),
               along_frac=frac, trained_on=dict(frames=int(good.sum()), recordings=int(F.rec_id.nunique()), subjects=int(F.subject.nunique())),
               validation=dict(comparison=summ, tests=tests),
               note="learned from hand-mask overlap of the FLIROne dataset; not a target-based stereo calibration")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cal, open(MODEL_PATH, "w"), indent=1)
    out = dict(n_frames=len(F), n_train_frames=int(good.sum()), corr_size_shift=corr,
               within_recording_corr_median={ax: float(within[ax].median()) for ax in ("tx", "ty")},
               comparison=summ, tests=tests, chosen_mode=best, model=final.to_dict())
    json.dump(out, open(REPORT_DIR / "summary.json", "w"), indent=1)
    _figure(F, S, final, D)
    return out


def _figure(F, S, model, D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    g = F[F.dice_best >= 0.75]
    for task, c in (("static", "#2a78d6"), ("open_close", "#eb6834")):
        h = g[g.task == task]
        ax[0].scatter(h.hand_size, h.ty, s=6, alpha=0.4, color=c, label=task)
    xs = np.linspace(g.hand_size.min(), g.hand_size.max(), 50)
    X = np.c_[xs, np.full(50, g.cx.median()), np.full(50, g.cy.median())]
    ax[0].plot(xs, model.predict(X)[:, 1], color="black", lw=2, label="learned model")
    ax[0].set_xlabel("hand size in RGB (wrist–middle MCP, px)")
    ax[0].set_ylabel("best vertical shift ty (px)")
    ax[0].set_title("Shift vs hand distance: almost constant (rigid rig)")
    ax[0].legend(frameon=False)
    names = {"rig": "one shift\nfor all", "per_recording": "per recording\n(v3)", "learned": "learned\n(v9, chosen)",
             "hybrid": "hybrid", "oracle": "per-frame fit\n(upper bound)"}
    cols = ["#9aa7b8", "#9aa7b8", "#2a78d6", "#1e8a4c", "#6b6a66"]
    ax[1].bar([names[m] for m in S.method], S.joint_err_corrected_median, color=cols)
    for i, v in enumerate(S.joint_err_corrected_median):
        ax[1].text(i, v, f"{v:.2f}", ha="center", va="bottom")
    ax[1].axhline(8, color="#d03b3b", ls="--", lw=1)
    ax[1].text(4.4, 8.1, "8 px target", color="#d03b3b", ha="right", fontsize=9)
    ax[1].set_ylabel("median joint error vs clicks (px, LOSO)")
    ax[1].set_title("Joint position error (777 clicked joints)")
    ax[2].bar([names[m] for m in S.method], S.dice_median, color=cols)
    for i, v in enumerate(S.dice_median):
        ax[2].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax[2].set_ylim(0.8, 1.0)
    ax[2].set_ylabel("median Dice (held-out frames)")
    ax[2].set_title("Hand-outline overlap")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "fig_calibration_model.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- 4. auto-calibrate a new camera / dataset
def auto_calibrate(infos, cfg: dict, name: str, n_frames: int = 60) -> dict:
    """Find the calibration of an unknown camera/dataset from its own recordings (see src.calibration)."""
    from src.calibration import find_calibration
    cal = find_calibration([(i.rgb_path, i.npy_path) for i in infos], name, n_frames=n_frames,
                           landmark_model=cfg["pipeline"]["landmark_model"])
    prof = cal.to_profile()
    prof["self_check"] = cal.quality
    return prof


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["collect", "train", "auto"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dataset", help="uploaded dataset id (auto)")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(Path(a.config))
    if a.cmd == "collect":
        F = collect(cfg, a.workers)
        print(f"collected {len(F)} frames from {F.rec_id.nunique()} recordings -> {CAL_DIR / 'frames.csv'}")
    elif a.cmd == "train":
        out = train(cfg)
        print(pd.DataFrame(out["comparison"]).round(3).to_string(index=False))
        print("tests:", out["tests"], "\nchosen mode:", out["chosen_mode"], "\nsaved", MODEL_PATH)
    else:
        from src.dataset_manager import infos_from_manifest, load_manifest, save_calibration
        m = load_manifest(a.dataset)
        infos = infos_from_manifest(m, [r["rec_id"] for r in m["recordings"]])
        prof = auto_calibrate(infos, cfg, f"auto_{a.dataset}")
        print(json.dumps(prof["self_check"], indent=1))
        print("saved", save_calibration(prof))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
