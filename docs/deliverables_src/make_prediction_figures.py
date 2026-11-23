"""Correct-prediction figures for the report and slides.

Every hand is scored by a model that never saw that person (nested leave-one-subject-out, the same procedure
as the v7 evaluation) with the application's default model: Ledoit-Wolf Mahalanobis on D2JTD + spatial
features (32). Real recordings are healthy volunteers -> the correct output is "within healthy range".
Simulated warming (+2 degC on one joint ROI, frame level) -> the correct output is "potential anomaly".

Run from the project root:
  .venv312\\Scripts\\python.exe docs/deliverables_src/make_prediction_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path.cwd()))
from src.anomaly_models import Pipeline, nested_threshold  # noqa: E402
from src.feature_extraction import recording_features, warm_frames  # noqa: E402
from src.io_utils import load_config  # noqa: E402
from src import joint_view  # noqa: E402

MODEL = "mahalanobis_ledoit_wolf"
META = json.load(open("models/registry/v7_mahalanobis_ledoit_wolf_20261121_172656/metadata.json"))
FEATS = META["features"]
FIG = Path("deliverables/figures")
SUP = Path("deliverables/supporting")
TMP = FIG / "_joint_views"
WARM_JOINT, WARM_D = "index_MCP", 2.0
GREEN, RED, NAVY, GREY = "#1e8a4c", "#d03b3b", "#1B2A41", "#6b6a66"


def frames(rec):
    return pd.read_csv(f"data/processed_v6/frames/{rec}.csv")


def evaluate():
    F = pd.read_csv("reports/v6_features/features.csv")
    F = F[F.task == "static"].reset_index(drop=True)
    X, g = F[FEATS].to_numpy(float), F.subject.to_numpy()
    rows = []
    for s in np.unique(g):
        tr, te = g != s, g == s
        thr = nested_threshold(X[tr], g[tr], MODEL)
        p = Pipeline(MODEL).fit(X[tr])
        for i in np.where(te)[0]:
            rec = F.rec_id[i]
            fw = recording_features(warm_frames(frames(rec), [WARM_JOINT], WARM_D), "static")
            xw = np.array([[fw.get(c, np.nan) for c in FEATS]], float)
            sr, sw = float(p.score(X[[i]])[0]), float(p.score(xw)[0])
            z = p.model_.explain(p.imp_.transform(xw)[0])
            rows.append(dict(rec_id=rec, subject=s, hand=F.hand[i], threshold=thr, score_real=sr, score_warmed=sw,
                             real_correct=sr <= thr, warmed_correct=sw > thr,
                             top_feature_warmed=FEATS[int(np.nanargmax(np.abs(z)))]))
    R = pd.DataFrame(rows)
    SUP.mkdir(parents=True, exist_ok=True)
    R.to_csv(SUP / "prediction_examples.csv", index=False)
    return R


def gallery(R, cfg):
    ok = R[R.real_correct & R.warmed_correct].copy()
    ok["margin"] = ok.threshold - ok.score_real
    pick = ok.sort_values("margin", ascending=False).drop_duplicates("subject").head(6)
    TMP.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    for ax, r in zip(axes.flat, pick.itertuples()):
        png, J = joint_view.render(r.rec_id, Path("data/processed_v8"), cfg, out_png=TMP / f"{r.rec_id}.png")
        im = Image.open(png)
        w, h = im.size
        ax.imshow(im.crop((0, int(0.04 * h), int(0.5 * w), h)))
        ax.axis("off")
        n_vis = int((J.visible == "yes").sum())
        ax.set_title(f"{r.subject} {r.hand} hand — {n_vis}/14 joints measured\n"
                     f"score {r.score_real:.2f} ≤ threshold {r.threshold:.2f} → within healthy range ✓",
                     fontsize=11, color=GREEN)
    fig.suptitle("Correct outputs on unseen healthy volunteers (model never trained on the person shown)", fontsize=14, color=NAVY)
    fig.tight_layout()
    fig.savefig(FIG / "P01_correct_predictions_gallery.png", dpi=130)
    plt.close(fig)
    return pick


def detection(R, cfg, rec):
    r = R.set_index("rec_id").loc[rec]
    png = TMP / f"{rec}.png"
    im = Image.open(png)
    w, h = im.size
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.8, 1.0])
    ax = fig.add_subplot(gs[0])
    ax.imshow(im.crop((int(0.5 * w), int(0.04 * h), w, h)))
    ax.axis("off")
    ax.set_title("Measured thermal frame with the 14 joint ROIs", fontsize=11)
    ax = fig.add_subplot(gs[1])
    vals = [r.score_real, r.score_warmed]
    bars = ax.bar(["real recording\n(healthy)", f"index MCP +{WARM_D:.0f} °C\n(simulated)"], vals, color=[GREEN, RED], width=0.6)
    ax.axhline(r.threshold, color=NAVY, ls="--", lw=1.5)
    ax.text(1.42, r.threshold, f"threshold\n{r.threshold:.2f}", va="center", fontsize=9, color=NAVY)
    for b, v, lab in zip(bars, vals, ["within range ✓", "anomaly flagged ✓"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}\n{lab}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_ylabel("anomaly score (Mahalanobis distance)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Model output", fontsize=11)
    # per-joint explanation of the warmed case
    F = pd.read_csv("reports/v6_features/features.csv")
    F = F[F.task == "static"].reset_index(drop=True)
    tr = F.subject != r.subject
    p = Pipeline(MODEL).fit(F.loc[tr, FEATS].to_numpy(float))
    fw = recording_features(warm_frames(frames(rec), [WARM_JOINT], WARM_D), "static")
    xw = np.array([[fw.get(c, np.nan) for c in FEATS]], float)
    z = p.model_.explain(p.imp_.transform(xw)[0])[:14]
    names = [c.replace("_D2JTD", "").replace("_", " ") for c in FEATS[:14]]
    ax = fig.add_subplot(gs[2])
    cols = [RED if n == WARM_JOINT.replace("_", " ") else "#9aa7b8" for n in names]
    ax.barh(names[::-1], z[::-1], color=cols[::-1])
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("deviation from healthy mean (SD units)")
    ax.set_title("Explanation: which joint drives the flag", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{r.subject} {r.hand} hand (unseen subject): healthy recording passes, simulated warm joint is flagged and located",
                 fontsize=13, color=NAVY)
    fig.tight_layout()
    fig.savefig(FIG / "P02_simulated_detection_example.png", dpi=130)
    plt.close(fig)


def overview(R):
    fig, ax = plt.subplots(figsize=(12, 5.2))
    R = R.sort_values("score_real").reset_index(drop=True)
    x = np.arange(len(R))
    ax.scatter(x, R.score_real / R.threshold, color=GREEN, s=45, label=f"real healthy recording — correct {int(R.real_correct.sum())}/{len(R)}", zorder=3)
    ax.scatter(x, R.score_warmed / R.threshold, color=RED, s=45, marker="^",
               label=f"same hand, index MCP +{WARM_D:.0f} °C (simulated) — flagged {int(R.warmed_correct.sum())}/{len(R)}", zorder=3)
    for i in x:
        ax.plot([i, i], [R.score_real[i] / R.threshold[i], R.score_warmed[i] / R.threshold[i]], color="#c8c4bc", lw=1)
    ax.axhline(1, color=NAVY, ls="--", lw=1.4)
    ax.text(len(R) - 0.5, 1.03, "threshold", ha="right", color=NAVY, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}{h[0].upper()}" for s, h in zip(R.subject, R.hand)], rotation=90, fontsize=8)
    ax.set_ylabel("score / threshold  (> 1 = flagged)")
    ax.set_title("Every hand scored by a model that never saw that person (nested leave-one-subject-out, 34 hands)", fontsize=12, color=NAVY)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "P03_prediction_overview.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    cfg = load_config(Path("config.yaml"))
    R = evaluate()
    print(R[["real_correct", "warmed_correct"]].sum().to_dict(), "of", len(R))
    pick = gallery(R, cfg)
    detection(R, cfg, pick.rec_id.iloc[0])
    overview(R)
    json.dump(dict(n_hands=len(R), real_correct=int(R.real_correct.sum()), warmed_correct=int(R.warmed_correct.sum()),
                   warm=f"{WARM_JOINT} +{WARM_D} C (simulated, frame level)", model=META["model_id"],
                   gallery=list(pick.rec_id), detection_example=pick.rec_id.iloc[0],
                   top_feature_is_warmed_joint=int((R[R.warmed_correct].top_feature_warmed.str.startswith(WARM_JOINT)).sum())),
              open(SUP / "prediction_examples_summary.json", "w"), indent=1)
    print(open(SUP / "prediction_examples_summary.json").read())
