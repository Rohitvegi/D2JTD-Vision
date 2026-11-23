"""v8 — Report generation: PDF (matplotlib PDF backend, no extra dependency), Excel and CSV.

Every report states the number of samples behind each number and includes a limitations section.
"""
from __future__ import annotations

import datetime as dt
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

LIMITATIONS = [
    "Research and screening-support tool; not a clinically validated diagnostic device.",
    "Reference model built from 17 healthy volunteers (FLIROne dataset); no arthritis patients were available.",
    "Anomaly scores are distances from the healthy reference, not probabilities of arthritis.",
    "Sensitivity figures from the development data are from SIMULATED warming, not patients.",
    "RGB-to-thermal mapping is a fixed-scale similarity transform, not a full stereo calibration.",
    "Repeatability was measured within one session; between-day repeatability is unknown.",
    "Absolute temperatures depend on room conditions and acclimatisation, which were not recorded.",
    "Speed was measured on a laptop CPU, not on an edge device.",
]
INK, INK2 = "#0b0b0b", "#52514e"


def export_tables(tables: dict[str, pd.DataFrame], out_dir: Path, stem: str) -> dict:
    """One CSV per table + one Excel workbook with a sheet per table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in tables.items():
        p = out_dir / f"{stem}_{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    xl = out_dir / f"{stem}.xlsx"
    with pd.ExcelWriter(xl, engine="openpyxl") as w:
        for name, df in tables.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
    paths["excel"] = xl
    return paths


class PDF:
    """Small helper: text pages, tables (paginated) and images on A4 portrait."""

    def __init__(self, path: Path, title: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pdf = PdfPages(self.path)
        self.title = title
        self.page = 0

    def _new(self, landscape: bool = False):
        fig = plt.figure(figsize=(11.69, 8.27) if landscape else (8.27, 11.69))
        self.page += 1
        fig.text(0.07, 0.975, self.title, fontsize=8, color=INK2)
        fig.text(0.93, 0.975, f"page {self.page}", fontsize=8, color=INK2, ha="right")
        return fig

    def text(self, heading: str, paragraphs: list[str]):
        fig = self._new()
        fig.text(0.07, 0.93, heading, fontsize=15, weight="bold", color=INK)
        y = 0.89
        for para in paragraphs:
            for line in textwrap.wrap(para, 95) or [""]:
                fig.text(0.07, y, line, fontsize=9.5, color=INK, family="DejaVu Sans")
                y -= 0.019
                if y < 0.06:
                    self.pdf.savefig(fig)
                    plt.close(fig)
                    fig = self._new()
                    y = 0.93
            y -= 0.012
        self.pdf.savefig(fig)
        plt.close(fig)

    def table(self, heading: str, df: pd.DataFrame, rows_per_page: int = 32, max_cols: int = 8, fmt: str = "{:.3g}"):
        df = df.iloc[:, :max_cols].copy()
        for c in df.columns:
            df[c] = df[c].map(lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                              ("" if isinstance(v, float) else str(v)[:28]))
        for start in range(0, max(len(df), 1), rows_per_page):
            fig = self._new()
            fig.text(0.07, 0.93, heading + (f" (rows {start + 1}-{min(start + rows_per_page, len(df))} of {len(df)})" if len(df) > rows_per_page else f" (n = {len(df)})"),
                     fontsize=12, weight="bold", color=INK)
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.85])
            ax.axis("off")
            part = df.iloc[start:start + rows_per_page]
            if len(part):
                t = ax.table(cellText=part.values, colLabels=[str(c)[:22] for c in part.columns], loc="upper center", cellLoc="left")
                t.auto_set_font_size(False)
                t.set_fontsize(7)
                t.scale(1, 1.25)
            self.pdf.savefig(fig)
            plt.close(fig)

    def image(self, heading: str, png: Path, caption: str = ""):
        if not Path(png).is_file():
            return
        img = plt.imread(png)
        wide = img.shape[1] > 1.2 * img.shape[0]               # wide figures get a landscape page
        fig = self._new(landscape=wide)
        fig.text(0.07, 0.93, heading, fontsize=12, weight="bold", color=INK)
        ax = fig.add_axes([0.04, 0.12, 0.92, 0.78])
        ax.imshow(img)
        ax.axis("off")
        if caption:
            fig.text(0.07, 0.08, "\n".join(textwrap.wrap(caption, 100)), fontsize=8.5, color=INK2)
        self.pdf.savefig(fig)
        plt.close(fig)

    def figure(self, fig):
        self.page += 1
        self.pdf.savefig(fig)
        plt.close(fig)

    def close(self):
        d = self.pdf.infodict()
        d["Title"] = self.title
        d["CreationDate"] = dt.datetime.now()
        self.pdf.close()
        return self.path


def _confusion_fig(cm: dict, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    M = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(M, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=14, color=INK)
    ax.set_xticks([0, 1], ["flag: normal", "flag: anomaly"])
    ax.set_yticks([0, 1], ["label: healthy", "label: arthritis"])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def test_report(run_dir: Path) -> dict:
    """PDF + Excel + CSV for one independent-test run folder."""
    run_dir = Path(run_dir)
    res = json.load(open(run_dir / "results.json"))
    P = pd.read_csv(run_dir / "predictions.csv")
    lock = res["lock"]
    pdf = PDF(run_dir / "independent_test_report.pdf", f"Independent test — {lock['dataset_id']} — {lock['model_id']}")
    paras = [f"Run: {lock['run_id']}", f"Created: {lock['created']}", f"Dataset: {lock['dataset_id']} ({res['n_recordings']} recordings)",
             f"Model: {lock['model_id']} ({lock['rule']})",
             "Category counts: " + ", ".join(f"{k}: {v}" for k, v in res["category_counts"].items())]
    if res.get("warning"):
        paras.append("WARNING: " + res["warning"])
    m = res.get("metrics_flag_vs_label")
    if m:
        paras.append(f"Labelled samples used for metrics: {res['n_labelled_samples']} "
                     f"(excluded: {res['excluded_from_metrics']}).")
        keys = ["sensitivity", "specificity", "accuracy", "balanced_accuracy", "precision", "f1", "roc_auc", "average_precision"]
        paras.append("Metrics (flag vs label): " + ", ".join(f"{k} {m[k]:.3f}" for k in keys
                                                              if isinstance(m.get(k), float) and np.isfinite(m[k])))
        if m.get("small_sample_warning"):
            paras.append("WARNING: " + m["small_sample_warning"])
        if m.get("note"):
            paras.append("Note: " + m["note"])
        paras.append(m.get("interpretation", ""))
    else:
        paras.append("No labels supplied: predictions only, no accuracy metrics.")
    paras.append("Input fingerprints (SHA-256): features " + lock["features_sha256"][:16] + "…, model " + lock["model_sha256"][:16] + "…")
    pdf.text("Independent test summary", paras)
    if m and "confusion_matrix" in m:
        pdf.figure(_confusion_fig(m["confusion_matrix"], f"Flag vs label (n = {m['n']})"))
    if m and "roc_curve" in m:
        fig, ax = plt.subplots(1, 2, figsize=(8.27, 3.8))
        ax[0].plot(m["roc_curve"]["fpr"], m["roc_curve"]["tpr"], color="#2a78d6", lw=2)
        ax[0].plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
        ax[0].set(xlabel="false positive rate", ylabel="sensitivity", title=f"ROC (AUC {m['roc_auc']:.2f})")
        ax[1].plot(m["pr_curve"]["recall"], m["pr_curve"]["precision"], color="#eb6834", lw=2)
        ax[1].set(xlabel="recall", ylabel="precision", title=f"Precision-recall (AP {m['average_precision']:.2f})")
        fig.tight_layout()
        pdf.figure(fig)
    cols = [c for c in ["rec_id", "task", "category", "score", "threshold", "label", "top_features"] if c in P]
    pdf.table("Per-recording results", P[cols])
    pdf.text("Limitations", LIMITATIONS)
    pdf.close()
    tables = {"predictions": P, "summary": pd.json_normalize({k: v for k, v in res.items() if k != "lock"}, sep="."),
              "lock": pd.DataFrame([lock]).astype(str)}
    paths = export_tables(tables, run_dir, "independent_test")
    paths["pdf"] = run_dir / "independent_test_report.pdf"
    return paths


def dataset_report(dataset_id: str, max_views: int = 20) -> dict:
    from src.dataset_manager import DATASETS, load_manifest
    m = load_manifest(dataset_id)
    d = DATASETS / dataset_id
    out = d / "reports"
    R = pd.DataFrame([dict(rec_id=r["rec_id"], subject=r["subject"], hand=r["hand"], task=r["task"],
                           match=r["match_status"], missing=",".join(r["missing"])) for r in m["recordings"]])
    tables = {"recordings": R}
    if (d / "validation.json").is_file():
        V = json.load(open(d / "validation.json"))["results"]
        R = R.merge(pd.DataFrame([dict(rec_id=v["rec_id"], status=v["status"]) for v in V]), on="rec_id", how="left")
        tables["recordings"] = R
        tables["validation_checks"] = pd.DataFrame([dict(rec_id=v["rec_id"], **c) for v in V for c in v["checks"]])
    F = None
    if (d / "processed" / "features.csv").is_file():
        F = pd.read_csv(d / "processed" / "features.csv")
        tables["features"] = F
    pdf = PDF(out / f"{dataset_id}_dataset_report.pdf", f"Dataset report — {m['name']}")
    paras = [f"Dataset ID: {dataset_id}", f"Imported: {m['created']} from {m['source']}", f"Root: {m['root']}",
             f"Recordings: {len(m['recordings'])}; unmatched files: {len(m['unmatched_files'])}"]
    if "status" in R:
        paras.append("Validation: " + ", ".join(f"{k}: {v}" for k, v in R.status.value_counts().items()))
    pdf.text("Dataset summary", paras)
    pdf.table("Recordings", R)
    if F is not None:
        from src.landmarks import JOINT_NAMES
        cols = ["rec_id"] + [f"{j}_D2JTD" for j in JOINT_NAMES[:6]] + ["q_roi_usable"]
        pdf.table("Joint D2JTD (deg C) — first 6 joints, see Excel for all", F[[c for c in cols if c in F]])
        # joint temperatures: rows = joints, columns = recordings (median over usable frames)
        recs = F.rec_id.tolist()[:7]
        JT = pd.DataFrame({"joint": JOINT_NAMES})
        for r in recs:
            row = F[F.rec_id == r].iloc[0]
            JT[r] = [row.get(f"{j}_Tmed", np.nan) for j in JOINT_NAMES]
        pdf.table("Joint temperatures (°C, median of usable frames)" + (" — first 7 recordings" if len(F) > 7 else ""), JT,
                  fmt="{:.1f}")
        # per recording: the final value of each joint with its spread over frames and the number of frames used
        for r in recs:
            row = F[F.rec_id == r].iloc[0]
            K = pd.DataFrame({
                "joint": [j.replace("_", " ") for j in JOINT_NAMES],
                "final °C (median)": [row.get(f"{j}_Tmed", np.nan) for j in JOINT_NAMES],
                "Q1 °C (frames)": [row.get(f"{j}_Tq1", np.nan) for j in JOINT_NAMES],
                "Q3 °C (frames)": [row.get(f"{j}_Tq3", np.nan) for j in JOINT_NAMES],
                "frames": [str(int(v)) if np.isfinite(v := float(row.get(f"{j}_nframes", np.nan))) else "" for j in JOINT_NAMES],
                "D2JTD °C": [row.get(f"{j}_D2JTD", np.nan) for j in JOINT_NAMES]})
            pdf.table(f"{r}: final joint temperatures (median of frames)", K, fmt="{:.2f}")
        # one joint view per recording: RGB photo with joint positions and temperatures + thermal ROIs
        from src.io_utils import load_config
        from src.joint_view import render
        cfg = load_config(Path("config.yaml"))
        for r in F.rec_id.tolist()[:max_views]:
            try:
                png, _ = render(r, d / "processed", cfg, None, dataset_id, out / "joint_views" / f"{r}.png")
                pdf.image(f"Joint view — {r}", png, "Left: RGB photo, dots = detected joints, labels = ROI temperature (°C); "
                          "red = joint warmer than the back of the hand, blue = cooler. Right: thermal image with the "
                          "regions the temperatures were read from. Middle usable frame of the recording.")
            except Exception as e:                                   # a failed view must not stop the report
                pdf.text(f"Joint view — {r}", [f"Could not render: {e}"])
    pdf.text("Limitations", LIMITATIONS)
    pdf.close()
    paths = export_tables(tables, out, dataset_id)
    paths["pdf"] = out / f"{dataset_id}_dataset_report.pdf"
    return paths
