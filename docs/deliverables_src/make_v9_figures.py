"""Figures for v9 (calibration), frame aggregation and deployment. Run from the project root:
  .venv312\\Scripts\\python.exe docs/deliverables_src/make_v9_figures.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from make_figures import COL, INK, INK2, save  # noqa: E402
from make_arch_diagrams import A, B, setup  # noqa: E402

FIG = Path("deliverables/figures")
V9 = Path("reports/v9_calibration")


def aggregation():
    from src.landmarks import JOINT_NAMES
    d = pd.read_csv("data/processed_v8/frames/S2_right_static.csv")
    v = d[d.frame_valid.fillna(False).astype(bool)]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[2.2, 1, 1.5]))
    for j, c in (("index_MCP", COL["thermal"]), ("little_MCP", COL["rgb"])):
        x = v[f"{j}_T"]
        ax[0].plot(v.t_s, x, ".", ms=3, color=c, alpha=0.5)
        ax[0].axhline(x.median(), color=c, lw=2)
        ax[0].axhspan(x.quantile(.25), x.quantile(.75), color=c, alpha=0.12)
        ax[0].text(v.t_s.max() * 1.01, x.median(), f"{j.replace('_', ' ')}\nfinal {x.median():.2f} °C", color=c, va="center", fontsize=9)
    ax[0].plot(v.t_s, v.dorsum_T, "-", color=INK2, lw=1, alpha=0.6, label="back of hand")
    ax[0].set_xlim(0, v.t_s.max() * 1.22)
    ax[0].set_xlabel("time in recording (s) — 267 analysed frames")
    ax[0].set_ylabel("ROI temperature (°C)")
    ax[0].set_title("Each frame gives a slightly different value (slow drift of the whole hand)")
    ax[0].legend(loc="upper left", frameon=False)
    x = v["index_MCP_T"]
    ax[1].hist(x, bins=25, color=COL["thermal"], alpha=0.7, orientation="horizontal")
    for q, ls in ((x.quantile(.25), ":"), (x.median(), "-"), (x.quantile(.75), ":")):
        ax[1].axhline(q, color=INK, ls=ls, lw=1.6)
    ax[1].set_title("index MCP over frames\nsolid = median (final), dotted = Q1 / Q3")
    ax[1].set_xlabel("frames")
    meds = [v[f"{j}_T"].median() for j in JOINT_NAMES]
    q1 = [v[f"{j}_T"].quantile(.25) for j in JOINT_NAMES]
    q3 = [v[f"{j}_T"].quantile(.75) for j in JOINT_NAMES]
    y = np.arange(len(JOINT_NAMES))
    ax[2].errorbar(meds, y, xerr=[np.subtract(meds, q1), np.subtract(q3, meds)], fmt="o", color=COL["feat"], capsize=3)
    ax[2].set_yticks(y)
    ax[2].set_yticklabels([j.replace("_", " ") for j in JOINT_NAMES], fontsize=8)
    ax[2].invert_yaxis()
    ax[2].set_xlabel("final temperature (°C) with Q1–Q3")
    ax[2].set_title("One value per joint for the report")
    fig.suptitle("S2_right_static: from 267 frames to one temperature per joint (median of the frames)", color=INK, fontsize=13)
    fig.tight_layout()
    save(fig, "N01_frame_aggregation.png")


def selftest():
    S = pd.read_csv(V9 / "selftest_known_distortions.csv")
    short = {"scale +8 %, rotate 4°, shift (+18, -12) px": "+8 %, 4°, shift",
             "scale -10 %, rotate -6°, shift (-25, +20) px": "−10 %, −6°, shift",
             "different letterbox: picture moved 40 px right": "picture moved 40 px",
             "all together: scale +5 %, rotate 3°, shift (10, 15), letterbox +40 px": "all combined"}
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.8, 1]))
    recs = S.recording.unique()
    cols = [COL["rgb"], COL["thermal"], COL["feat"]]
    w = 0.26
    cases = list(short)
    for i, (r, c) in enumerate(zip(recs, cols)):
        g = S[S.recording == r].set_index("case").loc[cases]
        ax[0].bar(np.arange(4) + (i - 1) * w, g.mapping_error_px_median, w, color=c, label=r)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels([short[c] for c in cases])
    ax[0].set_ylabel("mapping error vs known truth (px, median)")
    ax[0].set_title("Known distortions applied to real recordings: all 12 recovered")
    ax[0].legend(frameon=False)
    ax[1].plot([1.1, 1.42], [1.1, 1.42], color=INK2, lw=1)
    ax[1].scatter(S.scale_true, S.scale_found, color=COL["calib"], s=40, label="scale")
    ax[1].set_xlabel("true scale")
    ax[1].set_ylabel("found scale")
    ax2 = ax[1].twinx()
    ax2.scatter(S.scale_true, S.angle_found - S.angle_true, color=COL["ml"], marker="x", s=40, label="angle error")
    ax2.set_ylabel("angle error (°)")
    ax2.set_ylim(-3, 3)
    ax[1].set_title("Scale found = true scale (±0.01); angle within 0.75°")
    fig.tight_layout()
    save(fig, "N02_calibration_selftest.png")


def benchmark():
    b = json.load(open(V9 / "benchmark_find_calibration.json"))["results"]
    names = list(b)
    lab = ["fitted per recording (v3)", "found calibration (v9)"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(3)
    for i, (n, c) in enumerate(zip(names, ["#9aa7b8", COL["feat"]])):
        vals = [b[n]["joint_err_median"], b[n]["static"], b[n]["open_close"]]
        bars = ax[0].bar(x + (i - 0.5) * 0.38, vals, 0.38, color=c, label=lab[i])
        for bb, v in zip(bars, vals):
            ax[0].text(bb.get_x() + bb.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["all joints", "static", "open–close"])
    ax[0].set_ylabel("median joint error vs clicked joints (px)")
    ax[0].set_title("12 unseen subjects (S6–S17), 551 clicked joints")
    ax[0].legend(frameon=False)
    vals = [b[n]["dice_median"] for n in names]
    bars = ax[1].bar(lab, vals, color=["#9aa7b8", COL["feat"]])
    for bb, v in zip(bars, vals):
        ax[1].text(bb.get_x() + bb.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    ax[1].set_ylim(0.85, 0.93)
    ax[1].set_ylabel("median Dice, held-out frames")
    ax[1].set_title("Hand-outline overlap")
    fig.suptitle("Calibration found from S1–S5 only, applied unchanged to S6–S17", color=INK, fontsize=13)
    fig.tight_layout()
    save(fig, "N03_calibration_benchmark.png")


def deployment():
    fig, ax = setup(17, 5.6)
    ax.text(0.3, 5.15, "Deployment: from the release zip to results", fontsize=14, color=INK, weight="bold")
    s = [("Release zip", "rgb_thermal_hand-1.0.0\n27 MB", COL["data"]), ("install.ps1", "Python 3.12 + exact\npackages (lock file)", COL["out"]),
         ("Health check", "rgbthermal doctor\n→ READY", COL["feat"]), ("Desktop app", "RGB-Thermal App.bat\n10 pages", COL["rgb"]),
         ("Results", "PDF · Excel · CSV\njoint views", COL["ml"])]
    bx = [B(ax, 0.3 + i * 3.35, 2.9, 2.9, 1.5, t, sub, c) for i, (t, sub, c) in enumerate(s)]
    for p, q in zip(bx[:-1], bx[1:]):
        A(ax, p, q)
    cli = B(ax, 6.95, 0.6, 2.9, 1.4, "Command line", "rgbthermal calibrate /\nprocess / test", COL["calib"])
    A(ax, bx[2], cli, "d")
    ax.text(10.2, 1.3, "models/, calibration profiles and the\nscreening model are in the release,\nchecked by SHA-256 at every health check",
            fontsize=9.5, color=INK2, va="center")
    save(fig, "N05_deployment_workflow.png")


def doctor_image():
    py = sys.executable
    out = subprocess.run([py, "-m", "src.cli", "doctor"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                         env=dict(os.environ, RGBT_DATASET_ROOT="")).stdout
    import re
    lines = [re.sub(r"[A-Za-z]:[\\/]Users[\\/]\S*", "<dataset folder>", ln.replace(str(Path.cwd()), "<install folder>"))
             for ln in out.splitlines() if ln.strip()]
    fig = plt.figure(figsize=(12.5, 0.32 * len(lines) + 0.6))
    fig.patch.set_facecolor("#1e1e1e")
    for i, ln in enumerate(lines):
        col = "#6fd36f" if "[OK" in ln or "READY" in ln else "#e6c84b" if "WARN" in ln else "#ff6b6b" if "FAIL" in ln else "#dddddd"
        fig.text(0.02, 1 - (i + 1) / (len(lines) + 1), ln, family="monospace", fontsize=10, color=col, va="center")
    fig.savefig(FIG / "N06_doctor_output.png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote N06_doctor_output.png")


def app_shots():
    os.environ["QT_QPA_PLATFORM"] = "windows"
    from PySide6.QtWidgets import QApplication

    from ui.app import MainWindow, STYLE
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.resize(1400, 860)
    names = [w.nav.item(i).text() if hasattr(w, "nav") else str(i) for i in range(w.stack.count())]
    want = {"Calibration": "N07_app_calibration.png", "Dataset upload": "N08_app_dataset_upload.png", "Models": "N09_app_models.png",
            "Reports": "N10_app_reports.png"}
    for i, n in enumerate(names):
        for key, fn in want.items():
            if key.lower() in n.lower():
                w.stack.setCurrentIndex(i)
                if hasattr(w, "nav"):
                    w.nav.setCurrentRow(i)
                page = w.stack.widget(i)
                if key == "Calibration":
                    for j in range(page.list.count()):
                        if "v9_found" in page.list.item(j).text():
                            page.list.setCurrentRow(j)
                if key == "Models":
                    for j in range(page.list.count()):
                        if "172656" in page.list.item(j).text():        # the application's default model
                            page.list.setCurrentRow(j)
                if key == "Reports" and page.list.count():
                    page.list.setCurrentRow(0)
                import time
                t_end = time.time() + 2.5                                   # let PDF pages and tables draw
                while time.time() < t_end:
                    app.processEvents()
                    time.sleep(0.05)
                w.grab().save(str(FIG / fn))
                print("wrote", fn)
    w.close()


def calibration_examples():
    from src.calibration import Calibration, apply_to_frame, verify
    from src.io_utils import load_config, load_recordings
    cfg = load_config(Path("config.yaml"))
    info = load_recordings(cfg, only=["S9_left_static"])[0]
    cal = Calibration.load("models/calibration/flirone_one_pro_v9_found.json")
    r = verify(cal, info.rgb_path, info.npy_path, n=8, out_png=FIG / "N11_calibration_verify.png")
    res = apply_to_frame(cal, info.rgb_path, info.npy_path, 300, FIG / "N12_calibration_apply.png")
    json.dump(dict(verify=r, apply=res), open(V9 / "example_S9_left_static.json", "w"), indent=1)
    print("verify", r)


if __name__ == "__main__":
    aggregation()
    selftest()
    benchmark()
    deployment()
    doctor_image()
    calibration_examples()
    app_shots()
