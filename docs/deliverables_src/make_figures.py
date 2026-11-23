"""Figures for the technical report and presentation (deliverables/figures/).

Diagrams are drawn with matplotlib. Every illustration of data uses a REAL frame or REAL saved results;
nothing is synthetic except the schematic diagrams, which are labelled as such in the captions.
Run from the project root:  .venv312\\Scripts\\python.exe docs\\deliverables_src\\make_figures.py
"""
import json
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

from src import landmark_correction as LC  # noqa: E402
from src import registration as R  # noqa: E402
from src import rois  # noqa: E402
from src.io_utils import crop_to_npy_scale, iter_frames, load_config, load_recordings  # noqa: E402
from src.landmarks import DORSUM_LM, FINGER_LM, JOINTS  # noqa: E402
from src.segmentation import _fill_holes, _largest_component, rgb_mask, thermal_mask  # noqa: E402

OUT = Path("deliverables/figures")
OUT.mkdir(parents=True, exist_ok=True)
M = json.load(open("deliverables/supporting/metrics_verified.json"))
CFG = load_config(Path("config.yaml"))
BOX = tuple(CFG["pipeline"]["video_box"])
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
COL = dict(rgb="#2a78d6", thermal="#eb6834", calib="#6250d6", feat="#1baf7a", ml="#d55181", out="#c98500", data="#52514e")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlecolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False})


def save(fig, name, dpi=160):
    fig.savefig(OUT / name, dpi=dpi, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, color, status="impl", fs=8.6, tc=None):
    ls = {"impl": "-", "partial": (0, (5, 2)), "proposed": (0, (1, 2))}[status]
    fc = color if status == "impl" else "white"
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=color, lw=1.6,
                       linestyle=ls, alpha=0.95 if status == "impl" else 1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or ("white" if status == "impl" else color), wrap=True)


def arrow(ax, x1, y1, x2, y2, color=INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=color, lw=1.2))


# ---------------------------------------------------------------- 1 architecture
def architecture():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.axis("off")
    W, H = 0.165, 0.075
    rows = [  # (y, [(x, text, color, status)])
        (0.89, [(0.02, "RGB video\n(MP4, 480×640)", COL["data"], "impl"), (0.215, "Thermal video\n(display only)", COL["data"], "impl"),
                (0.41, "Radiometric .npy\n(°C, 640×480)", COL["data"], "impl"), (0.605, "Ground-truth labels\n(CSV / Excel)", COL["data"], "impl"),
                (0.80, "Uploaded datasets\n(any folder layout)", COL["data"], "impl")]),
        (0.76, [(0.12, "Dataset import & audit\nZIP / folder / single", COL["out"], "impl"), (0.41, "Validation\n5 status categories", COL["out"], "impl"),
                (0.70, "Calibration profiles\n(versioned)", COL["calib"], "impl")]),
        (0.63, [(0.02, "In-order decoding\nframe i ↔ frame i", COL["calib"], "impl"), (0.215, "MediaPipe Hand\nLandmarker (21 pts)", COL["rgb"], "impl"),
                (0.41, "Thermal hand mask\nOtsu + hysteresis", COL["thermal"], "impl"), (0.605, "RGB→thermal similarity\ns = 1.25, shift (tx, ty)", COL["calib"], "impl"),
                (0.80, "Rig fallback\n(held-out Dice check)", COL["calib"], "impl")]),
        (0.50, [(0.02, "Anatomical checks\n+ joint visibility", COL["rgb"], "impl"), (0.215, "Along-finger joint\ncorrection (learned)", COL["rgb"], "impl"),
                (0.41, "14 joint ROIs + dorsum\n(finger-width radius)", COL["thermal"], "impl"), (0.605, "ROI statistics\nmedian, mean, P90, SD", COL["thermal"], "impl"),
                (0.80, "Accessory check\n(cold-band flag)", COL["thermal"], "impl")]),
        (0.37, [(0.12, "Features: D2JTD, spatial,\nabsolute, temporal", COL["feat"], "impl"), (0.41, "Quality control\nflags & categories", COL["feat"], "impl"),
                (0.70, "Feature ablation\n(nested LOSO)", COL["feat"], "impl")]),
        (0.24, [(0.02, "Healthy reference\nmodel (17 subjects)", COL["ml"], "impl"), (0.215, "Anomaly models\n(z/MAD, Mahalanobis,\nMCD, IF, OCSVM)", COL["ml"], "impl"),
                (0.41, "Nested-LOSO\nthresholds", COL["ml"], "impl"), (0.605, "Per-joint explanation\n(deviation in SD)", COL["ml"], "impl"),
                (0.80, "Saved model bundles\n(versioned registry)", COL["ml"], "impl")]),
        (0.11, [(0.02, "Joint view: RGB + temps\n+ thermal ROIs", COL["out"], "impl"), (0.215, "Locked independent\ntest + metrics", COL["out"], "impl"),
                (0.41, "PDF / Excel / CSV\nreports", COL["out"], "impl"), (0.605, "Desktop app\n(PySide6, 10 pages)", COL["out"], "impl"),
                (0.80, "Live mode\n(recorded replay)", COL["out"], "impl")]),
    ]
    for y, items in rows:
        for x, t, c, s in items:
            box(ax, x, y, W, H, t, c, s)
    for y0, y1 in ((0.89, 0.835), (0.76, 0.705), (0.63, 0.575), (0.50, 0.445), (0.37, 0.315), (0.24, 0.185)):
        arrow(ax, 0.5, y0, 0.5, y1)
    legend = [("RGB processing", COL["rgb"]), ("Thermal processing", COL["thermal"]), ("Calibration & alignment", COL["calib"]),
              ("Features & QC", COL["feat"]), ("Machine learning", COL["ml"]), ("Data, output & evaluation", COL["out"])]
    for i, (t, c) in enumerate(legend):
        ax.add_patch(FancyBboxPatch((0.02 + i * 0.165, 0.015), 0.018, 0.026, boxstyle="round,pad=0.002", fc=c, ec=c))
        ax.text(0.043 + i * 0.165, 0.028, t, va="center", fontsize=8, color=INK)
    ax.text(0.02, 1.005, "All blocks shown are implemented in the project code",
            fontsize=9, color=INK2)
    save(fig, "F01_architecture.png")


# ---------------------------------------------------------------- 2 pipeline flow
def pipeline():
    stages = [("1 Load & audit", "RGB MP4 + .npy\n(memory-mapped)", COL["out"]), ("2 Synchronise", "decode in order\nframe i ↔ i", COL["calib"]),
              ("3 Register", "s = 1.25,\nshift per recording", COL["calib"]), ("4 Landmarks", "MediaPipe\n21 points", COL["rgb"]),
              ("5 Joints", "14 joints, checks,\nvisibility, correction", COL["rgb"]), ("6 Segment", "thermal mask\nOtsu + hysteresis", COL["thermal"]),
              ("7 ROIs", "finger-width\ncircles + dorsum", COL["thermal"]), ("8 Temperatures", "median / mean /\nP90 / SD (°C)", COL["thermal"]),
              ("9 Features", "D2JTD, spatial,\nabsolute, temporal", COL["feat"]), ("10 Quality", "validity flags,\ncategories", COL["feat"]),
              ("11 Model", "healthy reference\n+ anomaly score", COL["ml"]), ("12 Outputs", "joint view, tables,\nreports, metrics", COL["out"])]
    fig, ax = plt.subplots(figsize=(13, 4.3))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, (t, sub, c) in enumerate(stages):
        r, k = divmod(i, 6)
        x = k if r == 0 else 5 - k
        y = 1.15 if r == 0 else 0.1
        box(ax, x + 0.05, y, 0.85, 0.7, f"{t}\n\n{sub}", c, "impl", fs=8.4)
        if i < len(stages) - 1:
            if r == 0 and k < 5:
                arrow(ax, x + 0.9, y + 0.35, x + 1.05, y + 0.35)
            elif r == 0 and k == 5:
                arrow(ax, x + 0.475, y, x + 0.475, 0.8)
            else:
                arrow(ax, x + 0.05, y + 0.35, x - 0.1, y + 0.35)
    save(fig, "F02_pipeline.png")


# ---------------------------------------------------------------- 3 dataset tree
def dataset_tree():
    d = M["dataset"]
    txt = ("FLIROne/   (17 subjects, 68 recordings, %.1f GB)\n"
           "├── S1/\n│   ├── left hand/\n│   │   ├── static_RGB.mp4                       RGB, 480×640, ~8.3 fps\n"
           "│   │   ├── static_Thermal.mp4                   colour-mapped thermal (display only)\n"
           "│   │   ├── static_Thermal_Radiometric_Sequence.npy   float32 °C, (N, 640, 480)\n"
           "│   │   ├── open_close_RGB.mp4\n│   │   ├── open_close_Thermal.mp4\n"
           "│   │   └── open_close_Thermal_Radiometric_Sequence.npy\n│   └── right hand/   (same 6 files)\n"
           "├── S2/ … S17/\n") % (d["npy_gb"] + d["video_gb"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis("off")
    ax.text(0.01, 0.98, txt, family="DejaVu Sans Mono", fontsize=10, va="top", color=INK)
    save(fig, "F03_dataset_tree.png")


# ---------------------------------------------------------------- real frame helpers
def frame(rec, f, proc="data/processed_v8"):
    info = load_recordings(CFG, only=[rec])[0]
    rgb, T = next((r, t) for i, r, t in iter_frames(info, [f], BOX))
    row = pd.read_csv(f"{proc}/frames/{rec}.csv").set_index("frame").loc[f]
    reg = pd.read_csv(f"{proc}/registration.csv").set_index("rec_id").loc[rec]
    M_ = R.matrix(crop_to_npy_scale(BOX), reg.scale, reg.tx, reg.ty)
    raw = np.array([[row[f"lm{j}_x"], row[f"lm{j}_y"]] for j in range(21)])
    Minv = cv2.invertAffineTransform(M_)
    return rgb, T, row, M_, raw, raw @ Minv[:, :2].T + Minv[:, 2]


def landmarks_fig():
    rgb, T, row, M_, raw, crop = frame("S2_right_static", 399)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 7.4))
    ax[0].imshow(rgb)
    for ch in FINGER_LM.values():
        c = [0] + ch
        ax[0].plot(crop[c, 0], crop[c, 1], "-", color="white", lw=1.4)
    for j in range(21):
        ax[0].plot(*crop[j], "o", ms=7, mfc=COL["rgb"], mec="white", mew=1.2)
        ax[0].text(crop[j, 0] + 5, crop[j, 1] - 5, str(j), color="white", fontsize=8, weight="bold")
    ax[0].set_title("21 MediaPipe landmarks on the RGB frame (S2_right_static, frame 399)", fontsize=10)
    ax[0].axis("off")
    ax[1].axis("off")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    lines = [("0", "wrist"), ("1–4", "thumb: CMC, MCP (2), IP (3), tip"), ("5–8", "index: MCP (5), PIP (6), DIP (7), tip"),
             ("9–12", "middle: MCP (9), PIP (10), DIP (11), tip"), ("13–16", "ring: MCP (13), PIP (14), DIP (15), tip"),
             ("17–20", "little: MCP (17), PIP (18), DIP (19), tip")]
    ax[1].text(0, 0.95, "Landmark index → anatomy", fontsize=12, weight="bold", color=INK)
    for i, (a, b) in enumerate(lines):
        ax[1].text(0.0, 0.86 - i * 0.075, a, fontsize=11, color=COL["rgb"], weight="bold")
        ax[1].text(0.16, 0.86 - i * 0.075, b, fontsize=11, color=INK)
    ax[1].text(0, 0.36, "14 joint regions measured:", fontsize=12, weight="bold", color=INK)
    ax[1].text(0, 0.29, "thumb MCP, thumb IP\nMCP, PIP and DIP of index, middle, ring, little\n\n"
               "MCP = metacarpophalangeal (knuckle)\nPIP = proximal interphalangeal (middle joint)\n"
               "DIP = distal interphalangeal (end joint)\nIP = interphalangeal (thumb)", fontsize=10.5, color=INK, va="top")
    save(fig, "F05_landmarks_real.png")


def roi_fig():
    rgb, T, row, M_, raw, crop = frame("S2_right_static", 399)
    pts = LC.apply(raw, LC.load())
    hand, _ = thermal_mask(T)
    rr = rois.joint_radii(pts, hand, CFG["pipeline"]["roi"]["beta"], "finger_width")
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 7.6))
    im = ax[0].imshow(T, cmap="inferno")
    ax[0].add_patch(Polygon(pts[DORSUM_LM], closed=True, fill=False, ec="#9ec5f4", lw=2, ls="--"))
    for name, lm, finger, kind in JOINTS:
        ax[0].add_patch(plt.Circle(pts[lm], rr[name], fill=False, color="white", lw=1.5))
        ax[0].text(*pts[lm], f"{kind}", color="white", fontsize=6.5, ha="center", va="center")
    ax[0].text(*pts[DORSUM_LM].mean(0), "dorsum\n(back of hand)", color="#cde2fb", ha="center", fontsize=9)
    fig.colorbar(im, ax=ax[0], fraction=0.04, label="temperature (°C)")
    ax[0].set_title("14 joint ROIs (white) and dorsum region (dashed) on the radiometric frame", fontsize=10)
    ax[0].axis("off")
    # D2JTD numbers for this frame
    tab = [(n, row[f"{n}_T"], row["dorsum_T"] - row[f"{n}_T"]) for n, *_ in JOINTS]
    ax[1].axis("off")
    ax[1].set_title(f"Measured values in this frame (dorsum = {row['dorsum_T']:.2f} °C)", fontsize=10)
    tb = ax[1].table(cellText=[[n.replace("_", " "), f"{t:.2f}", f"{d:+.2f}"] for n, t, d in tab],
                     colLabels=["joint", "joint T (°C)", "D2JTD (°C)"], loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.5)
    tb.scale(1, 1.45)
    save(fig, "F06_rois_d2jtd_real.png")


def segmentation_fig():
    rgb, T, row, M_, raw, crop = frame("S17_right_static", 450)
    t = cv2.GaussianBlur(T, (5, 5), 0)
    lo, hi = np.percentile(t, 1), np.percentile(t, 99.5)
    t8 = np.clip((t - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    thr8, otsu = cv2.threshold(t8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seed = _largest_component(otsu > 0)
    final, low = thermal_mask(T)
    core = cv2.erode(final.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    fig, ax = plt.subplots(1, 4, figsize=(15, 5.4))
    for a, im, t_ in zip(ax, [T, seed, final, core],
                         ["Radiometric frame (°C) — cold fingers", f"Otsu seed (> {lo + thr8 / 255 * (hi - lo):.1f} °C)",
                          f"Hysteresis: grown into connected\npixels > {low:.1f} °C (fingers kept)", "Eroded core (~1 sensor px):\ntemperatures sampled here"]):
        a.imshow(im, cmap="inferno" if im is T else "gray")
        a.set_title(t_, fontsize=9.5)
        a.axis("off")
    save(fig, "F08_thermal_segmentation_real.png")


def registration_fig():
    rgb, T, row, M_, raw, crop = frame("S2_right_static", 399)
    rm = rgb_mask(rgb, crop.astype(np.float32))
    tm, _ = thermal_mask(T)
    wm = R.warp_mask(rm, M_)
    ov = np.zeros((*T.shape, 3))
    ov[..., 0] = tm
    ov[..., 2] = wm
    ov[..., 1] = tm & wm
    fig, ax = plt.subplots(1, 4, figsize=(15, 5.6))
    ax[0].imshow(rgb)
    ax[0].set_title("RGB crop (360×480)", fontsize=9.5)
    ax[1].imshow(rm, cmap="gray")
    ax[1].set_title("RGB hand mask (landmark-seeded GrabCut)", fontsize=9.5)
    ax[2].imshow(tm, cmap="gray")
    ax[2].set_title("Thermal hand mask (480×640)", fontsize=9.5)
    ax[3].imshow(ov)
    d = 2 * (tm & wm).sum() / (tm.sum() + wm.sum())
    ax[3].set_title(f"Overlay after mapping (white = overlap)\nDice = {d:.3f}", fontsize=9.5)
    for a in ax:
        a.axis("off")
    save(fig, "F09_registration_real.png")


def transform_fig():
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    v = M["v3"]
    box(ax, 0.2, 1.4, 2.4, 2.4, "RGB video frame\n640 × 480\n(letterboxed)", COL["rgb"], fs=10)
    box(ax, 3.4, 1.4, 1.9, 2.4, "crop box\nx 140–500\n360 × 480", COL["rgb"], fs=10)
    box(ax, 6.1, 1.4, 2.2, 2.4, "× k = 4/3\n→ 480 × 640\n(.npy grid)", COL["calib"], fs=10)
    box(ax, 9.1, 1.4, 3.6, 2.4, f"similarity about centre c\nscale s = {v['scale']:.2f}\nshift t = ({v['tx']:.1f}, {v['ty']:.1f}) px\n(per recording; rig median shown)",
        COL["calib"], fs=10)
    for x1, x2 in ((2.6, 3.4), (5.3, 6.1), (8.3, 9.1)):
        arrow(ax, x1, 2.6, x2, 2.6)
    ax.text(6.5, 0.55, r"$p_{thermal} = s\,(k\,p_{crop} - c) + c + t$", fontsize=16, ha="center", color=INK)
    ax.text(6.5, 0.1, "c = (240, 320) = centre of the 480×640 radiometric frame · inverse transform maps thermal points back to the photo",
            fontsize=9, ha="center", color=INK2)
    save(fig, "F04_transform.png")


def workflow(name, steps, colors, title_note=""):
    fig, ax = plt.subplots(figsize=(13, 2.6))
    n = len(steps)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (s, c) in enumerate(zip(steps, colors)):
        box(ax, i + 0.04, 0.15, 0.86, 0.7, s, c, fs=8.6)
        if i < n - 1:
            arrow(ax, i + 0.9, 0.5, i + 1.04, 0.5)
    if title_note:
        ax.text(0, 0.95, title_note, fontsize=9, color=INK2)
    save(fig, name)


def loso_fig():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    n = 17
    ax.set_xlim(-2.3, n)
    ax.set_ylim(-0.6, 5.2)
    ax.axis("off")
    for r, (test, label) in enumerate([(0, "outer fold 1"), (1, "outer fold 2"), (16, "outer fold 17")]):
        y = 4 - r * 1.5
        for s in range(n):
            col = COL["thermal"] if s == test else COL["rgb"]
            ax.add_patch(FancyBboxPatch((s + 0.08, y), 0.84, 0.55, boxstyle="round,pad=0.01", fc=col, ec="white"))
            ax.text(s + 0.5, y + 0.27, f"S{s + 1}", ha="center", va="center", color="white", fontsize=7.5)
        ax.text(-0.3, y + 0.27, label, ha="right", va="center", fontsize=9, color=INK)
    ax.text(8.5, 1.35, "⋮", fontsize=16, ha="center")
    ax.text(0, -0.45, "Blue = 16 training subjects: imputation, scaling, model AND threshold learned only here "
            "(threshold = 95th percentile of an inner leave-one-subject-out over these 16).\nOrange = held-out subject: "
            "scored once, never used for fitting. All hands and frames of a subject stay together.", fontsize=9, color=INK2)
    save(fig, "F11_loso_validation.png")


def accessory_fig():
    workflow("F10_outlier_handling.png",
             ["Joint ROI\npixels (°C)", "Invalid pixels\n(NaN/inf) removed", "Cold band check:\n>15 % pixels\n1.5 °C below median",
              "Flag 'possibly\ncontaminated'\n(e.g. metal ring)", "Hot pixels KEPT:\nmay be genuine\ninflammation"],
             [COL["thermal"], COL["feat"], COL["feat"], COL["out"], COL["ml"]],
             "Outlier policy: remove only invalid values, flag likely non-skin (cold, low-emissivity metal), never discard warm skin")


# ---------------------------------------------------------------- result graphs
def g_localisation():
    e = pd.DataFrame(M["v5"]["errors"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.arange(0, 61, 2)
    ax[0].hist(e.err_raw.clip(upper=60), bins, color=COL["data"], alpha=0.55, label=f"raw (median {e.err_raw.median():.1f} px)")
    ax[0].hist(e.err_loso.clip(upper=60), bins, color=COL["rgb"], alpha=0.75, label=f"corrected, LOSO (median {e.err_loso.median():.1f} px)")
    ax[0].axvline(8, color=COL["thermal"], ls="--", lw=1.4)
    ax[0].text(8.5, ax[0].get_ylim()[1] * 0.92, "8 px target", color=COL["thermal"], fontsize=9)
    ax[0].set_xlabel("joint position error (px, .npy grid; 4 px ≈ 1 sensor pixel)")
    ax[0].set_ylabel("clicked joints")
    ax[0].legend(fontsize=8.5)
    ax[0].set_title(f"Joint localisation error, n = {len(e)} clicked joints / 136 frames", fontsize=10)
    g = e.groupby("task").err_loso.median()
    ax[1].bar(["static", "open–close"], [g["static"], g["open_close"]], color=[COL["rgb"], COL["thermal"]], width=0.5)
    for i, v in enumerate([g["static"], g["open_close"]]):
        ax[1].text(i, v + 0.3, f"{v:.1f} px", ha="center", fontsize=10)
    ax[1].axhline(8, color=COL["thermal"], ls="--", lw=1.2)
    ax[1].set_ylabel("median corrected error (px)")
    ax[1].set_title("By task (leave-one-subject-out corrected)", fontsize=10)
    save(fig, "G01_localisation_error.png")


def g_registration():
    d3 = np.array(M["v3"]["reg_dice"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(d3, bins=np.arange(0.68, 0.97, 0.01), color=COL["calib"], alpha=0.85)
    ax.axvline(np.median(d3), color=INK, lw=1.2)
    ax.text(np.median(d3) + 0.003, ax.get_ylim()[1] * 0.9, f"median {np.median(d3):.3f}", fontsize=9)
    ax.axvline(0.9, color=COL["thermal"], ls="--")
    ax.set_xlabel("held-out hand-mask Dice per recording")
    ax.set_ylabel("recordings")
    ax.set_title(f"RGB→thermal registration, 68 recordings ({int((d3 >= 0.9).sum())} ≥ 0.90)", fontsize=10)
    save(fig, "G02_registration_dice.png")


def g_repeatability():
    m = M["v3"]["mdc_joints"]
    names = [k for k in m if k != "D2JTD_mean10"] + ["D2JTD_mean10"]
    vals = [m[k] for k in names]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    cols = [COL["feat"] if v < 0.4 else COL["data"] for v in vals]
    cols[-1] = COL["rgb"]
    ax.bar(range(len(names)), vals, color=cols)
    ax.axhline(0.4, color=COL["thermal"], ls="--", lw=1.2)
    ax.text(len(names) - 0.6, 0.41, "0.4 °C healthy–RA gap (Qiao et al.)", ha="right", fontsize=8.5, color=COL["thermal"])
    ax.set_xticks(range(len(names)), [n.replace("_", " ").replace("D2JTD mean10", "10-joint mean") for n in names], rotation=45, ha="right")
    ax.set_ylabel("MDC95 (°C)")
    ax.set_title("Within-session repeatability of D2JTD (34 static hands, split-half): smaller = more repeatable", fontsize=10)
    save(fig, "G03_repeatability_mdc.png")


def g_ablation():
    ab = pd.DataFrame(M["v6"]["ablation"])
    ab = ab[ab.model == "mahalanobis_ledoit_wolf"]
    lab = [s.split("_", 1)[1].replace("_", " ").replace("(v3 baseline)", "(v3)") for s in ab.feature_set]
    x = np.arange(len(ab))
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.bar(x - 0.27, ab.specificity, 0.27, color=COL["rgb"], label="specificity (real held-out hands)")
    ax.bar(x, ab.sens_single_joint_2C, 0.27, color=COL["thermal"], label="sensitivity, one joint +2 °C (simulated)")
    ax.bar(x + 0.27, ab.sens_all10_1C, 0.27, color=COL["feat"], label="sensitivity, all 10 MCP/PIP +1 °C (simulated)")
    for xi, n in zip(x, ab.n_features):
        ax.text(xi, 1.02, f"{n} feat.", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, lab, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.set_title("Feature-set ablation (Ledoit–Wolf model, nested leave-one-subject-out)", fontsize=10)
    save(fig, "G04_feature_ablation.png")


def g_synthetic():
    S = pd.DataFrame(M["v7"]["synthetic"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    deltas = [c for c in S.columns if c not in ("model", "pattern")]
    xs = [float(d) for d in deltas]
    for a, model in zip(ax, ("robust_covariance_mcd", "mahalanobis_ledoit_wolf")):
        g = S[S.model == model].set_index("pattern")
        for pat, col in (("single_joint", COL["thermal"]), ("all_10_MCP_PIP", COL["rgb"]), ("MCP_only", COL["feat"]),
                         ("DIP_only", COL["ml"]), ("diffuse_whole_hand", COL["data"])):
            a.plot(xs, g.loc[pat, deltas].astype(float), "-o", ms=6, lw=2, color=col, label=pat.replace("_", " "))
        a.set_title(model.replace("_", " "), fontsize=10)
        a.set_xlabel("simulated warming (°C)")
        a.set_ylim(0, 1.03)
    ax[0].set_ylabel("sensitivity (fraction flagged)")
    ax[1].legend(fontsize=8.5, loc="lower right")
    fig.suptitle("SIMULATED anomalies on healthy hands (D2JTD features, nested LOSO, specificity 0.94)", fontsize=10.5)
    save(fig, "G05_synthetic_sensitivity.png")


def g_visibility():
    v = M["v8"]["visibility"]
    kinds = ["MCP", "PIP", "IP", "DIP", "ANY"]
    labels = ["knuckle\n(MCP)", "middle\n(PIP)", "thumb\nIP", "end\n(DIP)", "frames with\nany hidden"]
    x = np.arange(len(kinds))
    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.bar(x - 0.2, [v["static"][k] * 100 for k in kinds], 0.4, color=COL["rgb"], label="static (open hand)")
    ax.bar(x + 0.2, [v["open_close"][k] * 100 for k in kinds], 0.4, color=COL["thermal"], label="open–close")
    for xi, k in zip(x, kinds):
        ax.text(xi + 0.2, v["open_close"][k] * 100 + 1, f"{v['open_close'][k] * 100:.0f} %", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("% of detected frames hidden")
    ax.legend(fontsize=9)
    ax.set_title(f"Joints judged not visible in the RGB image (v8, {M['v8']['frames']} frames)", fontsize=10)
    save(fig, "G06_visibility.png")


def g_compute():
    p = M["v3"]["ms_parts"]
    parts = [("landmarks (MediaPipe)", p["landmarks"], COL["rgb"]), ("thermal mask", p["thermal_mask"], COL["thermal"]),
             ("ROIs + features", p["rois"], COL["feat"])]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    left = 0
    for name, v, c in parts:
        ax.barh(0, v, left=left, color=c, height=0.5)
        ax.text(left + v / 2, 0, f"{name}\n{v:.1f} ms", ha="center", va="center", color="white", fontsize=9)
        left += v
    ax.set_xlim(0, 32)
    note = f"camera delivers a frame every {1000 / 8.7:.0f} ms (8.7 Hz)" + "\n" + f"→ {1000 / 8.7 / p['total']:.1f}× headroom"
    ax.text(31.5, 0.38, note, ha="right", va="top", fontsize=8.5, color=COL["thermal"])
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("milliseconds per frame (median of 200 frames, laptop CPU)")
    ax.set_title(f"Per-frame processing time: {p['total']:.1f} ms ≈ {1000 / p['total']:.0f} fps", fontsize=10)
    save(fig, "G07_compute.png")


def g_annotation():
    h = M["v5"]["hat"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    names = ["hand clicks", "RGB-mapped joints\n(pipeline)", "MediaPipe on\nthermal image"]
    vals = [h["annotation"], h["rgb_mapped"], h["thermal_mediapipe"]]
    ax.bar(names, vals, color=[COL["data"], COL["rgb"], COL["thermal"]], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f"{v:.1f} px", ha="center")
    ax.set_ylabel("own RMS scatter (px), bias removed")
    ax.set_title("Three-cornered hat: random error of each joint estimate (upper bound for clicks)", fontsize=10)
    save(fig, "G08_annotation_hat.png")


def copy_existing():
    pairs = {
        "reports/results/fig_reference_d2jtd.png": "R01_healthy_reference.png",
        "reports/results/fig_repeatability.png": "R02_repeatability_bland_altman.png",
        "reports/results/fig_open_close.png": "R03_static_vs_open_close.png",
        "reports/v7_models/fig_model_comparison.png": "R04_model_comparison.png",
        "reports/v7_models/fig_learning_curve.png": "R05_learning_curve.png",
        "reports/v5_annotation_qc/contact_sheet.png": "R06_annotation_contact_sheet.png",
        "reports/audit/figures/S1_left_open_close_samples.png": "R07_audit_sample_frames.png",
        "reports/audit/figures/S1_left_static_timestamps.png": "R08_audit_timestamps.png",
        "reports/joint_views/S2_right_static_f399.png": "O01_joint_view_static.png",
        "reports/joint_views/S1_left_open_close_f9.png": "O02_joint_view_fist.png",
        "data/processed_v8/qc/S2_right_static.png": "O03_qc_overlay.png",
        "reports/results/edge_demo_S2_right_static_warm.png": "O04_edge_demo_flag.png",
    }
    for s, d in pairs.items():
        if Path(s).is_file():
            shutil.copy(s, OUT / d)
            print("copied", d)
        else:
            print("MISSING", s)


if __name__ == "__main__":
    architecture(); pipeline(); dataset_tree(); transform_fig(); landmarks_fig(); roi_fig(); segmentation_fig()
    registration_fig(); loso_fig(); accessory_fig()
    workflow("F12_upload_workflow.png", ["Import ZIP /\nfolder / single", "Safe extract\n+ name matching", "Manual\nmatching", "Validate\n(5 statuses)",
                                         "Calibration\ncheck", "Process\n(pipeline)", "Features\n.csv"],
             [COL["out"], COL["out"], COL["out"], COL["feat"], COL["calib"], COL["thermal"], COL["feat"]])
    workflow("F13_test_workflow.png", ["Processed\nnew dataset", "Load saved\nmodel bundle", "Lock inputs\n(SHA-256)", "Score + quality\ncategories",
                                       "Match labels\n(CSV/Excel)", "Guarded\nmetrics", "PDF / Excel\nreport"],
             [COL["feat"], COL["ml"], COL["out"], COL["ml"], COL["out"], COL["out"], COL["out"]],
             "No refitting, rescaling or threshold changes on test data · metrics only for label sets that support them")
    workflow("F14_app_modules.png", ["Dashboard", "Dataset upload\n+ validation", "Calibration", "Processing", "Models", "Independent\ntest",
                                     "Joint view", "Explainability", "Live\n(simulation)", "Reports\n(viewer)"],
             [COL["out"], COL["out"], COL["calib"], COL["thermal"], COL["ml"], COL["out"], COL["rgb"], COL["feat"], COL["data"], COL["out"]])
    g_localisation(); g_registration(); g_repeatability(); g_ablation(); g_synthetic(); g_visibility(); g_compute(); g_annotation()
    copy_existing()
