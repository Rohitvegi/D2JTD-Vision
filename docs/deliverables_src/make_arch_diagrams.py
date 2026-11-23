"""Architecture diagrams A-E for the defence guide (implemented components only, from the source code).

Run from the project root:
  .venv312\\Scripts\\python.exe docs/deliverables_src/make_arch_diagrams.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from make_figures import COL, INK, INK2, save  # noqa: E402


def setup(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.grid(False)
    return fig, ax


def B(ax, x, y, w, h, title, sub, col, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=col, ec=col, lw=1.4))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs, color="white", weight="bold")
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=fs - 2, color="white")
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, color="white", weight="bold")
    return (x, y, w, h)


def A(ax, a, b, side="r", label=None, color=INK2):
    (x1, y1, w1, h1), (x2, y2, w2, h2) = a, b
    if side == "r":
        p, q = (x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2)
    elif side == "d":
        p, q = (x1 + w1 / 2, y1), (x2 + w2 / 2, y2 + h2)
    elif side == "u":
        p, q = (x1 + w1 / 2, y1 + h1), (x2 + w2 / 2, y2)
    else:  # "rd": right edge to top of target
        p, q = (x1 + w1, y1 + h1 / 2), (x2 + w2 / 2, y2 + h2)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=color, lw=1.4,
                                 connectionstyle="arc3,rad=0" if side in "rdu" else "angle,angleA=0,angleB=90"))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, label, ha="center", fontsize=8, color=INK2, style="italic")


def lane(ax, y, h, text, color):
    ax.add_patch(FancyBboxPatch((0.1, y), 0.0, h, boxstyle="square,pad=0", ec="none"))
    ax.text(0.15, y + h / 2, text, rotation=90, ha="center", va="center", fontsize=9, color=color, weight="bold")


# ---------------------------------------------------------------- A high level
def high_level():
    fig, ax = setup(16, 5.2)
    rgb = B(ax, 0.4, 3.2, 2.2, 1.1, "RGB video", "MP4, hand photo", COL["data"])
    npy = B(ax, 0.4, 0.9, 2.2, 1.1, "Radiometric .npy", "°C per pixel", COL["data"])
    aud = B(ax, 3.1, 2.05, 2.2, 1.1, "Audit + validation", "match, check, pair frames", COL["out"])
    rb = B(ax, 5.8, 3.2, 2.3, 1.1, "RGB branch", "MediaPipe joints", COL["rgb"])
    tb = B(ax, 5.8, 0.9, 2.3, 1.1, "Thermal branch", "hand mask (Otsu)", COL["thermal"])
    fu = B(ax, 8.6, 2.05, 2.3, 1.1, "Fusion", "RGB→thermal mapping\n14 joint ROIs", COL["calib"])
    fe = B(ax, 11.4, 2.05, 2.0, 1.1, "Features", "D2JTD + spatial", COL["feat"])
    ml = B(ax, 13.8, 2.05, 2.0, 1.1, "Healthy-reference\nmodel", "", COL["ml"])
    A(ax, rgb, aud)
    A(ax, npy, aud)
    A(ax, aud, rb)
    A(ax, aud, tb)
    A(ax, rb, fu)
    A(ax, tb, fu)
    A(ax, fu, fe)
    A(ax, fe, ml)
    ax.text(14.8, 1.6, "→ within range / potential anomaly\n+ joint explanation + PDF/Excel report", ha="center", va="top", fontsize=9, color=INK)
    ax.text(0.4, 4.85, "A. High-level system architecture", fontsize=14, color=INK, weight="bold")
    save(fig, "D_A_high_level.png")


# ---------------------------------------------------------------- B detailed
def detailed():
    fig, ax = setup(18, 9)
    ax.text(0.3, 8.6, "B. Detailed technical architecture (module · algorithm · library)", fontsize=14, color=INK, weight="bold")
    ax.text(0.3, 7.6, "RGB", fontsize=11, color=COL["rgb"], weight="bold")
    ax.text(0.3, 5.0, "Thermal", fontsize=11, color=COL["thermal"], weight="bold")
    r0 = B(ax, 1.3, 7.0, 2.3, 1.2, "Decode in order", "OpenCV VideoCapture\ncrop letterbox x140–500", COL["rgb"], 9.5)
    r1 = B(ax, 4.0, 7.0, 2.3, 1.2, "Hand landmarks", "MediaPipe Tasks\n21 pts (x, y, z)", COL["rgb"], 9.5)
    r2 = B(ax, 6.7, 7.0, 2.3, 1.2, "Quality + visibility", "finger order, size,\nfolded / covered", COL["rgb"], 9.5)
    r3 = B(ax, 9.4, 7.0, 2.3, 1.2, "RGB hand mask", "GrabCut seeded\nby landmarks", COL["rgb"], 9.5)
    t0 = B(ax, 1.3, 4.4, 2.3, 1.2, "Read .npy", "NumPy mmap\nfloat32 °C 640×480", COL["thermal"], 9.5)
    t1 = B(ax, 4.0, 4.4, 2.3, 1.2, "Thermal hand mask", "Otsu + hysteresis\nlargest component", COL["thermal"], 9.5)
    t2 = B(ax, 6.7, 4.4, 2.3, 1.2, "Eroded core", "9×9 ellipse\n(no edge pixels)", COL["thermal"], 9.5)
    c1 = B(ax, 12.1, 5.7, 2.5, 1.2, "Calibration (v9)", "scale, rotation, per-frame\nshift (outline distance)", COL["calib"], 9.5)
    c2 = B(ax, 15.0, 5.7, 2.5, 1.2, "Joint mapping", "RGB joints → thermal\np_th = s(kp − c)+c+t + shift", COL["calib"], 9.5)
    o1 = B(ax, 15.0, 3.2, 2.5, 1.2, "14 joint ROIs", "finger-width radius\n+ dorsum polygon", COL["thermal"], 9.5)
    o2 = B(ax, 12.1, 3.2, 2.5, 1.2, "ROI statistics", "median, mean, P90, SD\ncold-band flag", COL["thermal"], 9.5)
    f1 = B(ax, 9.4, 3.2, 2.3, 1.2, "Per-frame table", "pandas CSV\none row per frame", COL["out"], 9.5)
    f2 = B(ax, 9.4, 0.8, 2.3, 1.2, "Features", "129 per recording\nD2JTD + spatial (32)", COL["feat"], 9.5)
    q1 = B(ax, 12.1, 0.8, 2.5, 1.2, "Quality control", "valid frames, ROI use,\nDice, out-of-range", COL["feat"], 9.5)
    m1 = B(ax, 15.0, 0.8, 2.5, 1.2, "Anomaly model", "Ledoit–Wolf Mahalanobis\nscikit-learn, θ = P95", COL["ml"], 9.5)
    for a, b in ((r0, r1), (r1, r2), (r2, r3), (t0, t1), (t1, t2)):
        A(ax, a, b)
    A(ax, r3, c1, "rd")
    A(ax, t2, c1)
    A(ax, c1, c2)
    A(ax, c2, o1, "d")
    A(ax, t2, o2, "rd")
    ax.add_patch(FancyArrowPatch((15.0, 3.8), (14.6, 3.8), arrowstyle="-|>", mutation_scale=13, color=INK2, lw=1.4))
    ax.add_patch(FancyArrowPatch((12.1, 3.8), (11.7, 3.8), arrowstyle="-|>", mutation_scale=13, color=INK2, lw=1.4))
    A(ax, f1, f2, "d")
    A(ax, f2, q1)
    A(ax, q1, m1)
    ax.text(12.1, 7.55, "branches combine here", fontsize=9, color=COL["calib"], weight="bold")
    save(fig, "D_B_detailed.png")


# ---------------------------------------------------------------- C training
def training():
    fig, ax = setup(17, 7.2)
    ax.text(0.3, 6.8, "C. Training pipeline (no neural network is trained; MediaPipe is used pre-trained)", fontsize=14, color=INK, weight="bold")
    ax.text(0.3, 5.9, "Model", fontsize=11, color=COL["ml"], weight="bold")
    a = B(ax, 0.3, 4.2, 2.4, 1.3, "34 static hands", "17 subjects × L/R\nfeature vectors", COL["feat"])
    b = B(ax, 3.1, 4.2, 2.4, 1.3, "Outer LOSO", "hold out 1 subject\n(both hands)", COL["calib"])
    c = B(ax, 5.9, 4.2, 2.4, 1.3, "Inner LOSO", "threshold θ = P95 of\nheld-out healthy scores", COL["calib"])
    d = B(ax, 8.7, 4.2, 2.4, 1.3, "Fit on 16 subjects", "median impute,\nLedoit–Wolf covariance", COL["ml"])
    e = B(ax, 11.5, 4.2, 2.4, 1.3, "Score held-out", "real (specificity)\n+ simulated warming", COL["ml"])
    g = B(ax, 14.3, 4.2, 2.4, 1.3, "Final model", "fit on all 17, θ from LOSO\nsaved bundle + metadata", COL["out"])
    for p, q in ((a, b), (b, c), (c, d), (d, e), (e, g)):
        A(ax, p, q)
    ax.text(0.3, 2.85, "Joint correction", fontsize=11, color=COL["rgb"], weight="bold")
    h = B(ax, 0.3, 1.0, 2.4, 1.3, "Manual annotation", "777 joints clicked on\n136 thermal frames", COL["data"])
    i = B(ax, 3.1, 1.0, 2.4, 1.3, "LOSO split", "learn on 16 subjects,\ntest on the 17th", COL["calib"])
    j = B(ax, 5.9, 1.0, 2.4, 1.3, "Learn shift", "median along-finger\noffset (MCP, PIP)", COL["rgb"])
    k = B(ax, 8.7, 1.0, 2.4, 1.3, "Error on held-out", "Euclidean px error\n+ temperature cost", COL["ml"])
    l2 = B(ax, 11.5, 1.0, 2.4, 1.3, "Saved correction", "landmark_correction.json\n(all subjects)", COL["out"])
    for p, q in ((h, i), (i, j), (j, k), (k, l2)):
        A(ax, p, q)
    save(fig, "D_C_training.png")


# ---------------------------------------------------------------- D inference on new data
def inference():
    fig, ax = setup(17, 6.4)
    ax.text(0.3, 6.0, "D. New-data inference architecture (application)", fontsize=14, color=INK, weight="bold")
    s = [("Upload", "ZIP / folder /\nsingle recording", COL["data"]), ("Match files", "filename profile\nor manual", COL["out"]),
         ("Validate", "16 checks →\n5 statuses", COL["out"]), ("Process", "same pipeline\nown folder", COL["thermal"]),
         ("Features", "same 32 features\nsame order", COL["feat"]), ("Load bundle", "frozen model\n+ θ + ranges", COL["ml"])]
    bx = []
    for n, (t, sub, c) in enumerate(s):
        bx.append(B(ax, 0.3 + n * 2.8, 3.6, 2.4, 1.3, t, sub, c))
    for p, q in zip(bx[:-1], bx[1:]):
        A(ax, p, q)
    t = [("Quality gate", "frames, ROIs, Dice", COL["feat"]), ("Range gate", "inside training\nexperience?", COL["feat"]),
         ("Score", "Mahalanobis\ndistance vs θ", COL["ml"]), ("Category", "Normal / Potential\nanomaly / …", COL["ml"]),
         ("Explain", "per-joint deviation\n(SD units)", COL["ml"]), ("Report", "PDF · Excel · CSV\n+ SHA-256 lock", COL["out"])]
    bt = []
    for n, (tt, sub, c) in enumerate(t):
        bt.append(B(ax, 14.3 - n * 2.8, 0.9, 2.4, 1.3, tt, sub, c))
    A(ax, bx[-1], bt[0], "d")
    for p, q in zip(bt[:-1], bt[1:]):
        ax.add_patch(FancyArrowPatch((p[0], p[1] + p[3] / 2), (q[0] + q[2], q[1] + q[3] / 2), arrowstyle="-|>", mutation_scale=13, color=INK2, lw=1.4))
    ax.text(14.3 + 1.2, 0.55, "fails → 'Insufficient data' /\n'Low-confidence' / 'Invalid'", ha="center", va="top", fontsize=8, color=INK2)
    ax.text(11.5 + 1.2, 0.55, "outside → 'Outside model\nexperience' (not scored)", ha="center", va="top", fontsize=8, color=INK2)
    save(fig, "D_D_inference.png")


# ---------------------------------------------------------------- E evaluation
def evaluation():
    fig, ax = setup(17, 7.6)
    ax.text(0.3, 7.2, "E. Evaluation architecture: each metric is paired with its own reference", fontsize=14, color=INK, weight="bold")
    rows = [("Hand-clicked joints (777)", "Euclidean error (px), along/across bias,\ntemperature cost (°C)", "Joint localisation"),
            ("Thermal hand mask (held-out frames)", "Dice, IoU", "Registration"),
            ("First vs second half of recording", "ICC(3,1), MDC95 (°C), Bland–Altman", "Repeatability"),
            ("Unseen healthy hands (LOSO)", "Specificity, false-positive rate", "Model on real data"),
            ("Same hands + simulated warming", "Sensitivity per pattern and ΔT", "Model response (simulated)"),
            ("Uploaded labels (when available)", "Confusion matrix, accuracy, F1, ROC-AUC\n(needs both classes; warns if n < 10 per class)", "Independent test")]
    for n, (ref, met, what) in enumerate(rows):
        y = 6.0 - n * 1.0
        r = B(ax, 0.3, y, 4.6, 0.8, ref, "", COL["data"], 9.5)
        m = B(ax, 5.6, y, 6.4, 0.8, met, "", COL["ml"], 9.5)
        w = B(ax, 12.7, y, 4.0, 0.8, what, "", COL["out"], 9.5)
        A(ax, r, m)
        A(ax, m, w)
    ax.text(0.3, 0.15, "Timing: wall-clock ms per frame and memory (psutil) on the laptop CPU", fontsize=9.5, color=INK2)
    save(fig, "D_E_evaluation.png")


if __name__ == "__main__":
    high_level()
    detailed()
    training()
    inference()
    evaluation()
