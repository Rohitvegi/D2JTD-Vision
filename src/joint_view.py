"""Joint view: RGB photo with the 14 joint positions and their measured temperatures, next to the
thermal image showing the ROI each temperature was read from.

  * Temperature label = median radiometric temperature of the joint ROI (deg C), exactly the value
    the pipeline stored for that frame (no recomputation).
  * Dot colour = D2JTD (back of hand minus joint): red = joint warmer than the back of the hand,
    blue = cooler, grey = about equal. Joints without a usable measurement are drawn hollow with "n/a".
  * Joint positions on the RGB photo are MediaPipe's landmarks (recovered exactly from the stored
    thermal coordinates with the inverse transform). The thermal panel shows the ROIs actually used,
    which include the learned along-finger correction for the RGB->thermal mapping.

Works for the original dataset (processed folder) and for uploaded datasets.

Usage:
  .venv312\\Scripts\\python.exe -m src.joint_view --rec S2_right_static
  .venv312\\Scripts\\python.exe -m src.joint_view --rec S2_right_static --frame 300 --processed-dir data/processed_v6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from src import landmark_correction as LC  # noqa: E402
from src import registration as R  # noqa: E402
from src.io_utils import crop_to_npy_scale, iter_frames, load_config, load_recordings  # noqa: E402
from src.landmarks import FINGER_LM, JOINTS, joint_visibility  # noqa: E402

OUT = Path("reports/joint_views")
DIVERGING = LinearSegmentedColormap.from_list("d2jtd", ["#d03b3b", "#e8e7e3", "#2a78d6"])   # warmer -> cooler
SHORT = {"MCP": "MCP", "PIP": "PIP", "DIP": "DIP", "IP": "IP"}


def find_info(rec_id: str, cfg: dict, dataset_id: str | None = None):
    if dataset_id:
        from src.dataset_manager import infos_from_manifest, load_manifest
        return infos_from_manifest(load_manifest(dataset_id), [rec_id])[0]
    return load_recordings(cfg, only=[rec_id])[0]


def pick_frame(df: pd.DataFrame, frame: int | None) -> int:
    ok = df[df["frame_valid"].fillna(False).astype(bool)]
    if ok.empty:
        raise ValueError("no valid frames in this recording")
    if frame is None:
        return int(ok.frame.iloc[len(ok) // 2])                      # middle of the usable frames
    return int(ok.frame.iloc[(ok.frame - frame).abs().argmin()])    # nearest analysed frame


def render(rec_id: str, processed_dir: str | Path, cfg: dict, frame: int | None = None,
           dataset_id: str | None = None, out_png: str | Path | None = None, video_box=None) -> tuple[Path, pd.DataFrame]:
    """Draw the joint view for one analysed frame. Returns (png path, per-joint table)."""
    processed_dir = Path(processed_dir)
    df = pd.read_csv(processed_dir / "frames" / f"{rec_id}.csv")
    reg = pd.read_csv(processed_dir / "registration.csv").set_index("rec_id").loc[rec_id]
    f = pick_frame(df, frame)
    row = df[df.frame == f].iloc[0]
    box = tuple(video_box or cfg["pipeline"]["video_box"])
    if np.isfinite(row.get("cal_tx", np.nan)):             # per-frame (learned) calibration was used
        M = R.matrix(crop_to_npy_scale(box), row["cal_scale"], row["cal_tx"], row["cal_ty"], row.get("cal_angle", 0.0) or 0.0)
    else:
        M = R.matrix(crop_to_npy_scale(box), reg.scale, reg.tx, reg.ty)
    Minv = cv2.invertAffineTransform(M)

    raw = np.array([[row[f"lm{j}_x"], row[f"lm{j}_y"]] for j in range(21)], float)      # thermal coords
    use_corr = bool(row.get("lm_corrected", False))
    pts_t = LC.apply(raw, LC.load()) if use_corr else raw                                 # joint centres used
    # On the photo, MediaPipe's own landmarks ARE the RGB joint positions (raw maps back exactly).
    # The along-finger correction compensates the RGB->thermal mapping, so it applies to the thermal panel only.
    raw_rgb = raw @ Minv[:, :2].T + Minv[:, 2]
    pts_rgb = raw_rgb

    info = find_info(rec_id, cfg, dataset_id)
    rgb, T = next((r, t) for i, r, t in iter_frames(info, [f], box))

    # visibility in the RGB image (folded under the hand / covered / end joint tucked under)
    z = np.array([row.get(f"lm{j}_z", np.nan) for j in range(21)], float)
    vis = joint_visibility(raw_rgb, z if np.isfinite(z).all() else None)
    if f"{JOINTS[0][0]}_vis" in row:                     # pipeline decision wins when it was stored
        vis = {n: (vis[n] or "not visible in RGB") if row[f"{n}_vis"] == 0 else "" for n in vis}
    visible = {n: not reason for n, reason in vis.items()}

    rows = []
    for name, lm, finger, kind in JOINTS:
        t = row.get(f"{name}_T", np.nan) if visible[name] else np.nan
        rows.append(dict(joint=name, visible="yes" if visible[name] else f"no — {vis[name]}", temperature_C=t,
                         d2jtd_C=row["dorsum_T"] - t if np.isfinite(t) else np.nan,
                         roi_radius_px=row.get(f"{name}_r", np.nan), on_hand_pct=100 * row.get(f"{name}_in", np.nan),
                         x_rgb=pts_rgb[lm, 0], y_rgb=pts_rgb[lm, 1]))
    J = pd.DataFrame(rows)
    n_hidden = int(sum(not v for v in visible.values()))
    # landmark-level visibility for drawing the skeleton: a fingertip counts as visible with its DIP / thumb IP
    lm_vis = np.ones(21, bool)
    for name, lm, finger, kind in JOINTS:
        lm_vis[lm] = visible[name]
    for tip, joint_lm in ((4, 3), (8, 7), (12, 11), (16, 15), (20, 19)):
        lm_vis[tip] = lm_vis[joint_lm]
    lim = max(1.0, np.nanmax(np.abs(J.d2jtd_C))) if J.d2jtd_C.notna().any() else 1.0
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, ax = plt.subplots(1, 2, figsize=(13, 8.2), gridspec_kw=dict(width_ratios=[1, 1.0]))
    # ---- RGB with joints and temperatures
    ax[0].imshow(rgb)
    for chain in FINGER_LM.values():
        c = [0] + chain
        for a, b in zip(c[:-1], c[1:]):                  # only bones whose both ends are visible
            if lm_vis[a] and lm_vis[b]:
                ax[0].plot(raw_rgb[[a, b], 0], raw_rgb[[a, b], 1], "-", color="white", lw=1.2, alpha=0.7)
    for (name, lm, finger, kind), r in zip(JOINTS, J.itertuples()):
        if not visible[name]:
            continue                                     # not visible in the photo: no point, no temperature
        x, y = pts_rgb[lm]
        if np.isfinite(r.temperature_C):
            ax[0].scatter(x, y, s=90, c=[DIVERGING(norm(r.d2jtd_C))], edgecolors="black", linewidths=1.2, zorder=3)
            label = f"{r.temperature_C:.1f}°"
        else:
            ax[0].scatter(x, y, s=90, facecolors="none", edgecolors="white", linewidths=1.5, zorder=3)
            label = "n/a"
        # label just to the upper right of its own dot, tied to it with a thin connector
        ax[0].annotate(label, (x, y), xytext=(9, 7), textcoords="offset points", fontsize=8, color="#0b0b0b",
                       bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9),
                       arrowprops=dict(arrowstyle="-", color="#0b0b0b", lw=0.6, shrinkA=0, shrinkB=4), zorder=4)
    ax[0].scatter(*pts_rgb[0], s=30, c="white", edgecolors="black", zorder=3)
    hidden_txt = f" · {n_hidden} joint(s) not visible, not shown" if n_hidden else ""
    ax[0].set_title(f"RGB photo — visible joints and ROI temperature (°C)\nback of hand {row['dorsum_T']:.1f} °C{hidden_txt}",
                    fontsize=10)
    ax[0].axis("off")
    # ---- thermal with ROIs
    im = ax[1].imshow(T, cmap="inferno")
    for (name, lm, finger, kind), r in zip(JOINTS, J.itertuples()):
        if not visible[name]:
            continue
        x, y = pts_t[lm]
        rad = r.roi_radius_px if np.isfinite(r.roi_radius_px) else 10
        col = "white" if np.isfinite(r.temperature_C) else "#7a7973"
        ax[1].add_patch(plt.Circle((x, y), rad, fill=False, color=col, lw=1.4))
        ax[1].annotate(f"{name.split('_')[0][:3]} {SHORT[kind]}", (x, y), xytext=(0, 0), textcoords="offset points",
                       fontsize=6, color="white", ha="center", va="center")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.035, pad=0.02)
    cb.set_label("temperature (°C)")
    ax[1].set_title("Thermal (radiometric) — circles are the ROIs the temperatures come from", fontsize=10)
    ax[1].axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
    cb2 = fig.colorbar(sm, ax=ax[0], fraction=0.035, pad=0.02)
    cb2.set_label("D2JTD = back of hand − joint (°C)\nred: joint warmer · blue: joint cooler")
    fig.suptitle(f"{rec_id} — frame {f} (t = {row['t_s']:.1f} s)"
                 + ("" if use_corr else "  [joint correction not applied]"), fontsize=12)
    fig.tight_layout()
    out_png = Path(out_png) if out_png else OUT / f"{rec_id}_f{f}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png, J


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--frame", type=int, default=None, help="frame number (nearest analysed frame is used)")
    ap.add_argument("--processed-dir", default=None, help="default: data/processed_v8 if present, else data/processed_v6")
    ap.add_argument("--dataset", default=None, help="uploaded dataset id (uses its processed folder)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config))
    proc = Path(args.processed_dir) if args.processed_dir else (
        Path("data/processed_v8") if Path("data/processed_v8/registration.csv").is_file() else Path("data/processed_v6"))
    if args.dataset:
        from src.dataset_manager import DATASETS
        proc = DATASETS / args.dataset / "processed"
    png, J = render(args.rec, proc, cfg, args.frame, args.dataset)
    print(J[["joint", "visible", "temperature_C", "d2jtd_C", "on_hand_pct"]].round(2).to_string(index=False))
    print("saved", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
