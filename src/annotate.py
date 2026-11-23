"""RQ2 annotation tool: click the true joint positions on the THERMAL image.

Guide with a worked example: docs/annotation_guide.md

The window shows two panels:  LEFT = thermal image (click here)   RIGHT = RGB photo (look only).
For each recording, the two frames reserved by src.run_pipeline (~10 % and ~90 % through) are
shown. Click 6 joints in the order printed in the black banner:
    1 index knuckle (MCP)   2 index middle joint (PIP)
    3 middle knuckle        4 middle middle joint
    5 ring knuckle          6 ring middle joint
Keys:  s = joint not visible (skip it)   u = undo last click   n = skip whole frame
       q = save and quit (you can resume later; finished frames are not shown again)
Clicks are saved after every frame to data/annotations/joints_clicked.csv (.npy pixel coordinates).
Then rerun `python -m src.analysis` to get the landmark error.

Annotate WITHOUT looking at the pipeline's QC images, so the reference is independent.

Usage:
  .venv312\\Scripts\\python.exe -m src.annotate                         # all recordings
  .venv312\\Scripts\\python.exe -m src.annotate --only S2_right_static  # one recording
  .venv312\\Scripts\\python.exe -m src.annotate --redo S2_right_static:78   # re-click one frame
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.io_utils import iter_frames, load_config, load_recordings
from src.run_pipeline import annotation_frames

ORDER = ["index_MCP", "index_PIP", "middle_MCP", "middle_PIP", "ring_MCP", "ring_PIP"]
PROMPT = {
    "index_MCP": "INDEX finger (next to thumb) - KNUCKLE (MCP)",
    "index_PIP": "INDEX finger (next to thumb) - MIDDLE joint (PIP)",
    "middle_MCP": "MIDDLE finger (longest) - KNUCKLE (MCP)",
    "middle_PIP": "MIDDLE finger (longest) - MIDDLE joint (PIP)",
    "ring_MCP": "RING finger - KNUCKLE (MCP)",
    "ring_PIP": "RING finger - MIDDLE joint (PIP)",
}
OUT = Path("data/annotations/joints_clicked.csv")
W_T = 480                                      # thermal panel width; clicks with x >= W_T are ignored


def render(T: np.ndarray, rgb: np.ndarray, pts: list, hand: str) -> np.ndarray:
    lo, hi = np.percentile(T, 1), np.percentile(T, 99.7)
    th = cv2.applyColorMap(np.clip((T - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    for k, p in enumerate(pts):
        if p is not None:
            cv2.circle(th, (int(p[0]), int(p[1])), 5, (255, 255, 255), 2)
            cv2.putText(th, str(k + 1), (int(p[0]) + 7, int(p[1]) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    ph = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (th.shape[1], th.shape[0]), interpolation=cv2.INTER_AREA)
    img = np.hstack([th, ph])
    cv2.rectangle(img, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)
    nxt = PROMPT[ORDER[len(pts)]] if len(pts) < len(ORDER) else "done"
    head = (f"{len(pts) + 1}/{len(ORDER)}  click on the LEFT image:  {nxt}" if len(pts) < len(ORDER)
            else f"{len(ORDER)}/{len(ORDER)}  all joints clicked - saving and moving to the next frame")
    cv2.putText(img, head, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(img, f"{hand.upper()} hand, back of hand towards camera  |  right panel = photo, for reference only  |"
                     "  s skip joint  u undo  n skip frame  q save+quit",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.line(img, (W_T, 50), (W_T, img.shape[0]), (255, 255, 255), 1)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--redo", nargs="*", default=[], metavar="REC:FRAME",
                    help="discard saved clicks for these frames and annotate them again, e.g. S1_left_static:81")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config))
    stride = int(cfg["pipeline"]["frame_stride"])
    box = tuple(cfg["pipeline"]["video_box"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(OUT) if OUT.is_file() else pd.DataFrame(columns=["rec_id", "frame", "joint", "x", "y"])
    for item in args.redo:
        rec_id, frame = item.rsplit(":", 1)
        drop = (done.rec_id == rec_id) & (done.frame == int(frame))
        print(f"--redo {item}: discarding {int(drop.sum())} saved clicks")
        done = done[~drop]
    done_keys = set(zip(done.rec_id, done.frame.astype(int)))
    rows = done.to_dict("records")

    def save():
        pd.DataFrame(rows, columns=["rec_id", "frame", "joint", "x", "y"]).to_csv(OUT, index=False)

    save()
    win = "annotate: LEFT thermal (click) | RIGHT photo (look)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    clicks: list = []
    cv2.setMouseCallback(win, lambda e, x, y, *_: clicks.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN and x < W_T and y > 50 else None)
    infos = load_recordings(cfg, only=args.only)
    todo = sum(1 for inf in infos for f in annotation_frames(inf.n_frames, stride) if (inf.rec_id, f) not in done_keys)
    print(f"{todo} frames to annotate ({len(done_keys)} already done). Press q any time to save and stop.")
    n_done = 0
    for info in infos:
        frames = [f for f in annotation_frames(info.n_frames, stride) if (info.rec_id, f) not in done_keys]
        if not frames:
            continue
        for f, rgb, T in iter_frames(info, frames, box):
            pts: list = []
            clicks.clear()
            while len(pts) < len(ORDER):
                while clicks and len(pts) < len(ORDER):
                    pts.append(clicks.pop(0))
                cv2.imshow(win, render(T, rgb, pts, info.rec.hand))
                if len(pts) >= len(ORDER):
                    cv2.waitKey(250)                       # show the last click briefly
                    break
                k = cv2.waitKey(30) & 0xFF
                if k == ord("s"):
                    pts.append(None)
                elif k == ord("u") and pts:
                    pts.pop()
                elif k == ord("n"):
                    pts = None
                    break
                elif k == ord("q"):
                    save()
                    cv2.destroyAllWindows()
                    print(f"Saved. {n_done} frames annotated this session; {len(rows)} points in {OUT}")
                    return 0
            if pts:
                for name, p in zip(ORDER, pts):
                    if p is not None:
                        rows.append(dict(rec_id=info.rec_id, frame=int(f), joint=name, x=float(p[0]), y=float(p[1] - 0)))
                save()
                n_done += 1
                print(f"  {info.rec_id} frame {f} saved ({n_done}/{todo})")
    cv2.destroyAllWindows()
    print(f"All done. {len(rows)} points in {OUT}. Now run: .venv312\\Scripts\\python.exe -m src.analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
