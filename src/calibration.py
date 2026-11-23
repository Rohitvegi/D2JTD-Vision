"""RGB <-> thermal calibration: FIND it from the data, APPLY it, VERIFY it.

One module for everything that needs the camera mapping:

    from src.calibration import find_calibration, Calibration
    cal = find_calibration([("S1_rgb.mp4", "S1_temp.npy"), ...])   # learns the mapping from the recordings
    cal.save("models/calibration/my_camera.json")
    cal = Calibration.load("models/calibration/my_camera.json")
    xy_thermal = cal.rgb_to_thermal(xy_rgb, landmarks=pts)          # points (e.g. joints) RGB -> thermal pixels
    xy_rgb     = cal.thermal_to_rgb(xy_thermal, landmarks=pts)      # and back
    aligned    = cal.warp_rgb_to_thermal(rgb_crop, landmarks=pts)   # whole RGB image on the thermal grid
    report     = verify(cal, "S9_rgb.mp4", "S9_temp.npy", out_png="check.png")

What `find_calibration` determines (no calibration target needed; the hand itself is the target):
  1. image area of the RGB video (letterbox bars removed), from several frames;
  2. orientation check: the radiometric frames must have the same aspect as the RGB image (a rotated or
     transposed export is reported, not silently mis-mapped);
  3. scale s, by a free search over many frames (hand-outline overlap, Dice);
  4. rotation angle (in-plane tilt between the two sensors), searched -10 deg .. +10 deg and refined;
  5. shift t per frame, learned from the hand seen in RGB (size ~ distance, position) with a robust
     regression, so parallax is followed when the hand moves (src.calibration_model.ShiftModel);
  6. self-check on frames NOT used for fitting: Dice and a quality grade (good / acceptable / poor).

Transform (crop pixels -> radiometric pixels):  p_th = s * Rot(angle) (k p - c) + c + t(frame),
with k = radiometric width / RGB image width and c the radiometric image centre.

CLI:
  python -m src.calibration find --rgb a.mp4 --npy a.npy [--rgb b.mp4 --npy b.npy ...] --name my_camera
  python -m src.calibration find --subjects S1 S2 S3 S4 S5 --name flirone_auto      (original dataset)
  python -m src.calibration find --dataset <uploaded dataset id>
  python -m src.calibration verify --profile models/calibration/my_camera.json --rgb x.mp4 --npy x.npy
  python -m src.calibration apply  --profile ... --rgb x.mp4 --npy x.npy --frame 300   (joints RGB -> thermal + overlay)
  python -m src.calibration selftest     (recovers known distortions of real recordings)
  python -m src.calibration benchmark    (calibrate on S1-S5, test on unseen S6-S17 against the clicked joints)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src import landmarks as L
from src import registration as R
from src.calibration_model import ShiftModel, frame_features
from src.segmentation import dice, rgb_mask, thermal_mask

CALIB_DIR = Path("models/calibration")
REPORT_DIR = Path("reports/v9_calibration")
LANDMARK_MODEL = "models/hand_landmarker.task"


class CalibrationError(RuntimeError):
    pass


# --------------------------------------------------------------------------- the calibration object
@dataclass
class Calibration:
    name: str
    video_hw: list                     # full RGB video frame (H, W)
    video_box: list                    # (y0, y1, x0, x1) image area inside the video frame
    npy_hw: list                       # radiometric frame (H, W)
    scale: float
    angle_deg: float
    shift_model: dict                  # ShiftModel.to_dict()
    ref_features: list                 # median [hand_size, cx, cy] of the calibration frames
    quality: dict = field(default_factory=dict)
    method: str = "find_calibration (hand-outline, learned per-frame shift)"
    created: str = ""

    # ---- geometry
    @property
    def k(self) -> float:
        return self.npy_hw[1] / (self.video_box[3] - self.video_box[2])

    def shift(self, landmarks: np.ndarray | None = None) -> np.ndarray:
        x = frame_features(landmarks) if landmarks is not None else np.asarray(self.ref_features, float)
        return ShiftModel.from_dict(self.shift_model).predict(x)[0]

    def matrix(self, landmarks: np.ndarray | None = None) -> np.ndarray:
        """2x3 matrix RGB-crop -> radiometric pixels (per frame if the frame's 21 landmarks are given)."""
        t = self.shift(landmarks)
        return R.matrix(self.k, self.scale, float(t[0]), float(t[1]), self.angle_deg)

    # ---- apply
    def crop(self, frame: np.ndarray) -> np.ndarray:
        y0, y1, x0, x1 = self.video_box
        return frame[y0:y1, x0:x1]

    def rgb_to_thermal(self, xy, landmarks=None, full_frame: bool = False) -> np.ndarray:
        """Map RGB pixel coordinates (N x 2) to radiometric pixel coordinates."""
        p = np.atleast_2d(np.asarray(xy, float))
        if full_frame:
            p = p - [self.video_box[2], self.video_box[0]]
        return R.map_points(p, self.matrix(landmarks))

    def thermal_to_rgb(self, xy, landmarks=None, full_frame: bool = False) -> np.ndarray:
        """Inverse mapping: radiometric pixel coordinates -> RGB pixel coordinates."""
        Minv = cv2.invertAffineTransform(self.matrix(landmarks))
        p = R.map_points(np.atleast_2d(np.asarray(xy, float)), Minv)
        return p + [self.video_box[2], self.video_box[0]] if full_frame else p

    def warp_rgb_to_thermal(self, rgb_crop: np.ndarray, landmarks=None) -> np.ndarray:
        """Resample the RGB image onto the radiometric grid (pixel (x, y) of the result = thermal pixel (x, y))."""
        return cv2.warpAffine(rgb_crop, self.matrix(landmarks), (self.npy_hw[1], self.npy_hw[0]), flags=cv2.INTER_LINEAR)

    def temperature_at(self, T: np.ndarray, xy_rgb, landmarks=None, radius: int = 3) -> np.ndarray:
        """Median temperature in a small disc around each RGB point after mapping (quick look-up helper)."""
        out = []
        for x, y in self.rgb_to_thermal(xy_rgb, landmarks):
            x0, x1 = max(0, int(x - radius)), min(T.shape[1], int(x + radius) + 1)
            y0, y1 = max(0, int(y - radius)), min(T.shape[0], int(y + radius) + 1)
            out.append(float(np.nanmedian(T[y0:y1, x0:x1])) if x1 > x0 and y1 > y0 else np.nan)
        return np.array(out)

    # ---- storage (also a calibration profile for the app / pipeline)
    def to_profile(self) -> dict:
        d = asdict(self)
        t0 = self.shift()
        d.update(mode="learned", fixed_scale=self.scale, rig_default=[self.scale, float(t0[0]), float(t0[1])],
                 camera=d.get("camera", "unknown (found from data)"))
        return d

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else CALIB_DIR / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.to_profile(), open(path, "w"), indent=1)
        return path

    @classmethod
    def from_profile(cls, d: dict) -> "Calibration":
        if "shift_model" not in d:                       # v3 profile: constant shift
            s, tx, ty = d["rig_default"]
            d = dict(d, scale=s, shift_model=ShiftModel(np.zeros((2, 3)), np.array([tx, ty])).to_dict(),
                     ref_features=[110.0, 180.0, 240.0])
        names = cls.__dataclass_fields__
        kw = {k: d[k] for k in names if k in d}
        kw.setdefault("angle_deg", 0.0)
        kw.setdefault("ref_features", [110.0, 180.0, 240.0])
        return cls(**kw)

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls.from_profile(json.load(open(path)))


# --------------------------------------------------------------------------- reading frames
def _video_frames(rgb_path, indices):
    cap = cv2.VideoCapture(str(rgb_path))
    want, i, out = set(int(x) for x in indices), 0, {}
    last = max(want) if want else -1
    while i <= last:
        if i in want:
            ok, f = cap.read()
            if not ok:
                break
            out[i] = f
        elif not cap.grab():
            break
        i += 1
    cap.release()
    return out


def _n_frames(rgb_path, npy_path):
    cap = cv2.VideoCapture(str(rgb_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return min(n, np.load(npy_path, mmap_mode="r").shape[0])


def image_box(rgb_path, n: int = 6, thr: int = 8) -> tuple[list, list]:
    """Image area (letterbox removed) from the union of several frames; returns (box, video_hw)."""
    cap = cv2.VideoCapture(str(rgb_path))
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frames = _video_frames(rgb_path, np.linspace(0, max(0, nf - 1), n).round().astype(int))
    if not frames:
        raise CalibrationError(f"cannot decode {rgb_path}")
    boxes = []
    for f in frames.values():                       # per frame, then the median edge: compression noise in
        g = f.max(axis=2)                           # single frames cannot widen the box
        rows, cols = np.where(g.max(axis=1) > thr)[0], np.where(g.max(axis=0) > thr)[0]
        if rows.size and cols.size:
            boxes.append([rows[0], rows[-1] + 1, cols[0], cols[-1] + 1])
    h, w = next(iter(frames.values())).shape[:2]
    box = [int(v) for v in np.median(np.array(boxes), axis=0).round()] if boxes else [0, h, 0, w]
    return box, [h, w]


# --------------------------------------------------------------------------- fitting
def fit_shift(rgb_m, th_m, k, s, angle, ds: int = 4) -> dict:
    """Best translation for one frame at a given scale and angle (phase correlation + local Dice refinement)."""
    H, W = th_m.shape
    hw_s = (H // ds, W // ds)
    th_s = cv2.resize(th_m.astype(np.float32), (hw_s[1], hw_s[0]), interpolation=cv2.INTER_AREA)
    win = cv2.createHanningWindow((hw_s[1], hw_s[0]), cv2.CV_32F)
    M = R.matrix(k, s, 0, 0, angle) / ds
    w = cv2.warpAffine(rgb_m.astype(np.float32), M, (hw_s[1], hw_s[0]), flags=cv2.INTER_LINEAR)
    (dx, dy), _ = cv2.phaseCorrelate(w * win, th_s * win)
    best = dict(tx=0.0, ty=0.0, dice=-1.0)
    for sx, sy in ((dx, dy), (-dx, -dy)):
        Ms = M.copy()
        Ms[:, 2] += (sx, sy)
        d = dice(cv2.warpAffine(rgb_m.astype(np.float32), Ms, (hw_s[1], hw_s[0])) > 0.5, th_s > 0.5)
        if d > best["dice"]:
            best = dict(tx=float(sx * ds), ty=float(sy * ds), dice=d)
    tx0, ty0 = best["tx"], best["ty"]
    for ddx in range(-4, 5, 2):
        for ddy in range(-4, 5, 2):
            d = dice(R.warp_mask(rgb_m, R.matrix(k, s, tx0 + ddx, ty0 + ddy, angle), (H, W)), th_m)
            if d > best["dice"]:
                best.update(tx=tx0 + ddx, ty=ty0 + ddy, dice=d)
    return best


CAP_PX = 10.0                     # outline distances are capped: a wrong wrist/forearm edge cannot dominate
# Grades: on the FLIROne reference data a correct calibration leaves ~3.8-4.2 px between the RGB and thermal hand
# outlines (edge blur, segmentation); good = within that range, acceptable = up to twice the reference spread.
GOOD_PX, OK_PX = 5.0, 8.0


def outline_points(mask: np.ndarray, n: int = 400) -> np.ndarray:
    """n points evenly along the outer contour of a binary mask, (x, y)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros((0, 2))
    c = max(cs, key=len)[:, 0, :].astype(float)
    return c[np.linspace(0, len(c) - 1, min(n, len(c))).astype(int)]


def outline_distance_map(mask: np.ndarray) -> np.ndarray:
    """Distance (px) of every pixel to the mask outline."""
    m = mask.astype(np.uint8)
    edge = m - cv2.erode(m, np.ones((3, 3), np.uint8))
    return cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3)


def outline_score(pts: np.ndarray, dmap: np.ndarray, M: np.ndarray) -> float:
    """Mean capped distance (px) from the mapped RGB outline to the thermal outline: 0 = perfect."""
    q = R.map_points(pts, M)
    H, W = dmap.shape
    x, y = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    d = np.full(len(q), CAP_PX)
    d[ok] = np.minimum(dmap[y[ok], x[ok]], CAP_PX)
    return float(d.mean())


class _Sample:
    """One calibration frame: features, masks, RGB outline and thermal outline-distance map (cached)."""

    def __init__(self, x, rgb_m, th_m):
        self.x, self.rgb_m, self.th_m = x, rgb_m, th_m
        self.pts = outline_points(rgb_m)
        self.dmap = outline_distance_map(th_m)

    def fit(self, k, s, a) -> dict:
        """Best shift at (s, a): phase-correlation start, then outline-distance refinement (2 px, then 1 px)."""
        b = fit_shift(self.rgb_m, self.th_m, k, s, a)
        tx, ty = b["tx"], b["ty"]
        best = outline_score(self.pts, self.dmap, R.matrix(k, s, tx, ty, a))
        for step, rng in ((2, 8), (1, 1)):
            cx, cy = tx, ty
            for dx in np.arange(-rng, rng + 1, step):
                for dy in np.arange(-rng, rng + 1, step):
                    sc = outline_score(self.pts, self.dmap, R.matrix(k, s, cx + dx, cy + dy, a))
                    if sc < best:
                        best, tx, ty = sc, cx + dx, cy + dy
        return dict(tx=float(tx), ty=float(ty), outline_px=best)


def _median_score(samples, k, s, a):
    return float(np.median([smp.fit(k, s, a)["outline_px"] for smp in samples]))


def collect_samples(recordings, box, n_frames: int, landmark_model: str = LANDMARK_MODEL, progress=None):
    """[(features, rgb_mask, thermal_mask)] from frames spread evenly over all recordings."""
    per = max(2, int(np.ceil(n_frames / len(recordings))))
    y0, y1, x0, x1 = box
    lm = L.make_landmarker(landmark_model, video=False)
    cv2.setRNGSeed(0)
    out = []
    for n, (rgb_path, npy_path) in enumerate(recordings, 1):
        seq = np.load(npy_path, mmap_mode="r")
        nf = _n_frames(rgb_path, npy_path)
        idx = np.linspace(0, nf - 1, per).round().astype(int)
        for i, bgr in _video_frames(rgb_path, idx).items():
            rgb = np.ascontiguousarray(bgr[y0:y1, x0:x1, ::-1])
            pts, _ = L.detect(lm, rgb)
            if pts is None:
                continue
            pts, fl = L.check_and_fix(pts, rgb.shape[:2])
            if not fl["ok"]:
                continue
            T = np.asarray(seq[i], np.float32)
            th, _ = thermal_mask(np.nan_to_num(T, nan=float(np.nanmedian(T))))
            out.append((frame_features(pts), rgb_mask(rgb, pts), th))
        if progress:
            progress(n, len(recordings), dict(rec_id=Path(rgb_path).stem))
    lm.close()
    return out


def find_calibration(recordings, name: str = "auto", n_frames: int = 60, landmark_model: str = LANDMARK_MODEL,
                     scales=np.arange(0.70, 1.81, 0.02), angles=np.arange(-10, 10.1, 1.0), progress=None) -> Calibration:
    """Find the RGB -> thermal calibration from paired recordings [(rgb_video, radiometric_npy), ...]."""
    recordings = [(Path(a), Path(b)) for a, b in recordings]
    if not recordings:
        raise CalibrationError("no recordings given")
    t_start = time.time()
    box, video_hw = image_box(recordings[0][0])
    npy_hw = list(np.load(recordings[0][1], mmap_mode="r").shape[1:])
    if len(npy_hw) != 2:
        raise CalibrationError(f"radiometric file must be (frames, H, W); got shape {npy_hw}")
    a_rgb = (box[3] - box[2]) / (box[1] - box[0])
    a_th = npy_hw[1] / npy_hw[0]
    if abs(a_rgb / a_th - 1) > 0.08:
        hint = " (looks rotated by 90° or transposed: use np.rot90 / transpose on the .npy)" if abs(a_rgb * a_th - 1) < 0.08 else ""
        raise CalibrationError(f"RGB image area {box[3] - box[2]}x{box[1] - box[0]} and radiometric {npy_hw[1]}x{npy_hw[0]} "
                               f"have different shapes{hint}")
    k = npy_hw[1] / (box[3] - box[2])
    samples = collect_samples(recordings, box, n_frames, landmark_model, progress)
    if len(samples) < 8:
        raise CalibrationError(f"only {len(samples)} frames with a detected hand; need at least 8")
    samples = [_Sample(*smp) for smp in samples]
    fit, test = samples[0::2], samples[1::2]
    sub = fit[:: max(1, len(fit) // 10)][:10]                       # subset for the parameter searches
    # Search by OUTLINE distance (finger contours fix the scale; extra warm area such as a forearm cannot pull it)
    # 1) coarse joint grid of scale x angle (scale and tilt interact), 2) refine the 3 best, separate starts,
    #    in two steps (0.02 / 0.5 deg, then 0.01 / 0.25 deg); the overall best wins
    coarse = {(float(s), float(a)): _median_score(sub, k, s, a) for s in scales[::2] for a in angles[::3]}
    s_scores = {f"{s:.2f}": min(v for (s2, _), v in coarse.items() if s2 == s) for s in sorted({s for s, _ in coarse})}
    a_scores = {a: min(v for (_, a2), v in coarse.items() if a2 == a) for a in sorted({a for _, a in coarse})}
    starts, picked = [], sorted(coarse, key=coarse.get)
    for g in picked:
        if all(abs(g[0] - s) > 0.06 or abs(g[1] - a) > 2.5 for s, a in starts):
            starts.append(g)
        if len(starts) == 3:
            break
    sc = {}
    for s0, a0 in starts:
        g1 = {(s0 + ds_, a0 + da): _median_score(sub, k, s0 + ds_, a0 + da)
              for ds_ in np.arange(-0.04, 0.041, 0.02) for da in np.arange(-1.5, 1.51, 0.5)}
        s1, a1 = min(g1, key=g1.get)
        g2 = {(s1 + ds_, a1 + da): _median_score(sub, k, s1 + ds_, a1 + da)
              for ds_ in (-0.01, 0.0, 0.01) for da in (-0.25, 0.0, 0.25)}
        sc.update(g1)
        sc.update(g2)
    s_best, a_best = min(sc, key=sc.get)
    if abs(a_best) < 0.5 and _median_score(sub, k, s_best, 0.0) <= sc[(s_best, a_best)] + 0.05:
        a_best = 0.0                                                 # no meaningful rotation: keep it exactly 0
    # 4) per-frame shift model
    per = [(smp.x, smp.fit(k, s_best, a_best)) for smp in fit]
    good = [(x, b) for x, b in per if b["outline_px"] <= 4.0]
    X = np.array([x for x, _ in good]) if good else np.zeros((0, 3))
    Y = np.array([[b["tx"], b["ty"]] for _, b in good]) if good else np.zeros((0, 2))
    if len(good) >= 10:
        model = ShiftModel().fit(X, Y)
    else:
        med = np.median([[b["tx"], b["ty"]] for _, b in per], axis=0)
        model = ShiftModel(np.zeros((2, 3)), med)
    ref = np.median(np.array([smp.x for smp in samples]), axis=0)
    cal = Calibration(name=name, video_hw=video_hw, video_box=box, npy_hw=npy_hw, scale=float(s_best), angle_deg=float(a_best),
                      shift_model=model.to_dict(), ref_features=ref.tolist(), created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    # 5) self-check on held-out frames: outline distance (px) and area overlap (Dice)
    Ms = [cal.matrix_from_features(smp.x) for smp in test]
    held_o = [outline_score(smp.pts, smp.dmap, M) for smp, M in zip(test, Ms)]
    held_d = [dice(R.warp_mask(smp.rgb_m, M, tuple(npy_hw)), smp.th_m) for smp, M in zip(test, Ms)]
    med_o = float(np.median(held_o))
    grade = "good" if med_o <= GOOD_PX else "acceptable" if med_o <= OK_PX else "poor"
    warnings = []
    if grade == "poor":
        warnings.append("hand outlines do not line up: check that RGB and radiometric files belong together and show the same moment")
    if len(good) < 10:
        warnings.append(f"only {len(good)} frames with a close outline fit: constant shift used (no per-frame parallax model)")
    cal.quality = dict(grade=grade, outline_px_heldout_median=med_o, dice_heldout_median=float(np.median(held_d)),
                       dice_heldout_p10=float(np.quantile(held_d, 0.1)), frames_fit=len(fit), frames_test=len(test),
                       frames_shift_model=len(good), recordings=len(recordings),
                       scale_search={s: round(v, 2) for s, v in s_scores.items()},
                       angle_search={f"{a:+.0f}": round(v, 2) for a, v in a_scores.items()},
                       seconds=round(time.time() - t_start, 1), warnings=warnings)
    return cal


def _matrix_from_features(self, x):
    t = ShiftModel.from_dict(self.shift_model).predict(np.asarray(x, float))[0]
    return R.matrix(self.k, self.scale, float(t[0]), float(t[1]), self.angle_deg)


Calibration.matrix_from_features = _matrix_from_features


# --------------------------------------------------------------------------- verify / apply
def verify(cal: Calibration, rgb_path, npy_path, n: int = 8, out_png: str | Path | None = None) -> dict:
    """Dice of the calibrated mapping on n frames of a recording, plus an overlay picture."""
    samples = collect_samples([(Path(rgb_path), Path(npy_path))], cal.video_box, n)
    if not samples:
        raise CalibrationError("no frame with a detected hand")
    d = [dice(R.warp_mask(r, cal.matrix_from_features(x), tuple(cal.npy_hw)), t) for x, r, t in samples]
    o = [outline_score(outline_points(r), outline_distance_map(t), cal.matrix_from_features(x)) for x, r, t in samples]
    mo = float(np.median(o))
    res = dict(frames=len(d), outline_px_median=mo, dice_median=float(np.median(d)), dice_min=float(np.min(d)),
               grade="good" if mo <= GOOD_PX else "acceptable" if mo <= OK_PX else "poor")
    if out_png:
        x, r, t = samples[len(samples) // 2]
        _overlay(cal, r, t, cal.matrix_from_features(x), out_png, f"{Path(rgb_path).stem}: Dice {d[len(samples) // 2]:.3f}")
    return res


def _overlay(cal, rgb_m, th_m, M, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = R.warp_mask(rgb_m, M, tuple(cal.npy_hw))
    img = np.zeros((*cal.npy_hw, 3))
    img[th_m] = [0.95, 0.45, 0.2]
    img[w & th_m] = [1, 1, 1]
    img[w & ~th_m] = [0.16, 0.47, 0.84]
    fig, ax = plt.subplots(figsize=(4.5, 5.5))
    ax.imshow(img)
    ax.set_title(f"{title}\nwhite = overlap · orange = thermal only · blue = RGB only", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def apply_to_frame(cal: Calibration, rgb_path, npy_path, frame: int, out_png=None) -> dict:
    """Map the 14 joints of one frame into the thermal image, read their temperature, draw an overlay."""
    bgr = _video_frames(rgb_path, [frame])[frame]
    rgb = np.ascontiguousarray(cal.crop(bgr)[:, :, ::-1])
    T = np.asarray(np.load(npy_path, mmap_mode="r")[frame], np.float32)
    lm = L.make_landmarker(LANDMARK_MODEL, video=False)
    pts, _ = L.detect(lm, rgb)
    lm.close()
    if pts is None:
        raise CalibrationError("no hand detected in this frame")
    jt = cal.rgb_to_thermal(pts[[j[1] for j in L.JOINTS]], landmarks=pts)
    temps = cal.temperature_at(T, pts[[j[1] for j in L.JOINTS]], landmarks=pts)
    if out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 6.2))
        ax[0].imshow(rgb)
        ax[0].scatter(*pts[[j[1] for j in L.JOINTS]].T, s=30, c="cyan", edgecolors="black")
        ax[0].set_title("RGB: joints found by MediaPipe")
        ax[1].imshow(T, cmap="inferno")
        ax[1].imshow(cal.warp_rgb_to_thermal(rgb, pts).mean(axis=2), cmap="gray", alpha=0.25)
        ax[1].scatter(*jt.T, s=30, c="cyan", edgecolors="black")
        for (x, y), v in zip(jt, temps):
            ax[1].text(x + 6, y, f"{v:.1f}", color="white", fontsize=7)
        ax[1].set_title("Thermal: the same joints after calibration")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
    return {name: dict(x=float(x), y=float(y), temperature_C=float(v)) for (name, *_), (x, y), v in zip(L.JOINTS, jt, temps)}


# --------------------------------------------------------------------------- proof: recover known distortions
def _distort_pair(rgb_path, npy_path, out_dir, frames, G=None, pad_x: int = 0):
    """Write a short RGB video and .npy where the thermal images are warped by G (2x3, radiometric pixels) and/or
    the RGB picture is moved sideways inside the video frame by pad_x pixels (a different letterbox)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fr = _video_frames(rgb_path, frames)
    seq = np.load(npy_path, mmap_mode="r")
    h, w = next(iter(fr.values())).shape[:2]
    vw = cv2.VideoWriter(str(out_dir / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (w, h))
    Ts = []
    for i in sorted(fr):
        f = fr[i]
        if pad_x:
            f = np.roll(f, pad_x, axis=1)
        vw.write(f)
        T = np.asarray(seq[i], np.float32)
        if G is not None:
            bg = float(np.percentile(T, 5))
            T = cv2.warpAffine(T, G, (T.shape[1], T.shape[0]), flags=cv2.INTER_LINEAR, borderValue=bg)
        Ts.append(T)
    vw.release()
    np.save(out_dir / "temp.npy", np.stack(Ts))
    return out_dir / "rgb.mp4", out_dir / "temp.npy"


def _G(scale, angle, tx, ty, hw=(640, 480)):
    c = (hw[1] / 2, hw[0] / 2)
    G = cv2.getRotationMatrix2D(c, -angle, scale)                # positive angle = counter-clockwise in image coords
    G[:, 2] += (tx, ty)
    return G


def _compose(G, M):
    return (np.vstack([G, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]]))[:2]


def selftest(cfg, subject_recs=("S7_left_static", "S12_right_static", "S3_left_open_close"), work=Path("data/calibration/selftest")):
    """For real recordings: calibrate the original pair (M0), distort the thermal side by a KNOWN transform G (and
    move the RGB picture inside the video frame), calibrate again (M1) and check M1 == G . M0 (with the shift of the box)."""
    from src.io_utils import load_recordings
    infos = {i.rec_id: i for i in load_recordings(cfg, only=list(subject_recs))}
    cases = [("scale +8 %, rotate 4°, shift (+18, -12) px", _G(1.08, 4.0, 18, -12), 0),
             ("scale -10 %, rotate -6°, shift (-25, +20) px", _G(0.90, -6.0, -25, 20), 0),
             ("different letterbox: picture moved 40 px right", None, 40),
             ("all together: scale +5 %, rotate 3°, shift (10, 15), letterbox +40 px", _G(1.05, 3.0, 10, 15), 40)]
    rows = []
    for rec in subject_recs:
        info = infos[rec]
        frames = np.linspace(0, info.n_frames - 1, 24).round().astype(int)
        base_rgb, base_npy = _distort_pair(info.rgb_path, info.npy_path, work / rec / "original", frames)
        cal0 = find_calibration([(base_rgb, base_npy)], f"{rec}_original", n_frames=24)
        grid = np.stack(np.meshgrid(np.linspace(60, 300, 7), np.linspace(80, 400, 9)), -1).reshape(-1, 2)   # crop pixels on the hand area
        for label, G, pad in cases:
            r, n = _distort_pair(info.rgb_path, info.npy_path, work / rec / label[:12].replace(" ", "_"), frames, G, pad)
            cal1 = find_calibration([(r, n)], f"{rec}_distorted", n_frames=24)
            M0 = cal0.matrix()
            truth = _compose(G, M0) if G is not None else M0
            # cal1 works in its own (possibly moved) crop: the same physical point is at crop + (pad - box shift)
            off = np.array([cal0.video_box[2] + pad - cal1.video_box[2], cal0.video_box[0] - cal1.video_box[0]])
            p_true = R.map_points(grid, truth)
            p_found = R.map_points(grid + off, cal1.matrix())
            err = np.linalg.norm(p_true - p_found, axis=1)
            sc_true = cal0.scale * (G is not None and np.sqrt(np.linalg.det(G[:, :2])) or 1.0)
            rows.append(dict(recording=rec, case=label, box_found=cal1.video_box, scale_true=round(float(sc_true), 3),
                             scale_found=round(cal1.scale, 3), angle_true=round(cal0.angle_deg + (0 if G is None else float(np.rad2deg(np.arctan2(G[1, 0], G[0, 0])))), 2),
                             angle_found=round(cal1.angle_deg, 2), mapping_error_px_median=round(float(np.median(err)), 2),
                             mapping_error_px_max=round(float(err.max()), 2), dice_heldout=round(cal1.quality["dice_heldout_median"], 3)))
            print(rows[-1], flush=True)
    import pandas as pd
    D = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    D.to_csv(REPORT_DIR / "selftest_known_distortions.csv", index=False)
    return D


def benchmark(cfg, train=("S1", "S2", "S3", "S4", "S5")):
    """Calibrate on a few subjects with find_calibration, apply unchanged to all other subjects,
    score against the hand-clicked joints and held-out Dice (frames from src.calibration_model collect)."""
    import pandas as pd
    from src import calibration_model as CM
    from src.io_utils import load_recordings
    infos = [i for i in load_recordings(cfg) if i.rec.subject in train]
    cal = find_calibration([(i.rgb_path, i.npy_path) for i in infos], "flirone_found_" + "-".join(train), n_frames=120)
    cal.save(REPORT_DIR / f"{cal.name}.json")
    F = pd.read_csv(CM.CAL_DIR / "frames.csv")
    F = F[~F.subject.isin(train)].reset_index(drop=True)
    reg = pd.read_csv("data/processed_v8/registration.csv").set_index("rec_id")
    # stored landmarks/masks are in the config crop (video_box from config.yaml); the found calibration works in
    # its own crop: convert with the offset between the two boxes, so both methods see the same physical points
    vb = cfg["pipeline"]["video_box"]
    k_cfg = (np.load(infos[0].npy_path, mmap_mode="r").shape[2]) / (vb[3] - vb[2])
    off = np.array([vb[2] - cal.video_box[2], vb[0] - cal.video_box[0]], float)
    conv = np.eye(3)
    conv[:2, 2] = off                                                     # config-crop -> cal-crop

    def found_matrix(i):
        pts = np.array([[F.at[i, f"lm{j}_x"], F.at[i, f"lm{j}_y"]] for j in range(21)]) + off
        return (np.vstack([cal.matrix(pts), [0, 0, 1]]) @ conv)[:2]       # expressed on config-crop pixels
    mats = {"v3: fitted per recording": [R.matrix(k_cfg, 1.25, reg.tx[r], reg.ty[r]) for r in F.rec_id],
            "found calibration (this module)": [found_matrix(i) for i in range(len(F))]}
    ann = pd.read_csv("data/annotations/joints_clicked.csv")
    from src.landmark_correction import ANN_LM, kind_of, loso_corrected
    Fi = {(r, int(f)): i for i, (r, f) in enumerate(zip(F.rec_id, F.frame))}
    out = {}
    for name, Ms in mats.items():
        rows = []
        for a in ann.itertuples():
            i = Fi.get((a.rec_id, int(a.frame)))
            if i is None:
                continue
            pts = np.array([[F.at[i, f"lm{j}_x"], F.at[i, f"lm{j}_y"]] for j in range(21)])
            P21 = R.map_points(pts, Ms[i])
            lmk = ANN_LM[a.joint]
            P, Q = P21[lmk], P21[lmk + 1]
            u = (Q - P) / np.linalg.norm(Q - P)
            aa, bb = L.RADIUS_PAIR[a.joint.split("_")[0]]
            e = np.array([a.x, a.y]) - P
            rows.append(dict(subject=a.rec_id.split("_")[0], kind=kind_of(a.joint), task="static" if a.rec_id.endswith("static") else "open_close",
                             x=a.x, y=a.y, px=P[0], py=P[1], ux=u[0], uy=u[1], seg=float(np.linalg.norm(P21[bb] - P21[aa])),
                             along=float(e @ u), err_raw=float(np.linalg.norm(e))))
        E = pd.DataFrame(rows)
        E["frac"] = E.along / E.seg
        E = loso_corrected(E)
        ann_keys = set(zip(ann.rec_id, ann.frame.astype(int)))
        dl = []
        for rec, g in F.groupby("rec_id"):
            masks = CM._masks(rec)
            for i in g.index:
                if (rec, int(F.frame[i])) in ann_keys:
                    continue
                rm, th = masks[int(F.frame[i])]
                dl.append(dice(R.warp_mask(rm, Ms[i]), th))
        out[name] = dict(joint_err_median=round(float(E.err_loso.median()), 2), static=round(float(E[E.task == "static"].err_loso.median()), 2),
                         open_close=round(float(E[E.task == "open_close"].err_loso.median()), 2), dice_median=round(float(np.median(dl)), 3),
                         n_joints=len(E), n_dice_frames=len(dl))
    res = dict(calibration=dict(box=cal.video_box, scale=cal.scale, angle_deg=cal.angle_deg, quality=cal.quality),
               train_subjects=list(train), test_subjects=sorted(F.subject.unique()), results=out)
    json.dump(res, open(REPORT_DIR / "benchmark_find_calibration.json", "w"), indent=1)
    return res


def production_profile(cfg, name: str = "flirone_one_pro_v9_found") -> Path:
    """Calibration for the original dataset from ALL subjects + the along-finger joint correction re-learned
    for this mapping from the hand-clicked joints (the method itself is validated by `benchmark`)."""
    import pandas as pd
    from src import calibration_model as CM
    from src.io_utils import load_recordings
    from src.landmark_correction import ANN_LM, kind_of
    infos = load_recordings(cfg)
    cal = find_calibration([(i.rgb_path, i.npy_path) for i in infos], name, n_frames=170)
    vb = cfg["pipeline"]["video_box"]
    off = np.array([vb[2] - cal.video_box[2], vb[0] - cal.video_box[0]], float)
    F = pd.read_csv(CM.CAL_DIR / "frames.csv")
    Fi = {(r, int(f)): i for i, (r, f) in enumerate(zip(F.rec_id, F.frame))}
    fr = {"MCP": [], "PIP": []}
    for a in pd.read_csv("data/annotations/joints_clicked.csv").itertuples():
        i = Fi.get((a.rec_id, int(a.frame)))
        if i is None:
            continue
        pts = np.array([[F.at[i, f"lm{j}_x"], F.at[i, f"lm{j}_y"]] for j in range(21)]) + off
        P21 = R.map_points(pts, cal.matrix(pts))
        lmk = ANN_LM[a.joint]
        u = (P21[lmk + 1] - P21[lmk]) / np.linalg.norm(P21[lmk + 1] - P21[lmk])
        aa, bb = L.RADIUS_PAIR[a.joint.split("_")[0]]
        fr[kind_of(a.joint)].append(float((np.array([a.x, a.y]) - P21[lmk]) @ u) / float(np.linalg.norm(P21[bb] - P21[aa])))
    prof = cal.to_profile()
    prof.update(camera="FLIR ONE Pro (smartphone add-on)", along_frac={k: float(np.median(v)) for k, v in fr.items()},
                validation=dict(selftest="reports/v9_calibration/selftest_known_distortions.csv",
                                benchmark="reports/v9_calibration/benchmark_find_calibration.json"))
    p = CALIB_DIR / f"{name}.json"
    json.dump(prof, open(p, "w"), indent=1)
    return p


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    from src.io_utils import load_config, load_recordings
    ap = argparse.ArgumentParser(description="Find, apply and verify the RGB->thermal calibration")
    ap.add_argument("cmd", choices=["find", "verify", "apply", "selftest", "benchmark", "production"])
    ap.add_argument("--rgb", action="append", default=[])
    ap.add_argument("--npy", action="append", default=[])
    ap.add_argument("--subjects", nargs="*", help="original dataset subjects to calibrate from (e.g. S1 S2)")
    ap.add_argument("--dataset", help="uploaded dataset id")
    ap.add_argument("--name", default=None)
    ap.add_argument("--profile")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(Path(a.config))
    if a.cmd == "find":
        if a.subjects:
            recs = [(i.rgb_path, i.npy_path) for i in load_recordings(cfg) if i.rec.subject in a.subjects]
        elif a.dataset:
            from src.dataset_manager import infos_from_manifest, load_manifest
            recs = [(i.rgb_path, i.npy_path) for i in infos_from_manifest(load_manifest(a.dataset))]
        else:
            recs = list(zip(a.rgb, a.npy))
        name = a.name or f"found_{time.strftime('%Y%m%d_%H%M%S')}"
        cal = find_calibration(recs, name, progress=lambda n, t, r: print(f"  frames from {n}/{t} recordings", flush=True))
        p = cal.save(a.out)
        print(json.dumps(dict(box=cal.video_box, scale=round(cal.scale, 3), angle_deg=cal.angle_deg, quality=cal.quality), indent=1))
        print("saved", p)
    elif a.cmd == "verify":
        cal = Calibration.load(a.profile)
        print(json.dumps(verify(cal, a.rgb[0], a.npy[0], out_png=a.out or REPORT_DIR / "verify_overlay.png"), indent=1))
    elif a.cmd == "apply":
        cal = Calibration.load(a.profile)
        res = apply_to_frame(cal, a.rgb[0], a.npy[0], a.frame, a.out or REPORT_DIR / "apply_frame.png")
        for k_, v in res.items():
            print(f"{k_:12s} thermal ({v['x']:.1f}, {v['y']:.1f})  {v['temperature_C']:.2f} °C")
    elif a.cmd == "selftest":
        print(selftest(cfg).to_string(index=False))
    elif a.cmd == "production":
        p = production_profile(cfg)
        print(json.dumps({k: v for k, v in json.load(open(p)).items() if k in ("scale", "angle_deg", "video_box", "quality", "along_frac")}, indent=1))
        print("saved", p)
    else:
        print(json.dumps(benchmark(cfg), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
