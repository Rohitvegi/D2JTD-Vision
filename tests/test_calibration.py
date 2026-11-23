"""src.calibration: transform algebra, outline-based fitting on synthetic hands with known calibration."""
import json

import cv2
import numpy as np
import pytest

from src import calibration as CAL
from src import registration as R
from src.calibration_model import ShiftModel

K = 4 / 3


def synthetic_hand(h=480, w=360):
    """Binary 'hand' in RGB crop pixels: palm + 4 fingers + thumb."""
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (180, 300), (75, 90), 0, 0, 360, 1, -1)
    for x, top in ((120, 90), (160, 60), (200, 70), (240, 110)):
        cv2.rectangle(m, (x - 14, top), (x + 14, 260), 1, -1)
    cv2.line(m, (115, 330), (50, 230), 1, 26)
    return m.astype(bool)


def thermal_of(rgb_m, s, tx, ty, a, forearm=False):
    th = R.warp_mask(rgb_m, R.matrix(K, s, tx, ty, a))
    if forearm:                                          # warm forearm below the wrist, absent from the RGB mask
        th = th.copy()
        th[560:, 150:340] = True
    return th


def test_matrix_decompose_roundtrip():
    for s, tx, ty, a in ((1.25, -12.3, -31.5, 0.0), (1.1, 20, -5, 4.0), (0.9, -3, 8, -7.5)):
        assert np.allclose(R.decompose(R.matrix(K, s, tx, ty, a), K), (s, tx, ty, a), atol=1e-4)


@pytest.mark.parametrize("s,tx,ty,a", [(1.25, -12.0, -31.0, 0.0), (1.10, 15.0, -10.0, 4.0)])
def test_outline_fit_recovers_shift(s, tx, ty, a):
    rm = synthetic_hand()
    smp = CAL._Sample(np.zeros(3), rm, thermal_of(rm, s, tx, ty, a))
    b = smp.fit(K, s, a)
    assert abs(b["tx"] - tx) <= 1.5 and abs(b["ty"] - ty) <= 1.5
    assert b["outline_px"] < 1.0


def test_scale_search_ignores_forearm():
    """Whole-area overlap is pulled towards a larger scale by an extra warm forearm; outline distance is not."""
    rm = synthetic_hand()
    s_true = 1.25
    smp = CAL._Sample(np.zeros(3), rm, thermal_of(rm, s_true, -12, -31, 0.0, forearm=True))
    scores = {s: smp.fit(K, s, 0.0)["outline_px"] for s in np.arange(1.05, 1.51, 0.05)}
    assert abs(min(scores, key=scores.get) - s_true) < 0.03


def test_calibration_apply_and_inverse(tmp_path):
    sm = ShiftModel(np.zeros((2, 3)), np.array([-12.0, -31.0]))
    cal = CAL.Calibration(name="t", video_hw=[480, 640], video_box=[0, 480, 140, 500], npy_hw=[640, 480], scale=1.25,
                          angle_deg=2.0, shift_model=sm.to_dict(), ref_features=[110, 180, 240])
    pts = np.array([[100.0, 200.0], [250.0, 50.0]])
    th = cal.rgb_to_thermal(pts)
    assert np.allclose(cal.thermal_to_rgb(th), pts, atol=1e-3)
    assert np.allclose(cal.rgb_to_thermal(pts + [140, 0], full_frame=True), th)
    assert cal.warp_rgb_to_thermal(np.zeros((480, 360, 3), np.uint8)).shape == (640, 480, 3)
    p = cal.save(tmp_path / "c.json")
    prof = json.load(open(p))
    assert prof["mode"] == "learned" and prof["fixed_scale"] == 1.25 and prof["angle_deg"] == 2.0
    cal2 = CAL.Calibration.load(p)
    assert np.allclose(cal2.matrix(), cal.matrix())


def test_v3_profile_loads_as_constant_calibration():
    cal = CAL.Calibration.from_profile(dict(name="v3", video_hw=[480, 640], video_box=[0, 480, 140, 500], npy_hw=[640, 480],
                                            rig_default=[1.25, -12.3, -31.5]))
    assert np.allclose(cal.matrix(), R.matrix(K, 1.25, -12.3, -31.5))


def test_shape_mismatch_is_reported(tmp_path):
    vw = cv2.VideoWriter(str(tmp_path / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8, (640, 480))
    for _ in range(3):
        f = np.zeros((480, 640, 3), np.uint8)
        f[:, 140:500] = 120
        vw.write(f)
    vw.release()
    np.save(tmp_path / "t.npy", np.full((3, 480, 640), 30.0, np.float32))     # landscape: transposed export
    with pytest.raises(CAL.CalibrationError, match="rotated"):
        CAL.find_calibration([(tmp_path / "v.mp4", tmp_path / "t.npy")])
