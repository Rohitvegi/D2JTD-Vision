"""v9 learned calibration: model fitting, per-frame matrices, profile handling."""
import copy
from pathlib import Path

import numpy as np
import pytest

from src import calibration_model as CM
from src import dataset_manager as DM
from src import registration as R
from src.io_utils import load_config

CFG = load_config(Path("config.yaml"))


def _pts(size=110.0, cx=180.0, cy=240.0):
    """21 fake landmarks: wrist at (cx, cy + size/2), middle MCP (9) at (cx, cy - size/2)."""
    p = np.tile([cx, cy], (21, 1)).astype(float)
    p[0] = [cx, cy + size / 2]
    p[9] = [cx, cy - size / 2]
    return p


def test_shift_model_recovers_linear_parallax():
    rng = np.random.default_rng(0)
    X = np.c_[rng.uniform(60, 140, 300), rng.uniform(120, 240, 300), rng.uniform(150, 330, 300)]
    Y = np.c_[-12 + 0.05 * X[:, 0], -40 + 0.08 * X[:, 0]] + rng.normal(0, 0.3, (300, 2))
    Y[:10] += 80                                             # outliers: masks that disagree
    m = CM.ShiftModel().fit(X, Y)
    assert abs(m.coef[1, 0] - 0.08) < 0.01 and abs(m.intercept[1] + 40) < 1.5
    m2 = CM.ShiftModel.from_dict(m.to_dict())
    assert np.allclose(m2.predict(X[:5]), m.predict(X[:5]))


def test_frame_features_use_hand_size():
    x = CM.frame_features(_pts(size=100))
    assert x[0] == pytest.approx(100)


def test_frame_matrix_learned_and_hybrid():
    sm = CM.ShiftModel(np.array([[0.1, 0, 0], [0.2, 0, 0]]), np.array([-20.0, -50.0]))
    cal = dict(scale=1.25, shift_model=sm.to_dict(), mode="learned")
    k = 4 / 3
    M = CM.frame_matrix(cal, k, _pts(size=100))
    assert np.allclose(M, R.matrix(k, 1.25, -10.0, -30.0))
    # hybrid: recording shift + change predicted between this frame's hand size and the recording median
    cal["mode"] = "hybrid"
    M2 = CM.frame_matrix(cal, k, _pts(size=110), rec_shift=(-12.0, -31.0), rec_size=100.0)
    assert np.allclose(M2, R.matrix(k, 1.25, -12.0 + 1.0, -31.0 + 2.0))


def test_trained_profile_is_valid_and_applies():
    cal = CM.load_model()
    if cal is None:
        pytest.skip("calibration model not trained (python -m src.calibration_model train)")
    for key in ("scale", "fixed_scale", "video_box", "video_hw", "npy_hw", "rig_default", "shift_model", "validation"):
        assert key in cal
    c = DM.apply_calibration(CFG, cal)
    assert c["pipeline"]["registration"]["mode"] == cal["mode"]
    assert Path(c["pipeline"]["registration"]["calibration_model"]).is_file()


def test_v3_profile_forces_per_recording_mode():
    cfg = copy.deepcopy(CFG)
    cfg["pipeline"]["registration"]["mode"] = "learned"
    c = DM.apply_calibration(cfg, DM.calibration_from_config(CFG))
    assert c["pipeline"]["registration"]["mode"] == "per_recording"


def test_pipeline_rejects_missing_model(tmp_path):
    from src.run_pipeline import load_calibration_model
    pc = copy.deepcopy(CFG["pipeline"])
    pc["registration"]["mode"] = "learned"
    pc["registration"]["calibration_model"] = str(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        load_calibration_model(pc)
    pc["registration"]["mode"] = "per_recording"
    assert load_calibration_model(pc) is None


def test_every_saved_model_loads_in_a_fresh_program():
    """Bundles pickled while a module ran as __main__ must still load from any program (app, CLI, scripts)."""
    import subprocess
    import sys
    code = ("import glob,sys; sys.path.insert(0,'.'); from src.anomaly_models import load_bundle; "
            "[load_bundle(d) for d in glob.glob('models/registry/*') if __import__('os').path.isfile(d + '/bundle.joblib')]; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-800:]


def test_final_joint_temperature_is_median_of_frames_with_spread():
    import pandas as pd
    from src.feature_extraction import recording_features
    from src.landmarks import JOINT_NAMES
    rng = np.random.default_rng(1)
    n = 40
    g = pd.DataFrame({"frame_valid": True, "t_s": np.arange(n) / 3, "openness": 2.5, "joint_centres_on_hand": 1.0,
                      "dorsum_T": 33.0, "hand_T": 33.0})
    for j in JOINT_NAMES:
        g[f"{j}_T"] = 32.0 + rng.normal(0, 0.2, n)
        g[f"{j}_sd"] = 0.1
    g.loc[:4, "index_MCP_T"] = 40.0                      # 5 bad frames: the median must not follow them
    g.loc[5:9, "middle_MCP_T"] = np.nan                  # 5 frames where the joint was hidden
    f = recording_features(g, "static")
    assert abs(f["index_MCP_Tmed"] - 32.0) < 0.2
    assert f["index_MCP_Tq1"] <= f["index_MCP_Tmed"] <= f["index_MCP_Tq3"]
    assert f["middle_MCP_nframes"] == n - 5 and f["thumb_MCP_nframes"] == n
