"""Tests for v6 features and v8 dataset / evaluation / live / GUI modules.
Run:  .venv312\\Scripts\\python.exe -m pytest -q        (end-to-end test needs tests/make_fixture.py first)"""
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import dataset_manager as DM
from src.evaluation import categorise, classification_metrics, load_labels, match_labels, measurement_metrics
from src.feature_extraction import recording_features, warm_frames
from src.live_acquisition import Frame, PairingBuffer

FIX = Path("data/test_fixtures/mini_dataset.zip")


# ---------------------------------------------------------------- dataset import
def test_safe_extract_refuses_path_traversal(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("../../outside.txt", "x")
    with pytest.raises(ValueError):
        DM.safe_extract(z, tmp_path / "out")
    assert not (tmp_path / "outside.txt").exists()


def test_safe_extract_normal_archive(tmp_path):
    z = tmp_path / "ok.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("d/a.txt", "hello")
    out = DM.safe_extract(z, tmp_path / "out")
    assert out[0].read_text() == "hello"


def test_generic_and_flirone_names_are_matched(tmp_path):
    for n in ("P01_L_static_rgb.mp4", "P01_L_static_thermal.mp4", "P01_L_static_radiometric.npy", "P01_R_oc_rgb.mp4",
              "S3/left hand/open_close_RGB.mp4", "S3/left hand/open_close_Thermal_Radiometric_Sequence.npy", "holiday.mp4"):
        p = tmp_path / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"0")
    m = DM.build_manifest(tmp_path, "t")
    recs = {r["rec_id"]: r for r in m["recordings"]}
    assert set(recs["P01_left_static"]["files"]) == {"rgb", "thermal_video", "radiometric"}
    assert recs["P01_left_static"]["match_status"] == "complete"
    assert recs["P01_right_open_close"]["match_status"] == "requires manual matching"     # no radiometric file
    assert "radiometric" in recs["S3_left_open_close"]["files"]                          # FLIROne layout
    assert any(u.endswith("holiday.mp4") for u in m["unmatched_files"])                  # never paired silently


def test_manual_matching_updates_status(tmp_path):
    (tmp_path / "P02_L_static_rgb.mp4").write_bytes(b"0")
    m = DM.build_manifest(tmp_path, "t")
    assert m["recordings"][0]["match_status"] == "requires manual matching"
    m = DM.set_file(m, "P02_left_static", "radiometric", str(tmp_path / "x.npy"))
    assert m["recordings"][0]["match_status"].startswith("complete")


# ---------------------------------------------------------------- labels + metrics
def test_label_precedence_and_unmatched(tmp_path):
    p = tmp_path / "l.csv"
    pd.DataFrame([dict(subject="P1", label="healthy", source="exam"), dict(subject="P1", hand="L", label="RA", source="US"),
                  dict(rec_id="P1_left_static", label="control"), dict(subject="P9", label="healthy")]).to_csv(p, index=False)
    recs = [dict(rec_id="P1_left_static", subject="P1", hand="left", task="static"),
            dict(rec_id="P1_left_open_close", subject="P1", hand="left", task="open_close"),
            dict(rec_id="P1_right_static", subject="P1", hand="right", task="static")]
    M, rep = match_labels(load_labels(p), recs)
    lab = dict(zip(M.rec_id, M.label))
    assert lab == {"P1_left_static": "healthy", "P1_left_open_close": "arthritis", "P1_right_static": "healthy"}
    assert len(rep["unmatched_label_rows"]) == 1


def test_classification_metrics_known_values():
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    p = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
    m = classification_metrics(y, p)
    assert m["confusion_matrix"] == dict(tn=4, fp=2, fn=1, tp=3)
    assert m["sensitivity"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(0.6)
    assert m["accuracy"] == pytest.approx(0.7)
    assert "small_sample_warning" in m


def test_single_class_does_not_produce_misleading_metrics():
    m = classification_metrics([0, 0, 0], [0, 1, 0], scores=np.array([0.1, 0.9, 0.2]))
    assert m["specificity"] == pytest.approx(2 / 3)
    assert "accuracy" not in m and "roc_auc" not in m and "note" in m


def test_measurement_metrics():
    m = measurement_metrics([1, 2, 4], [1, 1, 1], unit="px")
    assert m["mae"] == pytest.approx(4 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(10 / 3))
    assert m["mean_signed_error"] == pytest.approx(4 / 3)


def test_categories():
    good = pd.Series(dict(q_frames_used=50, q_roi_usable=1.0, q_contaminated_rois=0, q_registration_dice=0.9))
    assert categorise(good, True, False, True) == "Potential anomaly"
    assert categorise(good, False, True, True) == "Outside model experience"
    assert categorise(good, True, False, False) == "Not applicable"
    assert categorise(good.copy().replace({50: 3}), True, False, True) == "Insufficient data"
    assert categorise(pd.Series(dict(q_frames_used=50, q_roi_usable=0.3)), True, False, True) == "Invalid measurement"


# ---------------------------------------------------------------- features
def _frames(n=40):
    from src.landmarks import JOINT_NAMES
    rng = np.random.default_rng(0)
    d = dict(t_s=np.arange(n) / 8.0, frame_valid=True, openness=2.0, joint_centres_on_hand=1.0, dorsum_T=33.0, hand_T=32.8)
    for j in JOINT_NAMES:
        d[f"{j}_T"] = 32.5 + rng.normal(0, 0.05, n)
        d[f"{j}_sd"] = 0.2
    return pd.DataFrame(d)


def test_feature_warming_moves_the_right_features():
    g = _frames()
    a = recording_features(g, "static")
    b = recording_features(warm_frames(g, ["index_MCP"], 2.0), "static")
    assert b["index_MCP_D2JTD"] == pytest.approx(a["index_MCP_D2JTD"] - 2.0, abs=1e-6)
    assert b["middle_MCP_D2JTD"] == pytest.approx(a["middle_MCP_D2JTD"])
    assert b["index_MCP_minus_middle_MCP"] == pytest.approx(a["index_MCP_minus_middle_MCP"] + 2.0, abs=1e-6)
    c = recording_features(warm_frames(g, None, 2.0), "static")
    assert c["index_MCP_D2JTD"] == pytest.approx(a["index_MCP_D2JTD"])       # diffuse: relative unchanged
    assert c["index_MCP_Tmed"] == pytest.approx(a["index_MCP_Tmed"] + 2.0)


# ---------------------------------------------------------------- live pairing
def test_pairing_uses_newest_fresh_thermal_frame():
    b = PairingBuffer(max_age_ms=150)
    for t in (0, 115, 230):
        b.push_thermal(Frame(t, np.zeros(1), index=t))
    f, age = b.pair(240)
    assert f.t_ms == 230 and age == pytest.approx(10)
    f, age = b.pair(500)                                    # newest thermal frame is 270 ms old -> not paired
    assert f is None and age == pytest.approx(270)


# ---------------------------------------------------------------- GUI
def test_gui_builds_all_pages():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from ui.app import MainWindow, QApplication
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    assert w.stack.count() == 10
    for i in range(w.stack.count()):
        w.switch(i)
    w.close()


# ---------------------------------------------------------------- end to end
@pytest.mark.skipif(not FIX.is_file(), reason="run tests/make_fixture.py first")
def test_end_to_end_new_dataset(tmp_path):
    from src.evaluation import run_independent_test
    from src.io_utils import load_config
    from src.report_generator import test_report
    cfg = load_config(Path("config.yaml"))
    m = DM.import_zip(FIX, "pytest_e2e")
    try:
        assert len(m["recordings"]) == 3 and len(m["unmatched_files"]) == 1
        cal = DM.load_calibration("flirone_one_pro_v3")
        V = DM.validate_manifest(m, cal, cfg)
        assert all(v["status"] in ("Valid", "Valid with warnings") for v in V)
        reg, F = DM.process_dataset(m, cfg, cal, [v["rec_id"] for v in V], workers=2)
        assert len(F) == 3 and F.q_frames_used.min() >= 10
        model = sorted(Path("models/registry").glob("v7_*"))[0]
        out = run_independent_test(m["dataset_id"], model, "data/test_fixtures/labels_DUMMY_software_test.csv", "pytest_e2e")
        res = json.load(open(out / "results.json"))
        assert res["n_recordings"] == 3 and "metrics_flag_vs_label" in res
        assert json.load(open(out / "lock.json"))["model_sha256"]
        paths = test_report(out)
        assert Path(paths["pdf"]).stat().st_size > 1000 and Path(paths["excel"]).is_file()
    finally:
        import shutil
        shutil.rmtree(DM.DATASETS / m["dataset_id"], ignore_errors=True)
        for p in Path("results/independent_tests").glob("pytest_e2e__*"):
            shutil.rmtree(p, ignore_errors=True)


@pytest.mark.skipif(not Path("data/processed_v6/frames/S2_right_static.csv").is_file(), reason="needs v6 pipeline output")
def test_joint_view_renders_temperatures(tmp_path):
    from src.io_utils import load_config
    from src.joint_view import render
    png, J = render("S2_right_static", "data/processed_v6", load_config(Path("config.yaml")), out_png=tmp_path / "jv.png")
    assert png.stat().st_size > 10000 and len(J) == 14
    assert J.temperature_C.between(15, 45).all()                      # plausible skin temperatures in deg C
    assert J.x_rgb.between(0, 360).all() and J.y_rgb.between(0, 480).all()   # joints inside the RGB photo


def _hand(folded: bool):
    """Synthetic dorsal view, fingers pointing up (image y grows downwards)."""
    p = np.zeros((21, 2))
    p[0] = (175, 420)                                        # wrist
    p[1:5] = [(260, 360), (290, 310), (310, 270), (325, 235)]  # thumb CMC, MCP, IP, tip
    for k, x in enumerate((240, 195, 150, 110)):             # index, middle, ring, little
        b = 5 + 4 * k
        p[b] = (x, 220)                                      # MCP (knuckle)
        xf = 0.4 * x + 0.6 * 200                              # folded fingers tuck in towards the palm centre
        p[b + 1:b + 4] = [(xf, 290), (xf, 330), (xf, 350)] if folded else [(x, 150), (x, 100), (x, 60)]
    return p


def test_open_hand_all_joints_visible():
    from src.landmarks import joint_visibility
    assert not any(joint_visibility(_hand(False)).values())


def test_folded_fingers_hidden_knuckles_visible():
    from src.landmarks import joint_visibility
    v = joint_visibility(_hand(True))
    for f in ("index", "middle", "ring", "little"):
        assert v[f"{f}_PIP"] and v[f"{f}_DIP"], f            # folded back over the palm -> hidden
        assert v[f"{f}_MCP"] == ""                           # knuckles stay visible
