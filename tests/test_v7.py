"""Tests for v5 annotation QC and v7 anomaly models.  Run:  .venv312\\Scripts\\python.exe -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import anomaly_models as AM
from src.annotation_qc import consistency_flags, three_cornered_hat
from src.landmark_correction import apply as apply_correction, loso_corrected
from src.landmarks import JOINT_NAMES


def toy(n_subjects=12, hands=2, p=14, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.4, 0.5, size=(n_subjects * hands, p))
    groups = np.repeat([f"S{i}" for i in range(n_subjects)], hands)
    return X, groups


# ---------------------------------------------------------------- leakage
def test_no_model_scores_a_subject_it_was_trained_on(monkeypatch):
    X, groups = toy(n_subjects=6, p=4)
    X[:, 0] = np.repeat(np.arange(6) * 1000.0, 2)          # column 0 encodes the subject
    events = []
    orig_fit, orig_score = AM.Pipeline.fit, AM.Pipeline.score

    def fit(self, Xf):
        events.append(("fit", set(np.unique(Xf[:, 0]))))
        return orig_fit(self, Xf)

    def score(self, Xs):
        events.append(("score", set(np.unique(np.round(Xs[:, 0], -3)))))
        return orig_score(self, Xs)

    monkeypatch.setattr(AM.Pipeline, "fit", fit)
    monkeypatch.setattr(AM.Pipeline, "score", score)
    monkeypatch.setattr(AM, "FEATURES", [f"f{i}" for i in range(4)])
    monkeypatch.setattr(AM, "PATTERNS", {"single_joint": None})
    monkeypatch.setattr(AM, "DELTAS", (1.0,))
    AM.evaluate_model(X, groups, "robust_z_max")
    last_fit = None
    n_scores = 0
    for kind, subjects in events:
        if kind == "fit":
            last_fit = subjects
        else:
            n_scores += 1
            assert not (subjects & last_fit), f"scored subjects {subjects} that were in training set"
    assert n_scores > 0


def test_threshold_uses_only_given_subjects():
    X, groups = toy()
    thr_a = AM.nested_threshold(X[groups != "S0"], groups[groups != "S0"], "robust_z_max")
    X2 = X.copy()
    X2[groups == "S0"] += 100                              # change the held-out subject only
    thr_b = AM.nested_threshold(X2[groups != "S0"], groups[groups != "S0"], "robust_z_max")
    assert thr_a == thr_b


# ---------------------------------------------------------------- models
def test_robust_z_known_values():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    m = AM.RobustZ().fit(X)
    assert m.c_[0] == 2.0
    assert m.s_[0] == pytest.approx(1.4826 * 1.0)
    assert m.score(np.array([[2.0 + 1.4826 * 3]]))[0] == pytest.approx(3.0)


@pytest.mark.parametrize("name", list(AM.MODELS))
def test_every_model_fits_and_scores(name):
    X, groups = toy()
    p = AM.Pipeline(name).fit(X)
    s = p.score(X[:3])
    assert s.shape == (3,) and np.all(np.isfinite(s))


def test_nested_specificity_close_to_target_on_healthy_gaussian_data():
    X, groups = toy(n_subjects=40, p=4, seed=1)
    spec = []
    for s in np.unique(groups):
        tr = groups != s
        thr = AM.nested_threshold(X[tr], groups[tr], "mahalanobis_ledoit_wolf")
        p = AM.Pipeline("mahalanobis_ledoit_wolf").fit(X[tr])
        spec += list(p.score(X[~tr]) <= thr)
    assert 0.85 <= np.mean(spec) <= 1.0


def test_warm_only_changes_selected_joints_and_diffuse_changes_nothing():
    X, _ = toy(p=14)
    Z = AM.warm(X, [2, 5], 1.0)
    assert np.allclose(Z[:, [2, 5]], X[:, [2, 5]] - 1.0)
    assert np.allclose(np.delete(Z, [2, 5], axis=1), np.delete(X, [2, 5], axis=1))
    assert np.allclose(AM.warm(X, [], 1.0), X)


def test_out_of_range_flags_extremes_only():
    X, _ = toy()
    p = AM.Pipeline("robust_z_max").fit(X)
    assert not p.out_of_range(X).any()
    assert p.out_of_range(X[:1] + 50).all()


def test_imputer_uses_training_median():
    X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    imp = AM.Imputer().fit(X)
    assert np.allclose(imp.transform(np.array([[np.nan, np.nan]])), [[3.0, 5.0]])


# ---------------------------------------------------------------- annotation QC
def test_three_cornered_hat_recovers_independent_errors():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 400, size=(20000, 2))
    A = truth + rng.normal(0, 3, truth.shape)
    R = truth + rng.normal(0, 1, truth.shape)
    T = truth + rng.normal(0, 2, truth.shape)
    own = three_cornered_hat(A, R, T, debias=False)["own_rms_error_px"]
    assert own["annotation"] == pytest.approx(3 * np.sqrt(2), rel=0.05)
    assert own["rgb_mapped"] == pytest.approx(1 * np.sqrt(2), rel=0.1)
    assert own["thermal_mediapipe"] == pytest.approx(2 * np.sqrt(2), rel=0.05)


def _frame(mcp_y, pip_y):
    rows = []
    for fin, x in (("index", 100.0), ("middle", 150.0), ("ring", 200.0)):
        for kind, y in (("MCP", mcp_y), ("PIP", pip_y)):
            rows.append(dict(rec_id="S1_left_static", frame=1, joint=f"{fin}_{kind}", kind=kind, x=x, y=y,
                             px=x, py=300.0 if kind == "MCP" else 220.0, ux=0.0, uy=-1.0, seg=80.0))
    return pd.DataFrame(rows)


def test_consistency_flags_clean_frame_passes():
    assert consistency_flags(_frame(300.0, 220.0)).empty


def test_consistency_flags_detects_swapped_mcp_pip():
    f = consistency_flags(_frame(220.0, 300.0))            # MCP clicked above PIP
    assert (f.check == "order").sum() == 3


# ---------------------------------------------------------------- joint correction
def test_apply_correction_moves_joint_along_finger():
    pts = np.zeros((21, 2))
    for i in range(21):
        pts[i] = [100.0, 400.0 - 10 * i]                    # straight finger pointing up
    out = apply_correction(pts, {"along_frac": {"MCP": 0.1, "PIP": 0.0}})
    seg = np.linalg.norm(pts[6] - pts[5])
    assert out[5][1] == pytest.approx(pts[5][1] - 0.1 * seg)  # index MCP moved towards the tip
    assert np.allclose(out[6], pts[6])                         # PIP unchanged
    assert np.allclose(out[0], pts[0])                         # wrist never moved


def test_loso_correction_never_uses_own_subject():
    rows = []
    for subj, frac in (("S1", 0.1), ("S2", 0.3)):
        rows.append(dict(subject=subj, kind="MCP", frac=frac, seg=100.0, px=0.0, py=0.0, ux=0.0, uy=-1.0,
                         x=0.0, y=-frac * 100.0))
    E = loso_corrected(pd.DataFrame(rows))
    s1 = E[E.subject == "S1"].iloc[0]
    assert s1.cy == pytest.approx(-0.3 * 100.0)               # S1 corrected with S2's fraction
    assert s1.err_loso == pytest.approx(20.0)


def test_joint_list_is_the_14_joint_definition():
    assert len(JOINT_NAMES) == 14
