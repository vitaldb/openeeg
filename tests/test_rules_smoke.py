"""Smoke tests for predict_bis(method=...) — Morimoto / Cusenza / Sleigh."""
from __future__ import annotations

import numpy as np
import pytest

from openeeg import predict_bis, sleigh_gate, higuchi_fd, se50d


def _synth_eeg(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(seconds * 128))) / 128.0
    return (
        8.0 * np.sin(2 * np.pi * 10.0 * t)
        + 4.0 * np.sin(2 * np.pi * 1.5 * t + 0.3)
        + 1.5 * rng.standard_normal(t.size)
    )


# --- Base features ---------------------------------------------------------

def test_higuchi_fd_in_expected_range():
    eeg = _synth_eeg(120.0)
    fd = higuchi_fd(eeg)
    valid = fd[np.isfinite(fd)]
    assert valid.size > 0
    assert valid.min() >= 0.8
    assert valid.max() <= 2.2


def test_se50d_in_band():
    eeg = _synth_eeg(120.0)
    se = se50d(eeg)
    valid = se[np.isfinite(se)]
    assert valid.size > 0
    assert valid.min() >= 0.0
    assert valid.max() <= 47.0


# --- predict_bis(method=...) -----------------------------------------------

@pytest.mark.parametrize("method", ["gbm", "lee", "connor", "morimoto", "cusenza", "sleigh"])
def test_predict_bis_method_shape_and_range(method):
    eeg = _synth_eeg(120.0)
    pred = predict_bis(eeg, method=method)
    assert pred.ndim == 1 and pred.size > 0
    finite = pred[np.isfinite(pred)]
    assert finite.size > 0
    assert finite.min() >= 0.0
    assert finite.max() <= 100.0


def test_predict_bis_method_default_is_gbm():
    eeg = _synth_eeg(120.0)
    default = predict_bis(eeg)
    gbm = predict_bis(eeg, method="gbm")
    np.testing.assert_allclose(default, gbm)


def test_predict_bis_unknown_method_raises():
    eeg = _synth_eeg(60.0)
    with pytest.raises(ValueError, match="Unknown method"):
        predict_bis(eeg, method="not-a-real-method")


def test_predict_bis_method_rejects_bad_fs():
    eeg = _synth_eeg(60.0)
    with pytest.raises(ValueError):
        predict_bis(eeg, method="morimoto", fs=256)


# --- Sleigh gate -----------------------------------------------------------

def test_sleigh_gate_returns_ternary():
    eeg = _synth_eeg(120.0)
    g = sleigh_gate(eeg)
    finite = g[np.isfinite(g)]
    assert finite.size > 0
    assert set(np.unique(finite)).issubset({-1.0, 0.0, 1.0})
