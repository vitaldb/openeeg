"""Smoke tests for predict_bis(method="lee") — data-driven piecewise model."""
from __future__ import annotations

import numpy as np
import pytest

from openeeg import predict_bis


def _synth_eeg(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(seconds * 128))) / 128.0
    sig = (
        8.0 * np.sin(2 * np.pi * 10.0 * t)
        + 4.0 * np.sin(2 * np.pi * 1.5 * t + 0.3)
        + 1.5 * rng.standard_normal(t.size)
    )
    return sig


def test_predict_bis_lee_returns_in_range():
    eeg = _synth_eeg(300.0)
    pred = predict_bis(eeg, method="lee")
    assert pred.ndim == 1 and pred.size > 0
    finite = pred[np.isfinite(pred)]
    assert finite.size > 0
    assert finite.min() >= 0.0
    assert finite.max() <= 100.0


def test_predict_bis_lee_modest_input_handled():
    eeg = _synth_eeg(60.0)
    pred = predict_bis(eeg, method="lee")
    assert isinstance(pred, np.ndarray)
    assert pred.size > 0


def test_predict_bis_lee_rejects_wrong_fs():
    eeg = _synth_eeg(60.0)
    with pytest.raises(ValueError):
        predict_bis(eeg, method="lee", fs=256)


def test_predict_bis_lee_smoothing_reduces_step_noise():
    eeg = _synth_eeg(300.0, seed=1)
    raw = predict_bis(eeg, method="lee", smooth_W=0)
    sm = predict_bis(eeg, method="lee", smooth_W=15.0)
    raw_steady = raw[60:][np.isfinite(raw[60:])]
    sm_steady = sm[60:][np.isfinite(sm[60:])]
    if len(raw_steady) > 20 and len(sm_steady) > 20:
        raw_dabs = np.median(np.abs(np.diff(raw_steady)))
        sm_dabs = np.median(np.abs(np.diff(sm_steady)))
        assert sm_dabs <= raw_dabs + 1e-6


def test_predict_bis_lee_deep_rule_activates_on_isoelectric():
    rng = np.random.default_rng(7)
    t = np.arange(int(60 * 128)) / 128.0
    eeg = 0.3 * rng.standard_normal(t.size)
    pred = predict_bis(eeg, method="lee", smooth_W=0)
    finite = pred[np.isfinite(pred)]
    if finite.size > 30:
        assert np.median(finite[20:]) < 30.0
