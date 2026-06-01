"""Smoke tests: every public function returns the right shape on a synthetic input."""
import numpy as np

from openeeg import (
    openibis, openbsr, emg_correct,
    sef, bcsef, beta_ratio, emg_estimate,
    band_power, spectral_entropy,
    predict_bis,
)


def make_eeg(n_seconds: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fs = 128
    n = n_seconds * fs
    # 1/f-ish noise + a 10 Hz oscillation
    t = np.arange(n) / fs
    return 20 * np.sin(2 * np.pi * 10 * t) + 10 * rng.standard_normal(n)


def expected_n_epochs(n_samples: int) -> int:
    fs = 128
    stride = int(fs * 0.5)
    return int(np.floor((n_samples - fs) / stride) - 10)


def test_openibis_paper_shape():
    eeg = make_eeg()
    out = openibis(eeg, deep="paper")
    assert out.shape == (expected_n_epochs(len(eeg)),)
    valid = out[~np.isnan(out)]
    assert valid.min() >= 0.0 - 1e-6
    assert valid.max() <= 100.0 + 1e-6


def test_openibis_ellerkmann_shape():
    eeg = make_eeg()
    out = openibis(eeg, deep="ellerkmann")
    assert out.shape == (expected_n_epochs(len(eeg)),)


def test_openibis_quazi_bsr_shape():
    eeg = make_eeg()
    out = openibis(eeg, bsr="quazi", deep="paper")
    assert out.shape == (expected_n_epochs(len(eeg)),)


def test_openibis_rejects_bad_bsr():
    import pytest
    with pytest.raises(ValueError):
        openibis(make_eeg(), bsr="banana")


def test_openibis_rejects_bad_deep():
    import pytest
    with pytest.raises(ValueError):
        openibis(make_eeg(), deep="banana")


def test_openibis_rejects_bad_fs():
    import pytest
    with pytest.raises(ValueError):
        openibis(make_eeg(), fs=256)


def test_openbsr_shape():
    eeg = make_eeg()
    out = openbsr(eeg)
    assert out.shape == (expected_n_epochs(len(eeg)),)
    valid = out[~np.isnan(out)]
    assert valid.min() >= 0.0 - 1e-6
    assert valid.max() <= 100.0 + 1e-6


def test_emg_correct_no_effect_below_threshold():
    bis = np.array([95.0, 50.0, 20.0])
    emg = np.array([25.0, 30.0, 33.0])  # all below 34 dB
    out = emg_correct(bis, emg)
    np.testing.assert_allclose(out, bis)


def test_emg_correct_subtracts_above_threshold():
    bis = np.array([95.0, 50.0])
    emg = np.array([44.0, 34.0])  # 10 dB above / at threshold
    out = emg_correct(bis, emg)
    np.testing.assert_allclose(out, [95.0 - 0.54 * 10.0, 50.0])


def test_emg_correct_clips_to_valid_range():
    bis = np.array([5.0, 100.0])
    emg = np.array([80.0, 80.0])  # very high EMG
    out = emg_correct(bis, emg)
    assert (out >= 0.0).all() and (out <= 100.0).all()


def test_emg_correct_preserves_nan_in_emg():
    bis = np.array([50.0, 60.0])
    emg = np.array([np.nan, 50.0])
    out = emg_correct(bis, emg)
    assert out[0] == 50.0  # NaN EMG → no correction
    assert out[1] < 60.0   # 50 dB → correction applied


def test_sef_shape_and_range():
    eeg = make_eeg()
    out = sef(eeg)
    assert out.shape == (expected_n_epochs(len(eeg)),)
    valid = out[~np.isnan(out)]
    # Default band is (0.5, 30) → SEF must lie in that band
    assert (valid >= 0.0).all() and (valid <= 30.0).all()


def test_sef_rejects_bad_percentage():
    import pytest
    with pytest.raises(ValueError):
        sef(make_eeg(), percentage=0.0)
    with pytest.raises(ValueError):
        sef(make_eeg(), percentage=100.0)


def test_bcsef_shape():
    eeg = make_eeg()
    out = bcsef(eeg)
    assert out.shape == (expected_n_epochs(len(eeg)),)


def test_beta_ratio_shape():
    eeg = make_eeg()
    out = beta_ratio(eeg)
    assert out.shape == (expected_n_epochs(len(eeg)),)


def test_emg_estimate_shape():
    eeg = make_eeg()
    out = emg_estimate(eeg)
    assert out.shape == (expected_n_epochs(len(eeg)),)
    valid = out[~np.isnan(out)]
    # dB power is a real scalar — just confirm finite
    assert np.isfinite(valid).all()


def test_band_power_shape():
    eeg = make_eeg()
    out = band_power(eeg, band=(0.5, 4.0))
    assert out.shape == (expected_n_epochs(len(eeg)),)
    assert np.isfinite(out[~np.isnan(out)]).all()


def test_spectral_entropy_shape():
    eeg = make_eeg()
    out = spectral_entropy(eeg)
    assert out.shape == (expected_n_epochs(len(eeg)),)
    valid = out[~np.isnan(out)]
    assert (valid >= 0).all()  # entropy is non-negative


def test_predict_bis_shape_and_range():
    eeg = make_eeg()
    out = predict_bis(eeg)
    # predict_bis downsamples 2 Hz → 1 Hz, so we expect floor(N_epochs / 2)
    expected = expected_n_epochs(len(eeg)) // 2
    assert abs(len(out) - expected) <= 1, f"got {len(out)}, expected ~{expected}"
    assert (out >= 0).all() and (out <= 100).all()


def test_predict_bis_rejects_bad_fs():
    import pytest
    with pytest.raises(ValueError):
        predict_bis(make_eeg(), fs=256)


def test_predict_bis_smoothing_is_smoother_than_raw():
    eeg = make_eeg(n_seconds=200, seed=42)
    raw = predict_bis(eeg, smooth_W=0)
    sm = predict_bis(eeg, smooth_W=15.0)
    # Smoothed series should have smaller successive-difference variance.
    if len(raw) >= 4:
        assert np.var(np.diff(sm)) <= np.var(np.diff(raw)) + 1e-9
