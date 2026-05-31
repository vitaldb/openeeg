"""Smoke tests: every public function returns the right shape on a synthetic input."""
import numpy as np

from openeeg import openibis, openbsr, emg_correct


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
