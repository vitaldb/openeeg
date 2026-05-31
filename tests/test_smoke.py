"""Smoke tests: every public function returns the right shape on a synthetic input."""
import numpy as np

from openeeg import openibis, openbsr


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
