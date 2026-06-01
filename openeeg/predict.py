"""openeeg.predict — high-level BIS-mimic prediction.

``predict_bis(eeg)`` takes a raw 128 Hz EEG channel and returns a
1 Hz BIS-mimic score by running a LightGBM regressor over a fixed
set of 15 spectral features. The model bundled with the package
was trained on 498 VitalDB BIS cases (caseid % 10 in {0..7}, sorted)
and validated on 100 cases (residue 8) with the following
epoch-weighted metrics::

      MAE = 4.25   Pearson r = 0.850   Lin's rc = 0.844

      0-21  21-41  41-61  61-78  78-98
      15.16  4.14   4.18   4.56   6.06

See ``scripts/06_extract_features.py`` and ``scripts/07_train_lightgbm.py``
to reproduce. The model file lives at
``openeeg/models/predict_bis_v1.txt``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from openeeg.openibis import openibis, bsr as _openibis_bsr, FS
from openeeg.features import (
    sef, bcsef, beta_ratio, emg_estimate, band_power, spectral_entropy,
)

_MODEL_PATH = Path(__file__).parent / "models" / "predict_bis_v1.txt"

_FEATURE_ORDER = (
    "openibis_paper",
    "openibis_quazi",
    "openibis_quazi_30s",
    "bsr_paper",
    "bsr_quazi",
    "sef95",
    "bcsef",
    "beta_ratio",
    "emg_proxy",
    "p_delta",
    "p_theta",
    "p_alpha",
    "p_beta",
    "p_lowgamma",
    "spectral_entropy",
)

_booster_cache: Optional[object] = None


def _get_booster():
    """Lazy-load the bundled LightGBM model."""
    global _booster_cache
    if _booster_cache is None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "predict_bis() requires lightgbm. Install with "
                "`pip install lightgbm`."
            ) from exc
        if not _MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Bundled model missing: {_MODEL_PATH}. Reinstall the package."
            )
        _booster_cache = lgb.Booster(model_file=str(_MODEL_PATH))
    return _booster_cache


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average; NaN-safe."""
    kernel = np.ones(w) / w
    return np.convolve(np.where(np.isnan(x), 0.0, x), kernel, mode="same")


def _extract_features(eeg: np.ndarray) -> np.ndarray:
    """Build the 15-column feature matrix at 2 Hz."""
    pred_paper = openibis(eeg, bsr="paper", deep="paper")
    pred_quazi = openibis(eeg, bsr="quazi", deep="paper")
    pred_quazi_30s = _rolling_mean(pred_quazi, 60)  # 60 strides @ 2Hz = 30 s
    bsr_p = _openibis_bsr(eeg, kind="paper")
    bsr_q = _openibis_bsr(eeg, kind="quazi")
    sef_v = sef(eeg)
    bcsef_v = bcsef(eeg)
    br = beta_ratio(eeg)
    emg = emg_estimate(eeg)
    p_delta = band_power(eeg, band=(0.5, 4.0))
    p_theta = band_power(eeg, band=(4.0, 8.0))
    p_alpha = band_power(eeg, band=(8.0, 13.0))
    p_beta = band_power(eeg, band=(13.0, 30.0))
    p_lowgamma = band_power(eeg, band=(30.0, 47.0))
    se = spectral_entropy(eeg)

    columns = [
        pred_paper, pred_quazi, pred_quazi_30s,
        bsr_p, bsr_q,
        sef_v, bcsef_v, br, emg,
        p_delta, p_theta, p_alpha, p_beta, p_lowgamma,
        se,
    ]
    # Equal lengths (all on the same 2 Hz epoch grid)
    n = min(len(c) for c in columns)
    return np.column_stack([c[:n] for c in columns])


def _ema(x: np.ndarray, W: float) -> np.ndarray:
    """Causal exponential moving average with effective window W seconds."""
    alpha = 2.0 / (W + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out


def predict_bis(eeg, fs: int = 128, smooth_W: float = 15.0) -> np.ndarray:
    """BIS-mimic score at 1 Hz from a raw 128 Hz EEG channel.

    Internally applies an EMA post-smoothing with effective window
    ``smooth_W`` seconds, which empirically matches the BIS Vista's
    smoothing convention and reduces val-cohort MAE 4.25 → 4.03
    relative to the raw model output.

    Parameters
    ----------
    eeg : array-like, 1-D
        Raw EEG in microvolts. Must be sampled at 128 Hz.
    fs : int, default 128
        Sampling frequency in Hz. Only 128 is supported (BIS standard).
    smooth_W : float, default 15.0
        EMA smoothing window in seconds. Pass ``0`` or ``None`` to
        return the raw model output (more dynamic, higher MAE against
        the smoothed Vista BIS).

    Returns
    -------
    np.ndarray of shape (N,)
        BIS-like score clipped to [0, 100] at 1 Hz (one value per
        second of input). NaN where the model cannot produce a value.
    """
    if fs != FS:
        raise ValueError(f"predict_bis() requires fs=128; got {fs}.")
    eeg = np.asarray(eeg, dtype=float)
    feats_2hz = _extract_features(eeg)
    feats_1hz = feats_2hz[::2]
    if len(feats_1hz) == 0:
        return np.empty(0)
    booster = _get_booster()
    pred = np.clip(booster.predict(feats_1hz), 0.0, 100.0)
    if smooth_W and smooth_W > 0:
        pred = _ema(pred, float(smooth_W))
    return pred
