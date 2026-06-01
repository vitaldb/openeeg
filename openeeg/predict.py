"""openeeg.predict — high-level BIS-mimic prediction.

``predict_bis(eeg)`` takes a raw 128 Hz EEG channel and returns a
1 Hz BIS-mimic score by running a LightGBM regressor over a fixed
set of 22 spectral features (15 base + 7 short-context / velocity).

The bundled model ``openeeg/models/predict_bis_v2.txt`` was trained on
397 VitalDB BIS cases drawn from the ``caseid % 10 ∈ {0..7}`` train
fold, filtered to:

  * SQI ≥ 80 on every epoch, and
  * oracle smoothing window W = 15 s (~80 % of the cohort — Vista's
    most common setting). The window is inferred per case by picking
    the W ∈ {15, 30, 45} that minimises ``MAE(EMA(predict_bis_v1, W),
    actual_BIS)`` on training data.

No sample weighting is applied — the model targets overall accuracy on
the BIS 21–100 range, since the deep regime (BIS < 21) is better
addressed by a rule-based override (e.g. the Ellerkmann formula
``BIS ≈ 44.1 − BSR / 2.25`` on a reliable BSR signal) applied by the
caller.

Validation on the 80-case W = 15 sub-cohort of the val fold
(epoch-weighted, SQI ≥ 80, EMA(15 s) post-smooth)::

      MAE = 3.63   Pearson r = 0.896   Lin's rc = 0.889

      0-21  21-41  41-61  61-78  78-98
      6.15  3.53   3.54   4.50   5.56

Compared to predict_bis_v1 on the same filtered cohort:

      MAE = 3.76   r = 0.891   (v2 is 3.5 % better overall, 17 % better
      0-21 = 7.43               at 0–21 even without explicit deep
                                weighting)

See ``scripts/06_extract_features.py``, ``scripts/16_augment_phase3f.py``,
``scripts/18_oracle_W_cache.py``, and ``scripts/20_train_w15_filtered.py``
for the reproduction pipeline. predict_bis_v1 remains the bundled
fall-back if v2 is removed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from openeeg.openibis import openibis, bsr as _openibis_bsr, FS
from openeeg.features import (
    sef, bcsef, beta_ratio, emg_estimate, band_power, spectral_entropy,
)

_MODEL_PATH = Path(__file__).parent / "models" / "predict_bis_v2.txt"

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
    # Phase 3f short-context + velocity additions
    "openibis_quazi_5s",
    "openibis_quazi_10s",
    "openibis_quazi_60s",
    "openibis_quazi_dt",
    "openibis_quazi_30s_dt",
    "sef95_dt",
    "emg_proxy_dt",
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


def _rolling_mean_2hz(x: np.ndarray, window_s: float) -> np.ndarray:
    """Causal trailing mean over a window expressed in seconds at 2 Hz."""
    w = max(int(round(window_s * 2)), 1)
    kernel = np.ones(w) / w
    return np.convolve(np.where(np.isnan(x), 0.0, x), kernel, mode="full")[: len(x)]


def _causal_30s_mean_2hz(x: np.ndarray) -> np.ndarray:
    """Backwards-compat alias kept for diagnostic scripts."""
    return _rolling_mean_2hz(x, 30.0)


def _diff(x: np.ndarray) -> np.ndarray:
    """First-difference at the input cadence (2 Hz here); first element 0."""
    out = np.empty_like(x, dtype=float)
    out[0] = 0.0
    out[1:] = x[1:] - x[:-1]
    return out


def _extract_features(eeg: np.ndarray) -> np.ndarray:
    """Build the 22-column feature matrix at 2 Hz."""
    pred_paper = openibis(eeg, bsr="paper", deep="paper")
    pred_quazi = openibis(eeg, bsr="quazi", deep="paper")
    pred_quazi_5s = _rolling_mean_2hz(pred_quazi, 5.0)
    pred_quazi_10s = _rolling_mean_2hz(pred_quazi, 10.0)
    pred_quazi_30s = _rolling_mean_2hz(pred_quazi, 30.0)
    pred_quazi_60s = _rolling_mean_2hz(pred_quazi, 60.0)
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
    # Velocity features (differences at 2 Hz)
    pred_quazi_dt = _diff(pred_quazi)
    pred_quazi_30s_dt = _diff(pred_quazi_30s)
    sef_dt = _diff(sef_v)
    emg_dt = _diff(emg)

    columns = [
        pred_paper, pred_quazi, pred_quazi_30s,
        bsr_p, bsr_q,
        sef_v, bcsef_v, br, emg,
        p_delta, p_theta, p_alpha, p_beta, p_lowgamma,
        se,
        pred_quazi_5s, pred_quazi_10s, pred_quazi_60s,
        pred_quazi_dt, pred_quazi_30s_dt,
        sef_dt, emg_dt,
    ]
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
    ``smooth_W`` seconds, matching the BIS Vista's most common
    smoothing setting (15 s) on the VitalDB cohort.

    Parameters
    ----------
    eeg : array-like, 1-D
        Raw EEG in microvolts. Must be sampled at 128 Hz.
    fs : int, default 128
        Sampling frequency in Hz. Only 128 is supported (BIS standard).
    smooth_W : float, default 15.0
        EMA smoothing window in seconds. Pass ``0`` (or ``None``) to
        return the raw model output (more dynamic, higher MAE against
        the smoothed Vista BIS).

    Returns
    -------
    np.ndarray of shape (N,)
        BIS-like score clipped to [0, 100] at 1 Hz (one value per
        second of input). NaN where the model cannot produce a value.

    See Also
    --------
    openeeg.openibis : Connor 2022 paper-faithful BIS reimplementation.
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
