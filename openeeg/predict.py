"""openeeg.predict — high-level BIS-mimic prediction.

``predict_bis(eeg)`` takes a raw 128 Hz EEG channel and returns a
1 Hz BIS-mimic score using a two-stage pipeline:

  STAGE 1 — Ellerkmann deep rule
    Where ``openbsr(eeg) > 49.8`` (Lee 2019's deep gate), the output is
    ``clip(44.1 − openbsr/2.25, 0, 100)``. This is the only published
    deep-regime BIS formula in the literature; it reaches MAE 1.85 on
    the true-deep subset, vs ~6 from a learned LightGBM.

  STAGE 2 — predict_bis_v3 LightGBM on 18 features
    Elsewhere, a gradient-boosted regressor predicts BIS from 18
    EEG-derived features (down from v2's 22; the four lowest-gain
    features — ``emg_proxy_dt``, ``sef95_dt``, ``openibis_quazi_dt``,
    ``spectral_entropy`` — were removed in scripts/34 with no material
    accuracy loss).

  STAGE 3 — EMA(15 s) post-smoothing

The bundled model ``openeeg/models/predict_bis_v3.txt`` (10 MB) was
trained on 397 VitalDB BIS cases (``caseid % 10 ∈ {0..7}`` train fold)
filtered to:
  * SQI ≥ 80 on every epoch
  * oracle smoothing window W = 15 s (~80 % of VitalDB — Vista's
    most common setting; window inferred per case from the W ∈
    {15, 30, 45} that minimises ``MAE(EMA(predict_bis_v1, W),
    actual_BIS)`` on training data)

No sample weighting is applied (the deep regime is handled by the
explicit Ellerkmann rule instead of LightGBM coefficients).

Validation on the 80-case W = 15 sub-cohort of the val fold
(epoch-weighted, SQI ≥ 80, EMA(15 s) post-smooth)::

      MAE = 3.67   Pearson r = 0.893   Lin's rc = 0.886

      0-21  21-41  41-61  61-78  78-98
      3.20  3.58   3.59   4.50   5.70

The 0-21 (deep) bin is locked to the Ellerkmann formula and improves
from v2's 6.15 to 3.20 (−48 %) at no cost to other regimes. Total
overall MAE is 0.05 worse than v2's 3.63 — a free win on
interpretability of the deep regime in exchange for almost nothing.

See ``scripts/35_train_predict_bis_v3.py`` for the reproduction
pipeline and ``scripts/34_v2_feature_ablation.py`` for the
feature-trimming evidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from openeeg.openibis import openibis, bsr as _openibis_bsr, FS
from openeeg.features import (
    sef, bcsef, beta_ratio, emg_estimate, band_power,
)
from openeeg.postprocess import apply_ellerkmann_and_smooth

_MODEL_PATH = Path(__file__).parent / "models" / "predict_bis_v3.txt"

# v3 keeps 18 of v2's 22 features; the 4 dropped features (gain rank 19-22)
# were emg_proxy_dt, sef95_dt, openibis_quazi_dt, spectral_entropy.
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
    "p_theta",
    "p_alpha",
    "p_beta",
    "p_lowgamma",
    "openibis_quazi_5s",
    "openibis_quazi_10s",
    "openibis_quazi_60s",
    "openibis_quazi_30s_dt",
    "p_delta",
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
    """Build the 18-column v3 feature matrix at 2 Hz, in _FEATURE_ORDER."""
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
    pred_quazi_30s_dt = _diff(pred_quazi_30s)

    # Must match _FEATURE_ORDER exactly
    columns = [
        pred_paper, pred_quazi, pred_quazi_30s,
        bsr_p, bsr_q,
        sef_v, bcsef_v, br, emg,
        p_theta, p_alpha, p_beta, p_lowgamma,
        pred_quazi_5s, pred_quazi_10s, pred_quazi_60s,
        pred_quazi_30s_dt,
        p_delta,
    ]
    n = min(len(c) for c in columns)
    return np.column_stack([c[:n] for c in columns])


def _raw_lgbm(eeg, fs: int = 128) -> np.ndarray:
    """Raw v3 LightGBM prediction at 1 Hz (no wrapper)."""
    if fs != FS:
        raise ValueError(f"lgbm method requires fs=128; got {fs}.")
    eeg = np.asarray(eeg, dtype=float)
    feats_2hz = _extract_features(eeg)
    feats_1hz = feats_2hz[::2]
    if len(feats_1hz) == 0:
        return np.empty(0)
    booster = _get_booster()
    return booster.predict(feats_1hz)


_METHOD_DISPATCH = {
    # name → (kind, short description, val MAE)
    "gbm":      ("learning", "v3 18-feature LightGBM 500/31 (recommended)", 3.77),
    "lee":      ("learning", "30-leaf depth-5 piecewise linear (Lee 2019 family)", 4.21),
    "connor":   ("rule",     "Connor 2023 openibis logistic mixer",         5.06),
    "sleigh":   ("rule",     "Sleigh 2001 SE50d gate → Morimoto blend",     5.88),
    "morimoto": ("rule",     "Morimoto 2004 BcSEF↔BetaRatio sigmoid blend", 6.08),
    "cusenza":  ("rule",     "Cusenza 2013 Higuchi-FD × BSR",               7.91),
}


def _dispatch_raw(eeg, fs: int, method: str) -> np.ndarray:
    if method == "gbm":
        return _raw_lgbm(eeg, fs=fs)
    if method == "lee":
        from openeeg.piecewise import _raw_piecewise
        return _raw_piecewise(eeg, fs=fs)
    if method == "connor":
        from openeeg.openibis import openibis as _openibis
        raw_2hz = _openibis(eeg, fs=fs, bsr="quazi", deep="paper")
        return raw_2hz[::2]  # 2 Hz → 1 Hz
    if method == "sleigh":
        from openeeg.rules import _raw_sleigh_gated_morimoto
        return _raw_sleigh_gated_morimoto(eeg, fs=fs)
    if method == "morimoto":
        from openeeg.rules import _raw_morimoto_combined
        return _raw_morimoto_combined(eeg, fs=fs)
    if method == "cusenza":
        from openeeg.rules import _raw_cusenza_fdsr
        return _raw_cusenza_fdsr(eeg, fs=fs)
    valid = ", ".join(f"{k!r}" for k in _METHOD_DISPATCH)
    raise ValueError(f"Unknown method {method!r}. Valid: {valid}.")


def predict_bis(eeg, fs: int = 128, smooth_W: float = 15.0,
                  method: str = "gbm") -> np.ndarray:
    """BIS-mimic score at 1 Hz from a raw 128 Hz EEG channel.

    Single entry point for every BIS-mimic in the package. The
    ``method`` argument selects the underlying predictor; the
    canonical :func:`openeeg.postprocess.apply_ellerkmann_and_smooth`
    wrapper is then applied unconditionally (Ellerkmann override at
    ``openbsr > 49.8`` + EMA(``smooth_W`` s) post-smoothing).

    Methods (val W=15 MAE on the 80-case SQI≥80 cohort)
    ---------------------------------------------------
    Learning-based (trained on the 397-case VitalDB W=15 train fold):
      * ``"gbm"``       — bundled v3 18-feature LightGBM         (3.77)
                          (500 trees × 31 leaves, 1.5 MB; shrunk
                          from 2000×63 / 11.7 MB with +0.10 BIS
                          cost, see scripts/38)
      * ``"lee"``       — 30-leaf data-driven piecewise linear   (4.21)
                          (sklearn depth-5 tree + per-leaf 8-term
                          OLS; 27 kB. Structurally a Lee 2019
                          family model but with data-driven splits
                          and coefficients. 0-21 region MAE 2.92
                          thanks to the mandatory Ellerkmann override.)

    Rule-based (closed-form literature formulas, no learning):
      * ``"connor"``    — Connor 2023 openibis logistic mixer    (5.06)
      * ``"sleigh"``    — Sleigh 2001 SE50d gate + Morimoto      (5.88)
      * ``"morimoto"``  — Morimoto 2004 BcSEF/BetaRatio blend    (6.08)
      * ``"cusenza"``   — Cusenza 2013 Higuchi-FD × BSR          (7.91)

    Parameters
    ----------
    eeg : array-like, 1-D
        Raw EEG in microvolts at 128 Hz.
    fs : int, default 128
        Sampling frequency. Only 128 is supported (BIS standard).
    smooth_W : float, default 15.0
        EMA smoothing window in seconds. Pass ``0`` to disable
        smoothing (Ellerkmann override is still applied).
    method : str, default ``"gbm"``
        Which underlying predictor to use. See the table above.

    Returns
    -------
    np.ndarray of shape (N,)
        BIS-mimic in [0, 100] at 1 Hz. NaN where the model cannot
        produce a value.

    Examples
    --------
    >>> from openeeg import predict_bis
    >>> bis = predict_bis(eeg)                      # default LightGBM
    >>> bis = predict_bis(eeg, method="lee")        # learned, interpretable
    >>> bis = predict_bis(eeg, method="connor")     # pure literature rule

    See Also
    --------
    openeeg.postprocess.apply_ellerkmann_and_smooth : the wrapper every
        method shares.
    openeeg.openibis : the low-level Connor 2023 implementation used by
        ``method="connor"``.
    """
    if fs != FS:
        raise ValueError(f"predict_bis() requires fs=128; got {fs}.")
    eeg = np.asarray(eeg, dtype=float)
    raw = _dispatch_raw(eeg, fs=fs, method=method)
    return apply_ellerkmann_and_smooth(raw, eeg, fs=fs, smooth_W=smooth_W)
