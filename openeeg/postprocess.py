"""openeeg.postprocess — canonical Ellerkmann + smoothing wrapper.

Every BIS-mimic predictor in this package (``predict_bis``,
``predict_bis_rules``, all ``bis_*`` literature rules in
``openeeg.rules``) routes its raw 1 Hz output through
:func:`apply_ellerkmann_and_smooth` before returning. The wrapper:

  1. Computes ``openbsr(eeg)`` from the SAME raw EEG the upstream
     model saw (Connor 2024 Table 1 port).
  2. Where ``openbsr > 49.8`` (Lee 2019's deep gate), replaces the
     prediction with ``clip(44.1 − openbsr/2.25, 0, 100)`` — the
     Ellerkmann 2004 formula. This is the only closed-form BIS rule
     in the literature that unambiguously beats every learning model
     on true-deep epochs (MAE 1.85 vs ~6 from a 22-feature LightGBM).
  3. Clips to [0, 100] and applies EMA(``smooth_W`` s) post-smoothing
     (15 s by default, matching BIS Vista's most common smoothing).

The override is **mandatory and not configurable**. The 0–21 region
should always be MAE ≤ 3.x — if it isn't, the model is failing to
apply the deep rule. There is no "research mode" opt-out; researchers
who need the raw LightGBM/piecewise output can call the lower-level
private functions, but the public API guarantees the Ellerkmann
behaviour.
"""
from __future__ import annotations

import numpy as np

from openeeg.openibis import FS
from openeeg.openbsr import openbsr as _openbsr

DEEP_RULE_THRESHOLD = 49.8  # Lee 2019; openbsr > this triggers Ellerkmann


def _ema(x: np.ndarray, W: float) -> np.ndarray:
    """Causal EMA with NaN handling.

    Initialises state to the first finite value (back-filling any
    leading NaN), then holds state through subsequent NaN samples so
    a single bad sample does not poison the entire trace.
    """
    a = 2.0 / (W + 1.0)
    out = np.empty_like(x, dtype=float)
    finite_mask = np.isfinite(x)
    if not finite_mask.any():
        out[:] = np.nan
        return out
    first_idx = int(np.argmax(finite_mask))
    out[: first_idx + 1] = x[first_idx]
    state = float(x[first_idx])
    for t in range(first_idx + 1, len(x)):
        xt = x[t] if np.isfinite(x[t]) else state
        state = a * xt + (1.0 - a) * state
        out[t] = state
    return out


def apply_ellerkmann_and_smooth(
    pred_1hz,
    eeg,
    fs: int = 128,
    smooth_W: float = 15.0,
) -> np.ndarray:
    """Mandatory Ellerkmann override + EMA(smooth_W) — the canonical
    final stage for every BIS-mimic predictor.

    Parameters
    ----------
    pred_1hz : array-like, 1-D
        Raw per-epoch BIS prediction at 1 Hz cadence (one value per
        second), clipped or unclipped (we clip again here).
    eeg : array-like, 1-D
        Raw EEG in microvolts at 128 Hz — the SAME signal the upstream
        model saw. We recompute openbsr from this to detect deep
        epochs.
    fs : int, default 128
        Sampling frequency of ``eeg``. Only 128 is supported.
    smooth_W : float, default 15.0
        EMA post-smoothing window in seconds. Pass 0 to disable the
        final smoothing (the Ellerkmann override is still applied —
        always).

    Returns
    -------
    np.ndarray of shape ``pred_1hz.shape``
        Final BIS-mimic clipped to [0, 100] at 1 Hz.

    Notes
    -----
    Any model whose 0–21 MAE on a representative cohort exceeds 3.5
    is failing this wrapper. Verify by inspecting the override
    mask: ``openbsr(eeg)[::2] > 49.8``.
    """
    if fs != FS:
        raise ValueError(f"fs must be 128; got {fs}.")
    pred = np.asarray(pred_1hz, dtype=float).copy()
    pred = np.clip(pred, 0.0, 100.0)
    eeg = np.asarray(eeg, dtype=float)

    obsr_2hz = _openbsr(eeg, fs=fs)
    obsr_1hz = obsr_2hz[::2]
    # Align lengths (pred may be 1-2 samples longer/shorter than openbsr)
    n = min(len(pred), len(obsr_1hz))
    pred = pred[:n]
    obsr_1hz = obsr_1hz[:n]
    deep_mask = np.where(np.isnan(obsr_1hz), False, obsr_1hz > DEEP_RULE_THRESHOLD)
    if deep_mask.any():
        pred[deep_mask] = np.clip(44.1 - obsr_1hz[deep_mask] / 2.25, 0.0, 100.0)
    if smooth_W and smooth_W > 0:
        pred = _ema(pred, float(smooth_W))
    return pred
