"""Evaluation metrics for BIS-mimic models.

All metrics expect ``actual`` and ``predicted`` to be 1-D numpy arrays
of equal length, with a ``valid`` boolean mask selecting the epochs
that should be scored (typically SQI ≥ 80 and non-NaN).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Lee 2019 regime bins on the actual BIS value.
LEE_BINS = (0, 21, 41, 61, 78, 98)
LEE_BIN_LABELS = ("0-21", "21-41", "41-61", "61-78", "78-98")


def lin_concordance(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    return 2.0 * cov / (vx + vy + (mx - my) ** 2)


def global_metrics(actual: np.ndarray, predicted: np.ndarray, valid: np.ndarray) -> dict:
    """Global MAE / r / Lin's rc over a valid mask."""
    a, p = actual[valid], predicted[valid]
    out = {"n": int(valid.sum()), "mae": float("nan"), "r": float("nan"), "lin_rc": float("nan")}
    if len(a) < 2:
        return out
    out["mae"] = float(np.mean(np.abs(p - a)))
    if a.std() > 0 and p.std() > 0:
        out["r"] = float(np.corrcoef(a, p)[0, 1])
        out["lin_rc"] = lin_concordance(a, p)
    return out


def per_regime_mae(actual: np.ndarray, predicted: np.ndarray, valid: np.ndarray,
                   bins: tuple[float, ...] = LEE_BINS) -> dict[str, dict]:
    """MAE in each Lee regime bin, plus the per-bin sample count."""
    a, p = actual[valid], predicted[valid]
    out: dict[str, dict] = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        m = (a >= lo) & (a < hi)
        label = f"{lo}-{hi}"
        n = int(m.sum())
        mae = float(np.mean(np.abs(p[m] - a[m]))) if n > 0 else float("nan")
        out[label] = {"n": n, "mae": mae}
    return out


def evaluate(actual: np.ndarray, predicted: np.ndarray, valid: np.ndarray) -> dict:
    """One-shot: global + per-regime."""
    return {
        "global": global_metrics(actual, predicted, valid),
        "per_regime": per_regime_mae(actual, predicted, valid),
    }
