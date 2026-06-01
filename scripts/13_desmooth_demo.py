"""Demonstrate BIS-Vista smoothing inversion (desmooth).

Forward smoothing model: ``y[t] = (1/W) sum_{k=0..W-1} x[t-k]``.
Exact causal inverse: ``x[t] = W*(y[t] - y[t-1]) + x[t-W]``.

For a handful of representative val cases, plot:
  * actual BIS (smoothed, as recorded)
  * desmoothed BIS at estimated W
  * predict_bis output (already running on raw EEG)
  * actual rolling-mean (sanity: should match smoothed)

Computes MAE of predict_bis vs actual AND vs desmoothed — if
desmoothed matches predict_bis better, the smoothing IS the
ceiling of the trained model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg.cohort import load_case, preprocess_eeg
from openeeg import predict_bis

RESULTS = Path(__file__).resolve().parents[1] / "results"


def desmooth_uniform(y: np.ndarray, W: int) -> np.ndarray:
    """Invert causal trailing-mean of window W. Numerically unstable
    when y is quantised — only useful as a baseline."""
    n = len(y)
    x = np.zeros(n, dtype=float)
    x[:W] = y[:W]
    for t in range(W, n):
        x[t] = W * (y[t] - y[t - 1]) + x[t - W]
    return x


def desmooth_ema(y: np.ndarray, W: int) -> np.ndarray:
    """Invert an exponential moving average.

    Forward: y[t] = α x[t] + (1-α) y[t-1] with α = 2 / (W + 1).
    Inverse: x[t] = (y[t] - (1-α) y[t-1]) / α  — non-recursive, stable.

    Output is clipped to [0, 100] so the inversion gain on isolated
    quantisation jumps does not produce out-of-range BIS values.
    """
    alpha = 2.0 / (W + 1.0)
    n = len(y)
    x = np.zeros(n, dtype=float)
    x[0] = y[0]
    x[1:] = (y[1:] - (1.0 - alpha) * y[:-1]) / alpha
    return np.clip(x, 0.0, 100.0)


def desmooth_wiener(y: np.ndarray, W: int, lam: float = 0.01) -> np.ndarray:
    """Frequency-domain Wiener desmoothing of a uniform-W trailing mean.

    Higher ``lam`` is more stable but recovers less detail; tune by
    eye on the plots.
    """
    n = len(y)
    h = np.zeros(n)
    h[:W] = 1.0 / W
    Y = np.fft.fft(y)
    H = np.fft.fft(h)
    X_hat = Y * np.conj(H) / (np.abs(H) ** 2 + lam)
    x = np.real(np.fft.ifft(X_hat))
    return np.clip(x, 0.0, 100.0)


def smoothed_check(x: np.ndarray, W: int) -> np.ndarray:
    """Forward smoothing (causal trailing mean) for sanity comparison."""
    kernel = np.ones(W) / W
    return np.convolve(x, kernel, mode="full")[: len(x)]


def w_from_max_step(bis: np.ndarray) -> float:
    db = np.diff(bis)
    db = db[np.isfinite(db)]
    if len(db) < 30:
        return float("nan")
    p99 = np.percentile(np.abs(db), 99)
    rng = float(np.nanmax(bis) - np.nanmin(bis))
    if p99 < 0.05 or rng < 20:
        return float("nan")
    return rng / p99


def plot_case(caseid: int, W_est: int, outpath: Path):
    case = load_case(caseid)
    if case is None:
        print(f"case {caseid}: load failed")
        return
    eeg = preprocess_eeg(case["eeg"])
    actual = case["bis"]
    sqi = case["sqi"]

    pred = predict_bis(eeg)
    n = min(len(pred), len(actual), len(sqi))
    pred, actual, sqi = pred[:n], actual[:n], sqi[:n]
    t = np.arange(n)

    # Three desmoothing flavours
    a_filled = np.where(np.isnan(actual), 50.0, actual)
    Wmax = max(W_est, 2)
    desm_uniform = desmooth_uniform(a_filled, Wmax)
    desm_ema     = desmooth_ema(a_filled, Wmax)
    desm_wiener  = desmooth_wiener(a_filled, Wmax, lam=0.01)

    # MAEs only on SQI>=80 epochs
    valid = ~np.isnan(actual) & ~np.isnan(pred) & ~np.isnan(sqi) & (sqi >= 80)
    mae_actual = float(np.mean(np.abs(pred[valid] - actual[valid])))
    mae_uni    = float(np.mean(np.abs(pred[valid] - desm_uniform[valid])))
    mae_ema    = float(np.mean(np.abs(pred[valid] - desm_ema[valid])))
    mae_wien   = float(np.mean(np.abs(pred[valid] - desm_wiener[valid])))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(t, actual, color="black", lw=1.0, alpha=0.9, label="Actual BIS (smoothed)")
    axes[0].plot(t, desm_ema, color="tab:purple", lw=0.9, alpha=0.85,
                 label=f"Desmoothed EMA (W={W_est} s)")
    axes[0].plot(t, desm_wiener, color="tab:green", lw=0.7, alpha=0.6,
                 label=f"Desmoothed Wiener λ=0.01")
    axes[0].plot(t, pred, color="tab:red", lw=1.0, alpha=0.7, label="predict_bis")
    axes[0].set_ylabel("BIS")
    axes[0].set_ylim(-10, 110)
    axes[0].set_title(f"Case {caseid}  —  est W={W_est} s  |  "
                       f"MAE(pred,actual)={mae_actual:.2f}  "
                       f"EMA={mae_ema:.2f}  Wiener={mae_wien:.2f}  "
                       f"Uniform={mae_uni:.1f}")
    axes[0].legend(loc="upper right")
    # SQI shade
    low_sqi = np.isnan(sqi) | (sqi < 80)
    shade = []
    in_low, start = False, 0
    for i in range(n):
        if low_sqi[i] and not in_low:
            start, in_low = i, True
        elif not low_sqi[i] and in_low:
            shade.append((start, i)); in_low = False
    if in_low:
        shade.append((start, n))
    for s0, s1 in shade:
        for ax in axes:
            ax.axvspan(s0, s1, color="gray", alpha=0.2)

    # Residuals — only the stable variants
    axes[1].plot(t, pred - actual,    color="tab:red",    lw=0.8, alpha=0.6, label="pred − actual")
    axes[1].plot(t, pred - desm_ema,  color="tab:purple", lw=0.8, alpha=0.6, label="pred − desm_EMA")
    axes[1].plot(t, pred - desm_wiener, color="tab:green", lw=0.6, alpha=0.5, label="pred − desm_Wiener")
    axes[1].axhline(0, color="black", lw=0.4)
    axes[1].set_ylabel("residual")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    print(f"  case {caseid:>4d}: W={W_est:>2d}  "
          f"MAE_act={mae_actual:5.2f}  EMA={mae_ema:5.2f}  "
          f"Wiener={mae_wien:5.2f}  Uniform={mae_uni:7.1f}")


def main():
    sm = pd.read_csv(RESULTS / "smoothing_estimates.csv")
    sm = sm.dropna(subset=["W_from_max_step"]).copy()
    sm["W_int"] = sm["W_from_max_step"].round().astype(int)

    # Pick three representative cases: low/median/high W
    sm_sorted = sm.sort_values("W_int")
    low = sm_sorted.iloc[len(sm_sorted) // 10]
    med = sm_sorted.iloc[len(sm_sorted) // 2]
    high = sm_sorted.iloc[len(sm_sorted) * 9 // 10]

    print(f"Selected cases:")
    print(f"  low W:    case {int(low.caseid)}  W~{int(low.W_int)} s")
    print(f"  median W: case {int(med.caseid)}  W~{int(med.W_int)} s")
    print(f"  high W:   case {int(high.caseid)}  W~{int(high.W_int)} s")
    print()
    print(f"Comparison MAE(pred vs actual) vs MAE(pred vs desmoothed):")

    RESULTS.mkdir(exist_ok=True)
    for tag, row in [("low", low), ("med", med), ("high", high)]:
        cid = int(row.caseid)
        W = int(row.W_int)
        plot_case(cid, W, RESULTS / f"desmooth_{tag}_case{cid}_W{W}.png")

    # Also run on case 988 (the deep-suppression worst case)
    if 988 in sm["caseid"].values:
        W988 = int(sm[sm["caseid"] == 988].iloc[0]["W_int"])
        plot_case(988, W988, RESULTS / f"desmooth_worst_case988_W{W988}.png")


if __name__ == "__main__":
    main()
