"""openbsr — frequency-domain burst-suppression ratio.

Reimplemented from the prose description in:

    Connor, C. W. (2024). OpenBSR — Open Reimplementation of the BIS
    Burst Suppression Ratio. (See ``references/2025 Connor, OpenBSR.pdf``.)

The original paper's Table 1 contains the MATLAB source as a raster
image, so this Python translation is paraphrased from the prose
description rather than copied verbatim. It is best-effort and should
be cross-checked against the paper's Table 1 before being relied upon
clinically.

Algorithm sketch (prose, Connor 2024):
  * Per 0.5-s epoch, compute spectral power in two bands:
        HG = mean PSD over 31–47 Hz
        LL = mean PSD over  1–30 Hz
  * Mark the epoch suppressed if **both** HG and LL drop below an
    adaptive threshold.
  * The threshold is **low** (close to a fixed reference) when the
    high-gamma power is high relative to the recent spectrogram —
    i.e. consciousness / sedation, where the chance of suppression
    is small.
  * When high-gamma drops, the threshold **rises** in proportion to
    the 95th-percentile of the 1–20 Hz band power over the **preceding
    20 minutes**.
  * The reference power level is calibrated against a sustained 5 µV
    signal (the same amplitude as Connor 2023's BSR threshold).
  * BSR = trailing 63-second mean of the per-epoch suppression flag,
    in percent. Identical to the Connor 2023 window.

This module is marked experimental and not yet exposed at the package
top level for that reason.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as signal

FS = 128
STRIDE_SEC = 0.5
STRIDE = int(FS * STRIDE_SEC)


def _psd_simple(x: np.ndarray) -> np.ndarray:
    """One-sided Blackman PSD."""
    w = signal.windows.blackman(len(x))
    f = np.fft.fft(w * (x - x.mean()))
    n = len(x)
    return 2.0 * np.abs(f[: n // 2]) ** 2 / (n * np.sum(w ** 2))


def _reference_power() -> float:
    """Reference high-gamma power from a sustained 5-µV sinusoid.

    Used to scale the adaptive threshold so that, at the no-suppression
    floor, an epoch with 5 µV-equivalent activity above 31 Hz is *not*
    flagged as suppressed.
    """
    n = 4 * STRIDE  # 2 seconds, matching the openibis PSD window
    t = np.arange(n) / FS
    # 35 Hz mid-HG carrier at 5 µV peak; phase irrelevant
    ref = 5.0 * np.sin(2.0 * np.pi * 35.0 * t)
    P = _psd_simple(ref)
    # mean PSD over 31–47 Hz, in dB
    bin_hz = FS / n  # 0.5
    lo, hi = int(31 / bin_hz), int(47 / bin_hz) + 1
    return float(np.mean(10.0 * np.log10(np.maximum(P[lo:hi], 1e-30))))


def openbsr(eeg, fs: int = 128) -> np.ndarray:
    """Compute OpenBSR burst-suppression ratio (percent) at 2 Hz.

    Parameters
    ----------
    eeg : array-like, 1-D
        Raw EEG in µV, sampled at 128 Hz.

    Returns
    -------
    np.ndarray of shape (N,)
        BSR percentage trajectory, 2 Hz.

    Notes
    -----
    Frequency-domain detector with an adaptive threshold. See module
    docstring for algorithm sketch. Best-effort prose-based port; not
    yet validated bit-for-bit against the paper's Table 1.
    """
    eeg = np.asarray(eeg, dtype=float)
    if fs != FS:
        raise ValueError(f"openbsr() requires fs=128; got {fs}.")

    psd_window = 4 * STRIDE  # 2 s
    bin_hz = FS / psd_window  # 0.5 Hz
    hg_lo, hg_hi = int(31 / bin_hz), int(47 / bin_hz) + 1
    ll_lo, ll_hi = int(1 / bin_hz), int(30 / bin_hz) + 1
    lo20_lo, lo20_hi = int(1 / bin_hz), int(20 / bin_hz) + 1

    N = int(np.floor((len(eeg) - FS) / STRIDE) - 10)
    if N <= 0:
        return np.empty(0)

    hg_db = np.full(N, np.nan)  # high-gamma dB power per epoch
    ll_db = np.full(N, np.nan)  # low-band dB power
    lo20_db = np.full(N, np.nan)  # 1–20 Hz dB power (for threshold history)

    for n in range(N):
        s = (n + 4) * STRIDE
        e = s + psd_window
        if e > len(eeg):
            continue
        P = _psd_simple(eeg[s:e])
        hg_db[n] = 10.0 * np.log10(np.maximum(np.mean(P[hg_lo:hg_hi]), 1e-30))
        ll_db[n] = 10.0 * np.log10(np.maximum(np.mean(P[ll_lo:ll_hi]), 1e-30))
        lo20_db[n] = 10.0 * np.log10(np.maximum(np.mean(P[lo20_lo:lo20_hi]), 1e-30))

    ref_db = _reference_power()  # ≈ HG dB for sustained 5-µV signal

    # 20-minute trailing 95th percentile of 1–20 Hz dB power
    W = int(20 * 60 / STRIDE_SEC)  # = 2400 strides
    pct95 = np.full(N, np.nan)
    for i in range(N):
        j = max(0, i - W + 1)
        seg = lo20_db[j : i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) > 0:
            pct95[i] = float(np.percentile(seg, 95))

    # Recent (~63 s) median of HG dB, used to decide if HG is "high"
    K = int(63.0 / STRIDE_SEC)
    hg_recent_med = np.full(N, np.nan)
    for i in range(N):
        j = max(0, i - K + 1)
        seg = hg_db[j : i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) > 0:
            hg_recent_med[i] = float(np.median(seg))

    # Adaptive threshold (in dB):
    #   when HG is high relative to recent median → threshold ≈ ref_db (low)
    #   when HG drops → threshold rises toward the 95-th percentile of LL
    # Blend factor σ ∈ [0,1] from a smoothstep on (hg_db − hg_recent_med + 6 dB).
    delta = hg_db - hg_recent_med
    sigma = 1.0 - np.clip((delta + 6.0) / 12.0, 0.0, 1.0)
    threshold_db = ref_db * (1.0 - sigma) + pct95 * sigma

    # An epoch is suppressed if BOTH HG and LL are below threshold.
    BSRmap = (hg_db < threshold_db) & (ll_db < threshold_db)
    BSRmap[np.isnan(threshold_db) | np.isnan(hg_db) | np.isnan(ll_db)] = False

    # Trailing 63-second mean (same window as Connor 2023)
    cs = np.concatenate([[0.0], np.cumsum(BSRmap.astype(float))])
    BSR = np.empty(N)
    for i in range(N):
        j = max(0, i - K + 1)
        BSR[i] = (cs[i + 1] - cs[j]) / (i - j + 1)
    return 100.0 * BSR
