"""openbsr — Connor 2024 OpenBSR, line-by-line port of Table 1.

Source:
    Connor, C. W. (2024). OpenBSR — an open algorithm for Burst Suppression
    Rate concordant with the BIS monitor.

This implementation is a direct Python port of the MATLAB Table 1
listing. Variable names follow the paper exactly. The previous
prose-based version diverged in several places (PSD window length,
single vs averaged threshold, missing refBand and amplitude
compression) and is replaced here.

Algorithm summary
-----------------
For each 0.5-s epoch ``n``:

1. Take a 1-s segment of EEG (two strides), starting at sample
   ``(n + 7.5) * stride`` (1-indexed MATLAB ``(n + 6.5) * stride + 1``).
2. Compute a Blackman-windowed PSD on that segment after **three
   passes of 7-SD amplitude compression around the linear baseline**.
   This is the paper's artefact rejection — outliers beyond 7 σ are
   pulled smoothly toward the baseline before the FFT.
3. ``hiBand`` = total dB power in 31–47 Hz; ``loBand`` = total dB
   power in 1–30 Hz; both with 1 Hz resolution.
4. Default ``refBand = 19 dB``. If ``loBand − hiBand > 15`` (high
   band collapsed relative to low band, indicative of suppression
   onset), raise ``refBand`` to the 1–20 Hz total power, clipped to
   ``[19, 23]``.
5. ``refTrend`` = 95th percentile of ``refBand`` over the previous
   20 minutes (1200 s).

After the loop:

6. ``hiBand`` is floored at ``piecewise(refTrend, [19,23], [-16,-14])``
   to prevent the high band from going below the trend.
7. ``threshold = dBref · piecewise(refTrend, [19,23], [-0.3,-0.1])``
   where ``dBref = 20·log10(5)``.
8. ``BSRmap[n] = (loBand[n] + hiBand[n]) / 2 < threshold[n]``.
9. ``BSR`` = trailing 63-second mean of ``BSRmap`` × 100, in percent.

Returned series is at 2 Hz (one value per 0.5-s epoch).
"""
from __future__ import annotations

import numpy as np
import scipy.signal as signal

FS = 128
STRIDE_SEC = 0.5
STRIDE = int(FS * STRIDE_SEC)          # 64 samples
DBREF = 20.0 * np.log10(5.0)           # ≈ 13.98 dB (5 µV reference)
TREND_SEC = 1200                       # 20 minutes
BSR_WIN_SEC = 63                       # trailing mean window


def _n_epochs(eeg: np.ndarray) -> int:
    """Paper's ``nEpochs``: floor((L - Fs) / nStride) − 10."""
    return int(np.floor((len(eeg) - FS) / STRIDE) - 10)


def _baseline(x: np.ndarray) -> np.ndarray:
    """Linear-regression baseline of x (matches paper's ``baseline``)."""
    n = len(x)
    v = np.column_stack([np.ones(n), np.arange(1, n + 1, dtype=float)])
    coef, *_ = np.linalg.lstsq(v, x, rcond=None)
    return v @ coef


def _bound(x, lo, hi):
    """Paper's ``bound`` — clamp scalar/array to [lo, hi]."""
    return np.minimum(np.maximum(x, lo), hi)


def _piecewise(x, xp, yp):
    """Paper's ``piecewise`` — linear interp with end-clamp."""
    return np.interp(_bound(x, xp[0], xp[-1]), xp, yp)


def _psd(seg: np.ndarray, compressions: int = 3) -> np.ndarray:
    """Blackman PSD with iterative 7-σ amplitude compression.

    Mirrors paper's ``powerSpectralDensity``: for ``compressions``
    iterations, soft-saturate samples lying beyond 7 standard
    deviations from a linear baseline, then FFT the result.
    """
    x = seg.astype(float).copy()
    for _ in range(compressions):
        base = _baseline(x)
        limit = 7.0 * float(np.std(x, ddof=0))
        if limit <= 0:
            break
        dx = x - base
        ratio = np.clip(dx / limit, -1.0, 1.0)
        x = base + (1.0 - ratio ** 2) ** 2 * dx
    n = len(x)
    win = signal.windows.blackman(n)
    f = np.fft.fft(win * (x - _baseline(x)))
    return 2.0 * np.abs(f[: n // 2]) ** 2 / (n * np.sum(win ** 2))


def _band_idx(from_hz: float, to_hz: float, bin_hz: float = 1.0) -> np.ndarray:
    """Paper's ``bandRange`` in 0-indexed form (bins from_hz..to_hz inclusive)."""
    return np.arange(int(from_hz / bin_hz), int(to_hz / bin_hz) + 1)


def _total_power(psd_arr: np.ndarray, from_hz: float, to_hz: float,
                 bin_hz: float = 1.0) -> float:
    """Paper's ``totalPower``: 10·log10(Σ |X[k]|^2) over the band, NaN-as-0."""
    sub = psd_arr[_band_idx(from_hz, to_hz, bin_hz)]
    s = float(np.nansum(sub))
    if s <= 0:
        return -np.inf
    return 10.0 * np.log10(s)


def openbsr(eeg, fs: int = 128) -> np.ndarray:
    """Compute OpenBSR (Connor 2024) burst-suppression ratio at 2 Hz.

    Parameters
    ----------
    eeg : array-like, 1-D
        Raw EEG in microvolts. Must be sampled at 128 Hz.
    fs : int, default 128
        Only 128 Hz is supported (BIS standard).

    Returns
    -------
    np.ndarray of shape (N,)
        BSR percentage trajectory at 2 Hz (one value per 0.5-s epoch).
    """
    if fs != FS:
        raise ValueError(f"openbsr() requires fs=128; got {fs}.")
    eeg = np.asarray(eeg, dtype=float)

    N = _n_epochs(eeg)
    if N <= 0:
        return np.empty(0)

    lo_band  = np.full(N, np.nan)
    hi_band  = np.full(N, np.nan)
    ref_band = np.full(N, 19.0)
    ref_trend = np.full(N, 19.0)

    K_trend = int(TREND_SEC / STRIDE_SEC)  # 2400 strides

    for n in range(N):
        # Paper (MATLAB 1-indexed n): segment(eeg, n+6.5, 2, nStride)
        # Python (0-indexed n) ↔ MATLAB n+1, so:
        #   start_sample = (n + 1 + 6.5) * stride = (n + 7.5) * stride
        s = int((n + 7.5) * STRIDE)
        e = s + 2 * STRIDE
        if e > len(eeg):
            continue
        seg = eeg[s:e]
        if np.isnan(seg).any():
            continue

        psd = _psd(seg, compressions=3)
        hi_band[n] = _total_power(psd, 31, 47, bin_hz=1.0)
        lo_band[n] = _total_power(psd, 1, 30, bin_hz=1.0)

        # Default refBand = 19 dB; raise it when high band collapses
        ref_band[n] = 19.0
        if (lo_band[n] - hi_band[n]) > 15.0:
            p120 = _total_power(psd, 1, 20, bin_hz=1.0)
            ref_band[n] = _bound(p120, 19.0, 23.0)

        # 20-min trailing 95th percentile of refBand
        lo = max(0, n - K_trend + 1)
        window = ref_band[lo : n + 1]
        ref_trend[n] = float(np.percentile(window, 95))

    # Floor on hiBand: prevent it dropping below the trend limit
    floor_hi = _piecewise(ref_trend, [19.0, 23.0], [-16.0, -14.0])
    hi_band = np.maximum(hi_band, floor_hi)

    # Threshold scaled by refTrend
    threshold = DBREF * _piecewise(ref_trend, [19.0, 23.0], [-0.3, -0.1])

    # Suppression flag: average band power below threshold
    band_mean = (lo_band + hi_band) / 2.0
    BSRmap = np.where(np.isnan(band_mean), False, band_mean < threshold)

    # Trailing 63-s mean
    K = int(BSR_WIN_SEC / STRIDE_SEC)  # 126 strides
    cs = np.concatenate(([0.0], np.cumsum(BSRmap.astype(float))))
    BSR = np.empty(N)
    for i in range(N):
        j = max(0, i - K + 1)
        BSR[i] = (cs[i + 1] - cs[j]) / (i - j + 1)
    return 100.0 * BSR
