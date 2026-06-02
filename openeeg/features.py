"""openeeg.features — standalone spectral features from raw EEG.

All features run on the same 2 Hz epoch grid as :func:`openeeg.openibis`:
a 2-s Blackman-windowed FFT after linear-baseline detrending of a
0.65 Hz one-sided high-passed signal. This lets features be stacked
with openibis output for hybrid models.

References
----------
- **SEF / SEF95** — Spectral Edge Frequency, a standard depth-of-
  anaesthesia EEG marker.
- **BcSEF** (burst-compensated SEF) — Morimoto et al. 2004,
  *Anesth Analg* 98:1336–1340. Reports ``BIS ≈ 2.3·BcSEF + 12`` for
  BIS < 80, r = 0.78 on isoflurane anaesthesia.
- **BetaRatio = log10(P_30-47 / P_11-20)** — canonical BIS
  sub-parameter (Noh 2017 BIS algorithm doc; Lee 2019 data-driven
  decomposition).
- **EMG band-power proxy** — BIS Vista reports EMG in dB from a
  ~70–110 Hz band, which exceeds the 128 Hz BIS-channel Nyquist
  frequency. ``emg_estimate`` returns the dB power in 47–63 Hz, the
  highest available band on a BIS-sampled signal, as a proxy.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as signal

from openeeg.openibis import (
    FS, STRIDE, PSD_WINDOW, BIN_HZ, N_BINS,
    _psd, _band, _n_epochs,
    bsr as _bsr_func,
)


def _per_epoch_psd(eeg: np.ndarray, hp_hz: float = 0.65) -> np.ndarray:
    """Compute per-epoch PSD with the same preprocessing as openibis."""
    b, a = signal.butter(2, hp_hz / (FS / 2.0), "high")
    eeg_hp = signal.lfilter(b, a, eeg)
    N = _n_epochs(eeg)
    psd_arr = np.full((N, N_BINS), np.nan)
    for n in range(N):
        s = (n + 4) * STRIDE
        e = s + PSD_WINDOW
        if e > len(eeg_hp):
            continue
        psd_arr[n] = _psd(eeg_hp[s:e])
    return psd_arr


def sef(
    eeg,
    fs: int = 128,
    percentage: float = 95.0,
    band: tuple[float, float] = (0.5, 30.0),
) -> np.ndarray:
    """Spectral Edge Frequency at ``percentage``% within ``band``.

    Per epoch, returns the lowest frequency below which the cumulative
    PSD in ``band`` reaches ``percentage``% of the total band power.
    Output is in Hz, sampled at 2 Hz (one value per 0.5-s epoch).
    """
    if fs != FS:
        raise ValueError(f"sef() requires fs=128; got {fs}.")
    if not 0.0 < percentage < 100.0:
        raise ValueError(f"percentage must be in (0, 100); got {percentage}.")
    eeg = np.asarray(eeg, dtype=float)
    psd_arr = _per_epoch_psd(eeg)
    bins = _band(band[0], band[1])
    freqs = bins.astype(float) * BIN_HZ
    sub = psd_arr[:, bins]
    cum = np.cumsum(sub, axis=1)
    total = cum[:, -1]
    thresh = total * (percentage / 100.0)

    out = np.full(psd_arr.shape[0], np.nan)
    for n in range(psd_arr.shape[0]):
        if np.isnan(sub[n]).any() or not (total[n] > 0):
            continue
        idx = int(np.searchsorted(cum[n], thresh[n]))
        if idx < len(freqs):
            out[n] = freqs[idx]
        else:
            out[n] = freqs[-1]
    return out


def bcsef(eeg, fs: int = 128, bsr_kind: str = "quazi") -> np.ndarray:
    """Burst-compensated SEF95 — Morimoto 2004.

    ``BcSEF = SEF95 · (1 − BSR/100)``. Suppressed epochs contribute
    proportionally less, giving a single parameter that spans
    surgical → deep anaesthesia.
    """
    s = sef(eeg, fs, percentage=95.0)
    bsr_pct = _bsr_func(eeg, fs, kind=bsr_kind)
    return s * (1.0 - bsr_pct / 100.0)


def beta_ratio(eeg, fs: int = 128) -> np.ndarray:
    """``log10(P_30-47 / P_11-20)`` per epoch — light-anaesthesia marker.

    Lee 2019 finds BetaRatio dominates BIS 61–100. Morimoto 2004 fits
    ``BIS ≈ 20·BetaRatio + 95`` over BIS > 60 with r = 0.90.
    """
    if fs != FS:
        raise ValueError(f"beta_ratio() requires fs=128; got {fs}.")
    eeg = np.asarray(eeg, dtype=float)
    psd_arr = _per_epoch_psd(eeg)
    num = np.nanmean(psd_arr[:, _band(30, 47)], axis=1)
    den = np.nanmean(psd_arr[:, _band(11, 20)], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(np.maximum(num / np.maximum(den, 1e-30), 1e-30))


def band_power(
    eeg,
    fs: int = 128,
    band: tuple[float, float] = (0.5, 4.0),
) -> np.ndarray:
    """Mean dB power in ``band`` per 0.5-s epoch.

    Same PSD path as :func:`openeeg.openibis` so the value lines up
    with that algorithm's component time series.
    """
    if fs != FS:
        raise ValueError(f"band_power() requires fs=128; got {fs}.")
    eeg = np.asarray(eeg, dtype=float)
    psd_arr = _per_epoch_psd(eeg)
    p = np.nanmean(psd_arr[:, _band(band[0], band[1])], axis=1)
    return 10.0 * np.log10(np.maximum(p, 1e-30))


def spectral_entropy(
    eeg,
    fs: int = 128,
    band: tuple[float, float] = (0.5, 30.0),
) -> np.ndarray:
    """Shannon entropy of the normalised PSD over ``band``, per epoch."""
    if fs != FS:
        raise ValueError(f"spectral_entropy() requires fs=128; got {fs}.")
    eeg = np.asarray(eeg, dtype=float)
    psd_arr = _per_epoch_psd(eeg)
    sub = psd_arr[:, _band(band[0], band[1])]
    norm = sub / np.maximum(np.nansum(sub, axis=1, keepdims=True), 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(norm * np.log(np.maximum(norm, 1e-30)), axis=1)


def _higuchi_fd_window(x: np.ndarray, k_max: int = 6) -> float:
    """Higuchi 1988 fractal dimension of a 1-D series.

    Returns the slope of log(L_avg(k)) vs log(1/k) for k = 1..k_max.
    Used by :func:`fdsr` (Cusenza 2013) — see that function for the
    BSR-blended deployable wrapper.
    """
    N = len(x)
    if N < 4:
        return float("nan")
    L_k = np.zeros(k_max, dtype=float)
    for k in range(1, k_max + 1):
        L_m = []
        for m in range(k):
            n_max = (N - m - 1) // k
            if n_max < 1:
                continue
            idx = m + np.arange(n_max + 1) * k
            diff_sum = float(np.sum(np.abs(np.diff(x[idx]))))
            norm = (N - 1) / (n_max * k * k)
            L_m.append(norm * diff_sum)
        if L_m:
            L_k[k - 1] = float(np.mean(L_m))
    valid = L_k > 0
    if valid.sum() < 2:
        return float("nan")
    log_inv_k = np.log(1.0 / np.arange(1, k_max + 1)[valid])
    log_L = np.log(L_k[valid])
    slope, _ = np.polyfit(log_inv_k, log_L, 1)
    return float(slope)


def higuchi_fd(
    eeg,
    fs: int = 128,
    k_max: int = 6,
    band: tuple[float, float] = (6.0, 40.0),
    window_s: float = 10.0,
) -> np.ndarray:
    """Higuchi fractal dimension over a rolling window — Cusenza 2013.

    Returns one FD value per 0.5-s epoch (2 Hz cadence, matching other
    features). Each value is the Higuchi FD over a ``window_s``-second
    window of ``band``-bandpassed EEG. The 6–40 Hz band and 10-s
    window are Cusenza's published settings; FD is typically in
    [1.0, 2.0].
    """
    if fs != FS:
        raise ValueError(f"higuchi_fd() requires fs=128; got {fs}.")
    eeg = np.asarray(eeg, dtype=float)
    # 6–40 Hz bandpass per Cusenza
    sos = signal.butter(4, [band[0] / (fs / 2.0), band[1] / (fs / 2.0)],
                         btype="band", output="sos")
    eeg_bp = signal.sosfiltfilt(sos, eeg)
    win = int(round(window_s * fs))
    stride = STRIDE  # match other features' 0.5-s stride
    n_epochs = _n_epochs(eeg)
    out = np.full(n_epochs, np.nan)
    for n in range(n_epochs):
        s = (n + 4) * stride  # match openibis offset
        e = s + win
        if e > len(eeg_bp):
            continue
        out[n] = _higuchi_fd_window(eeg_bp[s:e], k_max=k_max)
    return out


def se50d(
    eeg,
    fs: int = 128,
    band: tuple[float, float] = (0.5, 47.0),
) -> np.ndarray:
    """SE50d — median frequency of the 1st-derivative EEG (Sleigh 2001).

    Sleigh et al. 2001 (BJA 86:50) showed the median frequency of the
    EEG's time-derivative discriminates awake-vs-asleep with ROC AUC
    equal to BIS during induction. Thresholds:

      * ``SE50d < 17 Hz`` → 100% PPV asleep
      * ``SE50d ∈ [17, 21) Hz`` → uncertain
      * ``SE50d ≥ 21 Hz`` → awake

    Returns values in Hz at 2 Hz cadence. Use
    :func:`openeeg.rules.sleigh_gate` for the threshold rule itself.
    """
    if fs != FS:
        raise ValueError(f"se50d() requires fs=128; got {fs}.")
    eeg = np.asarray(eeg, dtype=float)
    deriv = np.diff(eeg)
    # Pad to original length so epoch indexing remains consistent
    deriv = np.concatenate([[deriv[0]], deriv]) if len(deriv) > 0 else deriv
    return sef(deriv, fs=fs, percentage=50.0, band=band)


def emg_estimate(
    eeg,
    fs: int = 128,
    band: tuple[float, float] = (47.0, 63.0),
) -> np.ndarray:
    """Estimate EMG-band activity in dB from the upper EEG band.

    BIS Vista's ``BIS/EMG`` track is the dB power in a ~70–110 Hz band,
    which exceeds the Nyquist frequency of a 128 Hz BIS channel. As a
    Nyquist-bounded proxy, this returns the dB power in ``band``
    (default 47–63 Hz, just below Nyquist).

    .. warning::
       **This is a feature, not a drop-in replacement for BIS/EMG.**
       On a 100-case VitalDB validation cohort the proxy correlates
       with the published ``BIS/EMG`` track at only r ≈ 0.32 (pooled)
       / r ≈ 0.42 (per case), because 47–63 Hz on a 128 Hz BIS
       channel also captures line noise and the alpha/beta tail.
       Applying :func:`openeeg.emg_correct` with this proxy in place
       of the real ``BIS/EMG`` track does **not** reproduce the
       awake-regime correction — see ``scripts/05_features_eval.py``.

       Use it as one input feature among many (e.g. in a Phase 3
       LightGBM model), not as an automatic correction signal.
    """
    if fs != FS:
        raise ValueError(f"emg_estimate() requires fs=128; got {fs}.")
    eeg = np.asarray(eeg, dtype=float)
    psd_arr = _per_epoch_psd(eeg)
    p = np.nanmean(psd_arr[:, _band(band[0], band[1])], axis=1)
    return 10.0 * np.log10(np.maximum(p, 1e-30))
