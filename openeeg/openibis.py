"""openibis — BIS-mimic from raw EEG.

Reimplemented from the methodology described in:

    Connor, C. W. (2022). Open Reimplementation of the BIS Algorithms
    for Depth of Anesthesia. Anesthesia & Analgesia, 135(4), 855-864.
    PMC9481655. https://pubmed.ncbi.nlm.nih.gov/35767469/

Deep-regime variant additionally uses the empirical BSR fit from:

    Ellerkmann, R. K., et al. (2004). The Entropy Module and Bispectral
    Index as Guidance for Propofol-Remifentanil Anaesthesia.
    Anesthesia & Analgesia, 98(5), 1275-1281.

No code was copied verbatim from either source; this implementation
follows the *described* algorithm in each paper.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as signal
from scipy import stats

FS = 128
STRIDE_SEC = 0.5
STRIDE = int(FS * STRIDE_SEC)  # 64 samples
PSD_WINDOW = 4 * STRIDE         # 2 s = 256 samples → Blackman PSD
BIN_HZ = FS / PSD_WINDOW        # 0.5 Hz bins
N_BINS = PSD_WINDOW // 2        # 128 positive-frequency bins, up to 63.5 Hz


# ----- low-level helpers -----------------------------------------------------

def _n_epochs(eeg: np.ndarray) -> int:
    """Number of 0.5-s epochs available, leaving room for the longest window."""
    return int(np.floor((len(eeg) - FS) / STRIDE) - 10)


def _baseline(x: np.ndarray) -> np.ndarray:
    """Linear regression baseline (intercept + slope) of x."""
    n = len(x)
    A = np.column_stack([np.ones(n), np.arange(1, n + 1)])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return A @ coef


def _psd(x: np.ndarray) -> np.ndarray:
    """Blackman-windowed PSD of (x − linear baseline), one-sided."""
    w = signal.windows.blackman(len(x))
    f = np.fft.fft(w * (x - _baseline(x)))
    n = len(x)
    return 2.0 * np.abs(f[: n // 2]) ** 2 / (n * np.sum(w ** 2))


def _band(lo: float, hi: float) -> np.ndarray:
    """Indices of 0.5-Hz PSD bins covering [lo, hi] Hz inclusive."""
    return np.arange(int(lo / BIN_HZ), int(hi / BIN_HZ) + 1)


def _pinterp(x, xp, yp) -> np.ndarray:
    """Piecewise linear interpolation, clamped at endpoints."""
    return np.interp(np.clip(x, xp[0], xp[-1]), xp, yp)


def _scurve(x, Eo, Emax, x50, xwidth):
    return Eo - Emax / (1.0 + np.exp((x - x50) / xwidth))


def _prctmean(x: np.ndarray, lo_pct: float, hi_pct: float) -> float:
    """Mean of values whose magnitude lies between two percentiles."""
    valid = x[~np.isnan(x)]
    if len(valid) == 0:
        return np.nan
    lo, hi = np.percentile(valid, [lo_pct, hi_pct])
    sub = valid[(valid >= lo) & (valid <= hi)]
    return float(np.mean(sub)) if len(sub) > 0 else np.nan


def _mean_band_power_db(psd_arr: np.ndarray, lo: float, hi: float) -> float:
    v = psd_arr[:, _band(lo, hi)]
    valid = v[~np.isnan(v)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(10.0 * np.log10(np.maximum(valid, 1e-30))))


# ----- burst-suppression detector (Connor 2023) -----------------------------

def _bsr(eeg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Burst-suppression detector per Connor 2023.

    For each 0.5-s epoch, examine a 1-second window starting at sample
    ``(n + 6.5) * STRIDE``. The window is marked suppressed if every
    sample stays within ±5 µV of its linear regression baseline.

    Returns
    -------
    BSRmap : (N,) bool — per-epoch suppression flag
    BSR    : (N,) float — trailing 63-second mean × 100, in percent
    """
    N = _n_epochs(eeg)
    BSRmap = np.zeros(N, dtype=bool)
    win = 2 * STRIDE  # 1 second

    for n in range(N):
        s = int((n + 6.5) * STRIDE)
        e = s + win
        if e > len(eeg):
            continue
        seg = eeg[s:e]
        BSRmap[n] = bool(np.all(np.abs(seg - _baseline(seg)) <= 5.0))

    # Trailing 63-second mean = 126 strides
    K = int(63.0 / STRIDE_SEC)
    cs = np.concatenate([[0.0], np.cumsum(BSRmap.astype(float))])
    BSR = np.empty(N)
    for i in range(N):
        j = max(0, i - K + 1)
        BSR[i] = (cs[i + 1] - cs[j]) / (i - j + 1)
    return BSRmap, 100.0 * BSR


# ----- sawtooth (K-complex / artifact) detector -----------------------------

def _sawtooth(eeg_seg: np.ndarray) -> bool:
    """Detect a sawtooth / K-complex artifact in a 2-second EEG segment.

    Cross-correlates the segment against a normalized 5-sample upramp
    template, in both directions. Returns True if the maximum normalised
    correlation exceeds 0.63 at every sliding-window position.
    """
    saw = np.concatenate([np.zeros(STRIDE - 5), np.arange(1, 6, dtype=float)])
    saw = (saw - saw.mean()) / saw.std(ddof=0)
    W = len(saw)
    N = len(eeg_seg) - W + 1
    if N <= 0:
        return False

    cs = np.concatenate([[0.0], np.cumsum(eeg_seg)])
    cs2 = np.concatenate([[0.0], np.cumsum(eeg_seg ** 2)])
    sums = cs[W:W + N] - cs[:N]
    sum2 = cs2[W:W + N] - cs2[:N]
    var = (sum2 - sums ** 2 / W) / (W - 1)

    c1 = np.convolve(eeg_seg, saw[::-1], mode="valid")
    c2 = np.convolve(eeg_seg, saw, mode="valid")
    m = np.column_stack([c1, c2]) ** 2 / (W ** 2)

    r = np.arange(N - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(var[r] > 10, m[r, 0] / var[r], 0.0)
        t2 = np.where(var[r] > 10, m[r, 1] / var[r], 0.0)
    return bool(np.all(np.maximum(t1, t2) > 0.63))


# ----- three EEG components --------------------------------------------------

def _components(eeg: np.ndarray, BSRmap: np.ndarray) -> np.ndarray:
    """Compute (N, 3) component matrix.

    C1 = mean power [30,47] Hz − midband     (sedation axis)
    C2 = trimmean(10·log10(vhighPC / wholePC), 50%)   (general axis)
    C3 = mean power [0.5,4] Hz − midband     (mixer-weight basis)

    The "power concentration" PC₍a,b₎ averages √(P_n · P_{n−1}) over the
    band, which approximates the joint power at adjacent frequency pairs
    used in the bispectrum.
    """
    N = _n_epochs(eeg)

    # 0.65 Hz one-sided high-pass for the PSD path (paper: causal filter)
    b, a = signal.butter(2, 0.65 / (FS / 2.0), "high")
    eeg_hp = signal.lfilter(b, a, eeg)

    # Sawtooth-triggered low-frequency suppression filter on the PSD axis
    freqs = np.arange(N_BINS) * BIN_HZ
    sup_filter = _pinterp(freqs, [0, 3, 6], [0, 0.25, 1]) ** 2

    psd_arr = np.full((N, N_BINS), np.nan)
    components = np.full((N, 3), np.nan)

    for n in range(N):
        # PSD only if none of the last 4 epochs (n-3..n) were burst-suppressed.
        if n >= 3 and not np.any(BSRmap[n - 3 : n + 1]):
            s = (n + 4) * STRIDE
            e = s + PSD_WINDOW
            if e <= len(eeg_hp):
                p = _psd(eeg_hp[s:e])
                if _sawtooth(eeg[s:e]):
                    p = sup_filter * p
                psd_arr[n] = p

        # Trailing 30-second window = 60 strides (inclusive of n)
        t30 = slice(max(0, n - 59), n + 1)
        psd_t30 = psd_arr[t30]

        # vhighPC and wholePC: √(P · P shifted by 1 bin)
        vhigh = np.sqrt(np.nanmean(
            psd_t30[:, _band(39.5, 46.5)] * psd_t30[:, _band(40.0, 47.0)], axis=1))
        whole = np.sqrt(np.nanmean(
            psd_t30[:, _band(0.5, 46.5)] * psd_t30[:, _band(1.0, 47.0)], axis=1))

        # midband: dB mean over [11,20] Hz, then "upper-half" percentile mean
        mid = psd_t30[:, _band(11, 20)]
        with np.errstate(divide="ignore", invalid="ignore"):
            mid_db_per_bin = np.nanmean(10.0 * np.log10(np.maximum(mid, 1e-30)), axis=0)
            mid_bp = _prctmean(mid_db_per_bin, 50, 100)

        components[n, 0] = _mean_band_power_db(psd_t30, 30, 47) - mid_bp

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = 10.0 * np.log10(np.maximum(vhigh / whole, 1e-30))
            valid = ratio[np.isfinite(ratio)]
            if len(valid) > 0:
                # trimmean(., 50%) trims 25% from each tail
                components[n, 1] = float(stats.trim_mean(valid, 0.25))

        components[n, 2] = _mean_band_power_db(psd_t30, 0.5, 4) - mid_bp

    return components


# ----- mixer variants --------------------------------------------------------

def _eeg_score(components: np.ndarray) -> np.ndarray:
    """EEG (non-suppression) score branch, shared by paper and Ellerkmann."""
    sedation = _scurve(components[:, 0], 104.4, 49.4, -13.9, 5.29)

    C2 = components[:, 1]
    pw = np.interp(C2, [-60.89, -30.0], [-40.0, 43.1])
    sc = _scurve(C2, 61.3, 72.6, -24.0, 3.55) * (C2 >= -30.0)
    general = pw + sc

    gen_weight = _pinterp(components[:, 2], [0, 5], [0.5, 1.0]) * (general < sedation)
    x = sedation * (1.0 - gen_weight) + general * gen_weight
    return _pinterp(x, [-40, 10, 97, 110], [0, 10, 97, 100])


def _mixer_paper(components: np.ndarray, BSR: np.ndarray) -> np.ndarray:
    """openibis mixer per Connor 2023 — deep-regime is BSR-linear: 50 − BSR/2."""
    eeg_score = _eeg_score(components)
    bsr_score = 50.0 - BSR / 2.0
    bsr_weight = _pinterp(BSR, [10, 50], [0.0, 1.0])
    return eeg_score * (1.0 - bsr_weight) + bsr_score * bsr_weight


def _mixer_ellerkmann(components: np.ndarray, BSR: np.ndarray) -> np.ndarray:
    """openibis mixer with the Ellerkmann 2004 empirical deep-regime fit.

    Ellerkmann reported BIS = 44.1 − BSR/2.25 (R² = 0.99) for BSR > 40%.
    Below 40% the deep-regime branch falls back to the paper linear
    50 − BSR/2 (and its weight is small anyway).
    """
    eeg_score = _eeg_score(components)
    bsr_score = np.where(BSR >= 40.0, 44.1 - BSR / 2.25, 50.0 - BSR / 2.0)
    bsr_weight = _pinterp(BSR, [10, 50], [0.0, 1.0])
    return eeg_score * (1.0 - bsr_weight) + bsr_score * bsr_weight


# ----- public entry point ----------------------------------------------------

def openibis(eeg, fs: int = 128, deep: str = "paper") -> np.ndarray:
    """Compute openibis BIS-mimic from raw EEG.

    Parameters
    ----------
    eeg : array-like, 1-D
        Raw EEG in microvolts. Must be sampled at 128 Hz (BIS standard).
    fs : int, default 128
        Sampling frequency, in Hz. Only 128 is currently supported.
    deep : {"paper", "ellerkmann"}, default "paper"
        Deep-anaesthesia (burst-suppression) scoring rule:
          * ``"paper"``      — Connor 2023, score = 50 − BSR/2.
          * ``"ellerkmann"`` — Ellerkmann 2004 fit ``44.1 − BSR/2.25``
            engaged for BSR ≥ 40 %, otherwise the paper formula.

    Returns
    -------
    np.ndarray of shape (N,)
        BIS-like score at 2 Hz (one value per 0.5-s epoch).
    """
    eeg = np.asarray(eeg, dtype=float)
    if fs != FS:
        raise ValueError(f"openibis is specified at fs=128 Hz; got fs={fs}.")
    if deep not in {"paper", "ellerkmann"}:
        raise ValueError(f"deep must be 'paper' or 'ellerkmann'; got {deep!r}.")

    BSRmap, BSR = _bsr(eeg)
    comp = _components(eeg, BSRmap)
    return _mixer_paper(comp, BSR) if deep == "paper" else _mixer_ellerkmann(comp, BSR)


def bsr(eeg, fs: int = 128) -> np.ndarray:
    """Compute Connor-2023 burst-suppression ratio (percent) at 2 Hz.

    Convenience wrapper that returns only the BSR trajectory.
    """
    eeg = np.asarray(eeg, dtype=float)
    if fs != FS:
        raise ValueError(f"bsr() requires fs=128; got {fs}.")
    _, BSR = _bsr(eeg)
    return BSR
