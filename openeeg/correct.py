"""Post-processing corrections for openibis-style BIS predictions.

The functions here take an already-computed BIS-mimic trajectory and
adjust it using an auxiliary signal. They are deliberately *not* part
of :func:`openeeg.openibis` so the paper-faithful implementation stays
pure; combine them only when the auxiliary signal is available.
"""
from __future__ import annotations

import numpy as np

# EMG-correction coefficient empirically derived from a 100-case
# subset of the VitalDB BIS cohort (val fold, caseid%10==8) by
# regressing the per-epoch residual ``pred − actual`` of
# ``openibis(bsr='quazi', deep='paper')`` against ``max(EMG-34, 0)``
# (Lee 2019 threshold, BIS/EMG track in dB). See
# ``scripts/04_emg_correction_analysis.py``.
EMG_CORRECTION_SLOPE = 0.54     # BIS points per dB EMG above threshold
EMG_CORRECTION_THRESHOLD = 34.0  # dB, per Lee 2019


def emg_correct(
    bis: np.ndarray,
    emg_db: np.ndarray,
    *,
    slope: float = EMG_CORRECTION_SLOPE,
    threshold: float = EMG_CORRECTION_THRESHOLD,
) -> np.ndarray:
    """Subtract an EMG-amplitude-driven correction from a BIS trajectory.

    openibis (and the underlying BIS algorithm family) takes its
    high-gamma activity from the 30–47 Hz band, which on a BIS-sensor
    EEG channel overlaps facial/jaw EMG. When the patient is awake or
    in light sedation, EMG inflates the high-gamma channel and pushes
    openibis's output upward. On the VitalDB cohort the inflation is
    approximately linear in dB above 34 dB, with slope ~0.54 BIS-point
    per dB.

    Parameters
    ----------
    bis : array-like, 1-D
        BIS-mimic trajectory (e.g. from :func:`openeeg.openibis`).
    emg_db : array-like, 1-D
        EMG amplitude in dB, sampled at the same rate as ``bis``.
        Typically the ``BIS/EMG`` track from VitalDB.
    slope : float, default 0.54
        Correction slope in BIS-points / dB.
    threshold : float, default 34.0
        dB threshold below which no correction is applied. Lee 2019
        identified 34.2 dB as the EMG decision-tree split for BIS.

    Returns
    -------
    np.ndarray
        Corrected BIS trajectory, clipped to [0, 100]. Same shape as
        ``bis``. NaN inputs in either array are preserved (correction
        is zero where either input is NaN).
    """
    bis = np.asarray(bis, dtype=float)
    emg_db = np.asarray(emg_db, dtype=float)
    if bis.shape != emg_db.shape:
        raise ValueError(
            f"bis and emg_db must have the same shape; got {bis.shape} vs {emg_db.shape}."
        )
    excess = np.where(np.isnan(emg_db), 0.0, np.maximum(emg_db - threshold, 0.0))
    corrected = bis - slope * excess
    return np.clip(corrected, 0.0, 100.0)
