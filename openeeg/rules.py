"""openeeg.rules — raw rule-based BIS-surrogate computations.

These are the raw 1 Hz BIS-mimic computations for each literature
rule. They are NOT public — each is exposed via
``openeeg.predict_bis(eeg, method=...)``. The canonical
:func:`openeeg.postprocess.apply_ellerkmann_and_smooth` wrapper is
applied by the dispatcher after the raw computation.

Methods (selected via ``predict_bis(method=...)``)
--------------------------------------------------
* ``"morimoto"`` — Morimoto 2004 sigmoid blend of BcSEF (deep/surgical,
  ``BIS = 2.3·BcSEF + 12.0``) and BetaRatio (awake/light, ``BIS =
  19.3·BetaRatio + 93.3``), centred at BIS ≈ 70. Each individual
  Morimoto formula saturates badly outside its valid range (BcSEF
  for BIS < 80 only, BetaRatio for BIS > 60 only), so only the
  blended form is exposed.
* ``"cusenza"`` — Cusenza 2013 FDSR (Higuchi-FD · (1 − 0.8·BSR/100),
  affine-rescaled).
* ``"sleigh"`` — Sleigh 2001 SE50d gate → Morimoto formula choice.

The :func:`sleigh_gate` helper (SE50d 3-level region classifier) is
also re-exported from the package since it is useful as a stand-alone
sleep/awake detector.
"""
from __future__ import annotations

import numpy as np

from openeeg.openibis import FS
from openeeg.openbsr import openbsr as _openbsr
from openeeg.features import (
    beta_ratio, bcsef, higuchi_fd, se50d,
)


# Cusenza FDSR → BIS [0, 100] affine constants (train-fitted, scripts/36)
_CUSENZA_A = 16.411
_CUSENZA_B = 22.132


def _raw_morimoto_combined(eeg, fs: int = 128) -> np.ndarray:
    """Raw Morimoto BcSEF/BetaRatio sigmoid blend at 1 Hz (no wrapper)."""
    bc_2hz = bcsef(eeg, fs=fs)
    br_2hz = beta_ratio(eeg, fs=fs)
    n = min(len(bc_2hz), len(br_2hz))
    bc_1hz = (2.3 * bc_2hz[:n] + 12.0)[::2]
    br_1hz = (19.3 * br_2hz[:n] + 93.3)[::2]
    n1 = min(len(bc_1hz), len(br_1hz))
    bc_1hz = bc_1hz[:n1]; br_1hz = br_1hz[:n1]
    w_awake = 1.0 / (1.0 + np.exp(-(bc_1hz - 70.0) / 5.0))
    return (1.0 - w_awake) * bc_1hz + w_awake * br_1hz


def _raw_cusenza_fdsr(eeg, fs: int = 128,
                        a: float = _CUSENZA_A, b: float = _CUSENZA_B) -> np.ndarray:
    """Raw Cusenza 2013 FDSR prediction at 1 Hz (no wrapper)."""
    fd_2hz = higuchi_fd(eeg, fs=fs)
    bsr_2hz = _openbsr(eeg, fs=fs)
    n = min(len(fd_2hz), len(bsr_2hz))
    fdsr_1hz = (fd_2hz[:n] * (1.0 - 0.8 * bsr_2hz[:n] / 100.0))[::2]
    return a * fdsr_1hz + b


def _raw_sleigh_gated_morimoto(eeg, fs: int = 128) -> np.ndarray:
    """Raw Sleigh-gated Morimoto blend at 1 Hz (no wrapper)."""
    g = sleigh_gate(eeg, fs=fs)
    bc_2hz = bcsef(eeg, fs=fs)
    br_2hz = beta_ratio(eeg, fs=fs)
    n = min(len(bc_2hz), len(br_2hz))
    bc_1hz = (2.3 * bc_2hz[:n] + 12.0)[::2]
    br_1hz = (19.3 * br_2hz[:n] + 93.3)[::2]
    n1 = min(len(g), len(bc_1hz), len(br_1hz))
    g = g[:n1]; bc_1hz = bc_1hz[:n1]; br_1hz = br_1hz[:n1]
    w_awake = np.where(g <= -1.0 + 1e-6, 1.0,
                        np.where(g >= 1.0 - 1e-6, 0.0, 0.5))
    w_awake = np.where(np.isnan(g), 0.5, w_awake)
    return (1.0 - w_awake) * bc_1hz + w_awake * br_1hz


def sleigh_gate(eeg, fs: int = 128) -> np.ndarray:
    """Sleigh 2001 SE50d 3-level gate at 1 Hz.

    Returns an array taking three values:

      * ``+1``  SE50d < 17 Hz  → asleep (100 % PPV)
      * ``0``  17 ≤ SE50d < 21 → uncertain
      * ``−1`` SE50d ≥ 21 Hz   → awake

    Intended as a region-selector / sleep–awake detector for ensemble
    or piecewise pipelines — NOT a BIS surrogate. (For the BIS-surrogate
    use of the Sleigh gate combined with Morimoto's BcSEF/BetaRatio
    formulas, see ``predict_bis(eeg, method="sleigh")``.)
    """
    if fs != FS:
        raise ValueError(f"sleigh_gate() requires fs=128; got {fs}.")
    s = se50d(eeg, fs=fs)
    s_1hz = s[::2]
    gate = np.where(s_1hz < 17.0, 1.0,
                     np.where(s_1hz < 21.0, 0.0, -1.0))
    gate = np.where(np.isnan(s_1hz), np.nan, gate)
    return gate
