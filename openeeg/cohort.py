"""VitalDB BIS cohort utilities — case listing, deterministic split, loader."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

import vitaldb

FS = 128

# Tracks every BIS-monitored VitalDB case provides.
BIS_TRACKS = (
    "BIS/EEG1_WAV", "BIS/BIS", "BIS/SQI",
    "BIS/SR",      # BIS Vista's published Suppression Ratio (= Lee 2019 "BSR")
    "BIS/EMG",     # BIS Vista's EMG amplitude in dB
    "BIS/SEF",     # BIS Vista's Spectral Edge Frequency 95%
    "BIS/TOTPOW",  # BIS Vista's total EEG power in dB
)

# If you have a local copy of the VitalDB Open Dataset .vital files,
# set VITALDB_DATA_ROOT to that directory and load_case() will read
# directly from disk instead of fetching over HTTPS. The directory
# should contain "<caseid>.vital" files at its top level (matching
# the layout of <https://api.vitaldb.net/<version>/>).
_DEFAULT_DATA_ROOTS = (
    "D:/vitaldb_open_1.0.1",
    "D:\\vitaldb_open_1.0.1",
)


def _resolve_local_vital(caseid: int) -> Optional[Path]:
    """Return a local .vital path for the case if available, else None."""
    env_root = os.environ.get("VITALDB_DATA_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    for r in _DEFAULT_DATA_ROOTS:
        candidates.append(Path(r))
    for root in candidates:
        if not root.exists():
            continue
        p = root / f"{caseid}.vital"
        if p.is_file():
            return p
    return None


def caseids_bis() -> list[int]:
    """Sorted list of VitalDB caseIDs known to contain a BIS module."""
    return sorted(vitaldb.caseids_bis)


_FOLDS = {"train": frozenset(range(8)), "val": frozenset({8}), "test": frozenset({9})}


def split(caseids: Iterable[int], fold: str) -> list[int]:
    """Deterministic case-level split using ``caseid % 10``.

    train: residues 0–7 (80%), val: 8 (10%), test: 9 (10%).
    """
    if fold not in _FOLDS:
        raise ValueError(f"fold must be one of {list(_FOLDS)}; got {fold!r}.")
    keep = _FOLDS[fold]
    return [c for c in caseids if (c % 10) in keep]


def load_case(
    caseid: int,
    cache_dir: Optional[Path] = None,
) -> Optional[dict]:
    """Load BIS-relevant tracks for a single case.

    Returns
    -------
    dict with keys ``eeg`` (128 Hz), ``bis`` / ``sqi`` / ``sr`` / ``emg``
    (each 1 Hz), or ``None`` if the case cannot be loaded.

    Resolution order for the underlying ``<caseid>.vital`` file:
      1. ``VITALDB_DATA_ROOT`` env var, then ``D:/vitaldb_open_1.0.1``
         (local Open Dataset mirror — fastest).
      2. ``cache_dir`` if given (HTTP-fetched on first miss).
      3. Direct HTTP fetch from the VitalDB API as a last resort.
    """
    local_path = _resolve_local_vital(caseid)
    if local_path is not None:
        vital_path = str(local_path)
    elif cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        local = cache_dir / f"{caseid}.vital"
        if not local.exists():
            url = f"https://api.vitaldb.net/{vitaldb.dataset.DATASET_VERSION}/{caseid}.vital"
            try:
                import urllib.request
                urllib.request.urlretrieve(url, str(local))
            except Exception:
                return None
        vital_path = str(local)
    else:
        vital_path = f"https://api.vitaldb.net/{vitaldb.dataset.DATASET_VERSION}/{caseid}.vital"

    try:
        vf = vitaldb.VitalFile(vital_path, list(BIS_TRACKS))
    except Exception:
        return None

    try:
        eeg = vf.to_numpy(["BIS/EEG1_WAV"], 1.0 / FS).flatten()
        rest = vf.to_numpy(
            ["BIS/BIS", "BIS/SQI", "BIS/SR", "BIS/EMG", "BIS/SEF", "BIS/TOTPOW"], 1.0)
        if rest.shape[1] != 6 or len(eeg) < FS * 60:
            return None
    except Exception:
        return None

    return {
        "caseid": caseid,
        "fs": FS,
        "eeg": eeg,
        "bis":    rest[:, 0],
        "sqi":    rest[:, 1],
        "sr":     rest[:, 2],
        "emg":    rest[:, 3],
        "sef":    rest[:, 4],
        "totpow": rest[:, 5],
    }


def preprocess_eeg(raw: np.ndarray, fs: int = FS) -> np.ndarray:
    """Interpolate short NaN gaps, zero-fill the rest, median-detrend."""
    eeg = raw.copy()
    nan_mask = np.isnan(eeg)
    max_gap = int(fs * 0.05)
    diff = np.diff(np.concatenate(([0], nan_mask.astype(int), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if e - s <= max_gap and s > 0 and e < len(eeg):
            eeg[s:e] = np.linspace(eeg[s - 1], eeg[e], e - s, endpoint=False)
    eeg[np.isnan(eeg)] = 0.0
    if (~nan_mask).any():
        eeg = eeg - np.median(eeg[~nan_mask])
    return eeg
