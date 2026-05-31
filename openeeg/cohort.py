"""VitalDB BIS cohort utilities — case listing, deterministic split, loader."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

import vitaldb

FS = 128

# Tracks every BIS-monitored VitalDB case provides.
BIS_TRACKS = ("BIS/EEG1_WAV", "BIS/BIS", "BIS/SQI", "BIS/SR", "BIS/EMG")


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

    If ``cache_dir`` is given, the .vital file is fetched once via
    HTTP and cached on disk; subsequent calls read from disk only.
    """
    vital_path: str
    if cache_dir is not None:
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
        rest = vf.to_numpy(["BIS/BIS", "BIS/SQI", "BIS/SR", "BIS/EMG"], 1.0)
        if rest.shape[1] != 4 or len(eeg) < FS * 60:
            return None
    except Exception:
        return None

    return {
        "caseid": caseid,
        "fs": FS,
        "eeg": eeg,
        "bis": rest[:, 0],
        "sqi": rest[:, 1],
        "sr":  rest[:, 2],
        "emg": rest[:, 3],
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
