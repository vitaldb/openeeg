"""Validate openibis variants on the bundled 1.vital case.

Computes:
  * openibis(deep="paper")
  * openibis(deep="ellerkmann")
  * Connor 2023 BSR
  * OpenBSR 2025 (best-effort prose port)

and compares all four against the BIS Vista's actual BIS / SR tracks.
Filters to SQI ≥ 80 epochs only. Reports MAE, Pearson r, and Lin's
concordance for each BIS variant, and MAE / r for both BSR variants.

Usage::

    python scripts/01_validate_1vital.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make `openeeg` importable when running this script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg import openibis, openbsr
from openeeg.openibis import bsr as openibis_bsr

import vitaldb
import matplotlib.pyplot as plt


VITAL_FILE = Path(__file__).resolve().parents[1] / "1.vital"
FS = 128
SQI_THRESH = 80


def lin_concordance(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    return 2.0 * cov / (vx + vy + (mx - my) ** 2)


def preprocess_eeg(raw: np.ndarray) -> np.ndarray:
    """Interpolate short NaN gaps, zero-fill the rest, baseline-correct."""
    eeg = raw.copy()
    nan_mask = np.isnan(eeg)
    max_gap = int(FS * 0.05)  # ≤50 ms gaps are interpolated
    diff = np.diff(np.concatenate(([0], nan_mask.astype(int), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if e - s <= max_gap and s > 0 and e < len(eeg):
            eeg[s:e] = np.linspace(eeg[s - 1], eeg[e], e - s, endpoint=False)
    eeg[np.isnan(eeg)] = 0.0
    eeg = eeg - np.median(eeg[~nan_mask])
    return eeg


def main() -> None:
    print(f"Reading {VITAL_FILE.name} ...")
    vf = vitaldb.VitalFile(str(VITAL_FILE))
    eeg_raw = vf.to_numpy(["BIS/EEG1_WAV"], 1.0 / FS).flatten()
    bis_actual = vf.to_numpy(["BIS/BIS"], 1.0).flatten()
    sqi_actual = vf.to_numpy(["BIS/SQI"], 1.0).flatten()
    sr_actual = vf.to_numpy(["BIS/SR"], 1.0).flatten()

    eeg = preprocess_eeg(eeg_raw)
    print(f"  EEG samples: {len(eeg):,}  duration: {len(eeg)/FS/60:.1f} min")

    print("Running openibis(deep='paper') ...")
    bis_paper = openibis(eeg, deep="paper")
    print("Running openibis(deep='ellerkmann') ...")
    bis_eller = openibis(eeg, deep="ellerkmann")
    print("Running Connor 2023 BSR ...")
    bsr_c23 = openibis_bsr(eeg)
    print("Running OpenBSR 2025 ...")
    bsr_c25 = openbsr(eeg)

    # Align everything to 1 Hz (BIS Vista output rate)
    bis_paper_1hz = bis_paper[::2]
    bis_eller_1hz = bis_eller[::2]
    bsr_c23_1hz = bsr_c23[::2]
    bsr_c25_1hz = bsr_c25[::2]

    n = min(len(bis_actual), len(bis_paper_1hz), len(sqi_actual), len(sr_actual))
    bis_actual = bis_actual[:n]
    sqi_actual = sqi_actual[:n]
    sr_actual = sr_actual[:n]
    bis_paper_1hz = bis_paper_1hz[:n]
    bis_eller_1hz = bis_eller_1hz[:n]
    bsr_c23_1hz = bsr_c23_1hz[:n]
    bsr_c25_1hz = bsr_c25_1hz[:n]

    # BIS metrics (SQI ≥ 80)
    print("\n=== BIS comparison (SQI >= {}) ===".format(SQI_THRESH))
    valid = (
        ~np.isnan(bis_actual)
        & ~np.isnan(bis_paper_1hz)
        & ~np.isnan(bis_eller_1hz)
        & ~np.isnan(sqi_actual)
        & (sqi_actual >= SQI_THRESH)
    )
    print(f"  N (valid epochs): {int(valid.sum()):,} / {n:,}")

    rows = []
    for name, pred in [
        ("openibis (paper)", bis_paper_1hz),
        ("openibis (ellerkmann)", bis_eller_1hz),
    ]:
        a = bis_actual[valid]
        p = pred[valid]
        mae = float(np.mean(np.abs(p - a)))
        r = float(np.corrcoef(a, p)[0, 1])
        rc = lin_concordance(a, p)
        rows.append((name, mae, r, rc))
        print(f"  {name:30s}  MAE={mae:5.2f}   r={r:.3f}   Lin_rc={rc:.3f}")

    # BSR metrics: compare against BIS Vista SR (which is its commercial BSR)
    print("\n=== BSR comparison (SQI >= {}) ===".format(SQI_THRESH))
    valid_bsr = (
        ~np.isnan(sr_actual)
        & ~np.isnan(bsr_c23_1hz)
        & ~np.isnan(bsr_c25_1hz)
        & ~np.isnan(sqi_actual)
        & (sqi_actual >= SQI_THRESH)
    )
    print(f"  N (valid epochs): {int(valid_bsr.sum()):,}")
    for name, pred in [
        ("Connor 2023 BSR", bsr_c23_1hz),
        ("OpenBSR 2025", bsr_c25_1hz),
    ]:
        a = sr_actual[valid_bsr]
        p = pred[valid_bsr]
        mae = float(np.mean(np.abs(p - a)))
        r = float(np.corrcoef(a, p)[0, 1]) if a.std() > 0 and p.std() > 0 else float("nan")
        rc = lin_concordance(a, p) if a.std() > 0 and p.std() > 0 else float("nan")
        print(f"  {name:30s}  MAE={mae:5.2f}   r={r:.3f}   Lin_rc={rc:.3f}")

    # Plot
    t = np.arange(n)
    low_sqi = np.isnan(sqi_actual) | (sqi_actual < SQI_THRESH)
    shade = []
    in_low = False
    start = 0
    for i in range(n):
        if low_sqi[i] and not in_low:
            start, in_low = i, True
        elif not low_sqi[i] and in_low:
            shade.append((start, i))
            in_low = False
    if in_low:
        shade.append((start, n))

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.5, 1]})

    t_eeg = np.arange(len(eeg)) / FS
    axes[0].plot(t_eeg[::FS], eeg[::FS], lw=0.3, color="steelblue")
    for s0, s1 in shade:
        axes[0].axvspan(s0, s1, color="gray", alpha=0.2)
    axes[0].set_ylabel("EEG (µV)")
    axes[0].set_title("Preprocessed EEG (gray = SQI < 80)")
    axes[0].set_ylim(-60, 60)

    axes[1].plot(t, bis_actual, lw=1.0, color="tab:blue", alpha=0.7, label="Actual BIS")
    axes[1].plot(t, bis_paper_1hz, lw=0.9, color="tab:red", alpha=0.7, label="openibis (paper)")
    axes[1].plot(t, bis_eller_1hz, lw=0.9, color="tab:green", alpha=0.7, label="openibis (ellerkmann)")
    for s0, s1 in shade:
        axes[1].axvspan(s0, s1, color="gray", alpha=0.2)
    axes[1].set_ylabel("BIS")
    axes[1].set_title("Actual BIS vs openibis variants")
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, sr_actual, lw=1.0, color="tab:blue", alpha=0.7, label="Actual SR")
    axes[2].plot(t, bsr_c23_1hz, lw=0.9, color="tab:red", alpha=0.7, label="Connor 2023 BSR")
    axes[2].plot(t, bsr_c25_1hz, lw=0.9, color="tab:green", alpha=0.7, label="OpenBSR 2025")
    for s0, s1 in shade:
        axes[2].axvspan(s0, s1, color="gray", alpha=0.2)
    axes[2].set_ylabel("BSR / SR (%)")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_title("Suppression Ratio comparison")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    out = Path(__file__).resolve().parents[1] / "validate_1vital.png"
    plt.savefig(str(out), dpi=140)
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
