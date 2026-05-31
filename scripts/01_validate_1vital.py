"""Validate openibis variants on the bundled 1.vital case.

Computes the full bsr × deep grid plus standalone BSR detectors, and
compares each against the BIS Vista's actual ``BIS/BIS`` and ``BIS/SR``
tracks (SQI ≥ 80 epochs only).

Usage::

    python scripts/01_validate_1vital.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg import openibis, openbsr
from openeeg.openibis import bsr as openibis_bsr

import vitaldb
import matplotlib.pyplot as plt


VITAL_FILE = Path(__file__).resolve().parents[1] / "1.vital"
FS = 128
SQI_THRESH = 80


def lin_concordance(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    return 2.0 * cov / (vx + vy + (mx - my) ** 2)


def preprocess_eeg(raw: np.ndarray) -> np.ndarray:
    eeg = raw.copy()
    nan_mask = np.isnan(eeg)
    max_gap = int(FS * 0.05)
    diff = np.diff(np.concatenate(([0], nan_mask.astype(int), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if e - s <= max_gap and s > 0 and e < len(eeg):
            eeg[s:e] = np.linspace(eeg[s - 1], eeg[e], e - s, endpoint=False)
    eeg[np.isnan(eeg)] = 0.0
    eeg = eeg - np.median(eeg[~nan_mask])
    return eeg


def metrics(actual: np.ndarray, predicted: np.ndarray, valid: np.ndarray):
    a, p = actual[valid], predicted[valid]
    mae = float(np.mean(np.abs(p - a)))
    if a.std() > 0 and p.std() > 0:
        r = float(np.corrcoef(a, p)[0, 1])
        rc = lin_concordance(a, p)
    else:
        r, rc = float("nan"), float("nan")
    return mae, r, rc


def main() -> None:
    print(f"Reading {VITAL_FILE.name} ...")
    vf = vitaldb.VitalFile(str(VITAL_FILE))
    eeg_raw = vf.to_numpy(["BIS/EEG1_WAV"], 1.0 / FS).flatten()
    bis_actual = vf.to_numpy(["BIS/BIS"], 1.0).flatten()
    sqi_actual = vf.to_numpy(["BIS/SQI"], 1.0).flatten()
    sr_actual = vf.to_numpy(["BIS/SR"], 1.0).flatten()

    eeg = preprocess_eeg(eeg_raw)
    print(f"  EEG samples: {len(eeg):,}  duration: {len(eeg)/FS/60:.1f} min")

    bis_grid = {}
    for bsr_kind in ("paper", "quazi"):
        for deep in ("paper", "ellerkmann"):
            print(f"  openibis(bsr={bsr_kind!r}, deep={deep!r}) ...")
            bis_grid[(bsr_kind, deep)] = openibis(eeg, bsr=bsr_kind, deep=deep)

    print("  bsr(kind='paper') ...")
    bsr_paper = openibis_bsr(eeg, kind="paper")
    print("  bsr(kind='quazi') ...")
    bsr_quazi = openibis_bsr(eeg, kind="quazi")
    print("  openbsr() ...")
    bsr_openbsr = openbsr(eeg)

    bis_1hz = {k: v[::2] for k, v in bis_grid.items()}
    bsr_paper_1hz = bsr_paper[::2]
    bsr_quazi_1hz = bsr_quazi[::2]
    bsr_openbsr_1hz = bsr_openbsr[::2]

    n = min(len(bis_actual), *[len(v) for v in bis_1hz.values()],
            len(sqi_actual), len(sr_actual))
    bis_actual = bis_actual[:n]
    sqi_actual = sqi_actual[:n]
    sr_actual = sr_actual[:n]
    bis_1hz = {k: v[:n] for k, v in bis_1hz.items()}
    bsr_paper_1hz = bsr_paper_1hz[:n]
    bsr_quazi_1hz = bsr_quazi_1hz[:n]
    bsr_openbsr_1hz = bsr_openbsr_1hz[:n]

    valid_bis = (
        ~np.isnan(bis_actual)
        & ~np.isnan(sqi_actual)
        & (sqi_actual >= SQI_THRESH)
        & np.all([~np.isnan(v) for v in bis_1hz.values()], axis=0)
    )
    print(f"\n=== BIS comparison (SQI>={SQI_THRESH}, N={int(valid_bis.sum()):,}) ===")
    print(f"{'variant':<32s}  {'MAE':>6s}  {'r':>6s}  {'Lin_rc':>6s}")
    for (bsr_kind, deep), pred in bis_1hz.items():
        mae, r, rc = metrics(bis_actual, pred, valid_bis)
        label = f"openibis(bsr={bsr_kind!r}, deep={deep!r})"
        print(f"{label:<32s}  {mae:6.2f}  {r:6.3f}  {rc:6.3f}")

    valid_bsr = (
        ~np.isnan(sr_actual)
        & ~np.isnan(sqi_actual)
        & (sqi_actual >= SQI_THRESH)
        & ~np.isnan(bsr_paper_1hz)
        & ~np.isnan(bsr_quazi_1hz)
        & ~np.isnan(bsr_openbsr_1hz)
    )
    print(f"\n=== BSR comparison (SQI>={SQI_THRESH}, N={int(valid_bsr.sum()):,}) ===")
    print(f"{'variant':<32s}  {'MAE':>6s}  {'r':>6s}  {'Lin_rc':>6s}")
    for name, pred in [
        ("Connor 2023 BSR (paper)", bsr_paper_1hz),
        ("QUAZI BSR", bsr_quazi_1hz),
        ("OpenBSR 2025", bsr_openbsr_1hz),
    ]:
        mae, r, rc = metrics(sr_actual, pred, valid_bsr)
        print(f"{name:<32s}  {mae:6.2f}  {r:6.3f}  {rc:6.3f}")

    # Plot grid
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

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6, 1]})

    t_eeg = np.arange(len(eeg)) / FS
    axes[0].plot(t_eeg[::FS], eeg[::FS], lw=0.3, color="steelblue")
    for s0, s1 in shade:
        axes[0].axvspan(s0, s1, color="gray", alpha=0.2)
    axes[0].set_ylabel("EEG (µV)")
    axes[0].set_title("Preprocessed EEG (gray = SQI < 80)")
    axes[0].set_ylim(-60, 60)

    axes[1].plot(t, bis_actual, lw=1.2, color="black", alpha=0.9, label="Actual BIS")
    colors = {
        ("paper", "paper"): "tab:red",
        ("paper", "ellerkmann"): "tab:orange",
        ("quazi", "paper"): "tab:green",
        ("quazi", "ellerkmann"): "tab:blue",
    }
    for key, pred in bis_1hz.items():
        axes[1].plot(t, pred, lw=0.8, color=colors[key], alpha=0.7,
                     label=f"openibis(bsr={key[0]!r}, deep={key[1]!r})")
    for s0, s1 in shade:
        axes[1].axvspan(s0, s1, color="gray", alpha=0.2)
    axes[1].set_ylabel("BIS")
    axes[1].set_title("Actual BIS vs openibis variants")
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t, sr_actual, lw=1.2, color="black", alpha=0.9, label="Actual SR")
    axes[2].plot(t, bsr_paper_1hz, lw=0.8, color="tab:red", alpha=0.7, label="Connor 2023 BSR")
    axes[2].plot(t, bsr_quazi_1hz, lw=0.8, color="tab:green", alpha=0.7, label="QUAZI BSR")
    axes[2].plot(t, bsr_openbsr_1hz, lw=0.8, color="tab:blue", alpha=0.7, label="OpenBSR 2025")
    for s0, s1 in shade:
        axes[2].axvspan(s0, s1, color="gray", alpha=0.2)
    axes[2].set_ylabel("BSR / SR (%)")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_title("Suppression Ratio comparison")
    axes[2].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out = Path(__file__).resolve().parents[1] / "validate_1vital.png"
    plt.savefig(str(out), dpi=140)
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
