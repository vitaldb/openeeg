"""Find a Lee-2019-style rule cutoff for BIS ≥ 80 (awake).

Reproduces the slide's quadrant-scatter analysis on our 100-case val
cohort using Vista oracle sub-parameters (cleanest signal) plus our
raw-EEG-derived counterparts.

Output: results/subparam_rule_80.png with 6 panels:
   row 1 — Vista oracle: BIS vs EMG, conditioned on SEF (high / low)
   row 2 — Vista oracle: BIS vs SEF, conditioned on EMG (high / low)
   row 3 — raw-EEG features (emg_proxy + sef95) same two views

Each quadrant is labelled with the percentage of total samples it
contains (mirrors the user's slide layout).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).resolve().parents[1] / "results"


def quadrant_scatter(ax, x, y, x_cut, y_cut, x_label, y_label, title,
                     x_min=0, x_max=80, y_min=0, y_max=100):
    """Scatter with quadrant percentage labels + cutoff lines."""
    m = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[m], y[m]
    n = len(xv)
    q_tl = ((xv < x_cut) & (yv >= y_cut)).sum() / max(n, 1)
    q_tr = ((xv >= x_cut) & (yv >= y_cut)).sum() / max(n, 1)
    q_bl = ((xv < x_cut) & (yv < y_cut)).sum() / max(n, 1)
    q_br = ((xv >= x_cut) & (yv < y_cut)).sum() / max(n, 1)

    ax.scatter(xv, yv, s=0.5, color="black", alpha=0.15)
    ax.axvline(x_cut, color="tab:red", lw=1.5)
    ax.axhline(y_cut, color="tab:red", lw=1.5)

    # Quadrant tints
    ax.fill_between([x_min, x_cut], y_cut, y_max, color="tab:red", alpha=0.05)
    ax.fill_between([x_cut, x_max], y_cut, y_max, color="tab:blue", alpha=0.05)
    ax.fill_between([x_min, x_cut], y_min, y_cut, color="tab:gray", alpha=0.05)
    ax.fill_between([x_cut, x_max], y_min, y_cut, color="tab:orange", alpha=0.05)

    # % labels
    pad_x = (x_max - x_min) * 0.02
    pad_y = (y_max - y_min) * 0.02
    ax.text(x_min + pad_x, y_max - pad_y, f"{q_tl*100:.1f}%",
            ha="left", va="top", fontsize=11, color="tab:red", weight="bold")
    ax.text(x_max - pad_x, y_max - pad_y, f"{q_tr*100:.1f}%",
            ha="right", va="top", fontsize=11, color="tab:blue", weight="bold")
    ax.text(x_min + pad_x, y_min + pad_y, f"{q_bl*100:.1f}%",
            ha="left", va="bottom", fontsize=11, color="tab:gray", weight="bold")
    ax.text(x_max - pad_x, y_min + pad_y, f"{q_br*100:.1f}%",
            ha="right", va="bottom", fontsize=11, color="tab:orange", weight="bold")

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=10)


def main():
    val = pd.read_parquet(RESULTS / "features_val_n100_v3.parquet")
    print(f"Loaded {len(val):,} rows from {val['case_id'].nunique()} val cases")

    bis = val["target"].values
    emg_o = val["bis_emg_oracle"].values
    sef_o = val["bis_sef_oracle"].values
    emg_r = val["emg_proxy"].values
    sef_r = val["sef95"].values
    br = val["beta_ratio"].values
    bsr_q = val["bsr_quazi"].values
    oibis30s = val["openibis_quazi_30s"].values

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    # --- Row 1: Vista EMG vs BIS, conditioned on Vista SEF ---
    hi_sef = sef_o >= 20
    lo_sef = sef_o < 20
    quadrant_scatter(axes[0, 0],
                     emg_o[hi_sef], bis[hi_sef],
                     x_cut=40, y_cut=80,
                     x_label="BIS/EMG (dB)", y_label="actual BIS",
                     title=f"Vista oracle  ·  SEF ≥ 20 Hz  (N={hi_sef.sum():,})")
    quadrant_scatter(axes[0, 1],
                     emg_o[lo_sef], bis[lo_sef],
                     x_cut=40, y_cut=80,
                     x_label="BIS/EMG (dB)", y_label="actual BIS",
                     title=f"Vista oracle  ·  SEF < 20 Hz  (N={lo_sef.sum():,})")
    # Marginal: EMG vs BIS, all data
    quadrant_scatter(axes[0, 2],
                     emg_o, bis,
                     x_cut=40, y_cut=80,
                     x_label="BIS/EMG (dB)", y_label="actual BIS",
                     title=f"Vista oracle  ·  all data  (N={len(bis):,})")

    # --- Row 2: Vista SEF vs BIS, conditioned on Vista EMG ---
    hi_emg = emg_o >= 40
    lo_emg = emg_o < 40
    quadrant_scatter(axes[1, 0],
                     sef_o[hi_emg], bis[hi_emg],
                     x_cut=20, y_cut=80,
                     x_label="BIS/SEF (Hz)", y_label="actual BIS",
                     title=f"Vista oracle  ·  EMG ≥ 40 dB  (N={hi_emg.sum():,})",
                     x_min=0, x_max=30)
    quadrant_scatter(axes[1, 1],
                     sef_o[lo_emg], bis[lo_emg],
                     x_cut=20, y_cut=80,
                     x_label="BIS/SEF (Hz)", y_label="actual BIS",
                     title=f"Vista oracle  ·  EMG < 40 dB  (N={lo_emg.sum():,})",
                     x_min=0, x_max=30)
    # BetaRatio vs BIS
    quadrant_scatter(axes[1, 2],
                     br, bis,
                     x_cut=-0.3, y_cut=80,
                     x_label="beta_ratio (log10)", y_label="actual BIS",
                     title=f"raw EEG  ·  BetaRatio vs BIS (N={len(bis):,})",
                     x_min=-3, x_max=2)

    # --- Row 3: Raw-EEG (emg_proxy, sef95) — for our standalone library ---
    hi_sef_r = sef_r >= 20
    lo_sef_r = sef_r < 20
    quadrant_scatter(axes[2, 0],
                     emg_r[hi_sef_r], bis[hi_sef_r],
                     x_cut=40, y_cut=80,
                     x_label="emg_proxy (dB)", y_label="actual BIS",
                     title=f"raw EEG  ·  sef95 ≥ 20  (N={hi_sef_r.sum():,})",
                     x_min=-40, x_max=60)
    quadrant_scatter(axes[2, 1],
                     emg_r[lo_sef_r], bis[lo_sef_r],
                     x_cut=40, y_cut=80,
                     x_label="emg_proxy (dB)", y_label="actual BIS",
                     title=f"raw EEG  ·  sef95 < 20  (N={lo_sef_r.sum():,})",
                     x_min=-40, x_max=60)
    # openibis_quazi_30s vs BIS
    quadrant_scatter(axes[2, 2],
                     oibis30s, bis,
                     x_cut=70, y_cut=80,
                     x_label="openibis_quazi_30s", y_label="actual BIS",
                     title=f"raw EEG  ·  oibis_30s vs BIS (N={len(bis):,})")

    plt.tight_layout()
    out = RESULTS / "subparam_rule_80.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nSaved: {out.name}")

    # --- Numerical analysis: best 2-rule combination for BIS≥80 ---
    print(f"\n=== Best 2-rule gates for BIS ≥ 80 ===")
    awake_mask = bis >= 80
    print(f"  Total awake (BIS≥80) epochs: {awake_mask.sum():,} ({100*awake_mask.mean():.2f}%)")

    rules = []
    for sef_cut in [15, 18, 20, 22, 25]:
        for emg_cut in [30, 35, 40, 45, 50]:
            # Vista oracle rule
            rule = (sef_o >= sef_cut) & (emg_o >= emg_cut)
            tp = (rule & awake_mask).sum()
            sens = tp / max(awake_mask.sum(), 1)
            prec = tp / max(rule.sum(), 1)
            f1 = 2 * sens * prec / max(sens + prec, 1e-9)
            rules.append(("Vista oracle", sef_cut, emg_cut, int(rule.sum()), sens, prec, f1))
    rules.sort(key=lambda r: r[-1], reverse=True)
    print(f"  Vista oracle composite rule (SEF≥X AND EMG≥Y):")
    print(f"  {'sef_cut':>8s}  {'emg_cut':>8s}  {'fires':>9s}  {'sens':>5s}  {'prec':>5s}  {'F1':>5s}")
    for src, sc, ec, n, s, p, f in rules[:5]:
        print(f"  {sc:>8.1f}  {ec:>8.1f}  {n:>9,d}  {s:.2f}  {p:.2f}  {f:.2f}")

    # Raw-EEG rule sweep
    rules_r = []
    for sef_cut in [15, 17, 20, 22, 25]:
        for emg_cut in [10, 20, 30, 40, 50]:
            rule = (sef_r >= sef_cut) & (emg_r >= emg_cut)
            tp = (rule & awake_mask).sum()
            sens = tp / max(awake_mask.sum(), 1)
            prec = tp / max(rule.sum(), 1)
            f1 = 2 * sens * prec / max(sens + prec, 1e-9)
            rules_r.append((sef_cut, emg_cut, int(rule.sum()), sens, prec, f1))
    rules_r.sort(key=lambda r: r[-1], reverse=True)
    print(f"\n  Raw-EEG composite rule (sef95≥X AND emg_proxy≥Y):")
    print(f"  {'sef_cut':>8s}  {'emg_cut':>8s}  {'fires':>9s}  {'sens':>5s}  {'prec':>5s}  {'F1':>5s}")
    for sc, ec, n, s, p, f in rules_r[:5]:
        print(f"  {sc:>8.1f}  {ec:>8.1f}  {n:>9,d}  {s:.2f}  {p:.2f}  {f:.2f}")


if __name__ == "__main__":
    main()
