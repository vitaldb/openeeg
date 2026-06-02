"""Phase 3j — extract a hard upper-envelope rule from BSR vs actual BIS.

The user's slide (Morimoto 2004 + this BSR/BIS scatter) shows that
``BSR ≈ X`` implies ``BIS ≲ f(X)``. The scatter is bounded above by a
clear envelope — once burst suppression accumulates, the BIS Vista
cannot report a high BIS.

We use the new line-by-line ``openbsr`` (r ≈ 0.93 vs Vista BSR on
deep-rich cases) to estimate this envelope from raw EEG alone, fit
the envelope, and compare to Ellerkmann's published formula
``BIS = 44.1 − BSR / 2.25``.

Outputs: results/bsr_bis_envelope.png with three panels:
  (a) actual BIS vs Vista BSR (oracle reference)
  (b) actual BIS vs new openbsr (standalone)
  (c) actual BIS vs bsr_quazi  (the previous best raw detector)
Each panel overlays Ellerkmann's line, plus a data-driven quantile
envelope (99-th percentile of BIS at each BSR bin).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg import openbsr
from openeeg.cohort import load_case, preprocess_eeg

RESULTS = Path(__file__).resolve().parents[1] / "results"


def add_openbsr_column(val_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute openbsr per case (the v4 parquet has the OLD prose port)."""
    pieces = []
    cids = sorted(val_df["case_id"].unique())
    print(f"Recomputing openbsr (line-by-line port) on {len(cids)} cases...")
    t0 = time.time()
    for i, cid in enumerate(cids, 1):
        case = load_case(int(cid))
        if case is None:
            continue
        eeg = preprocess_eeg(case["eeg"])
        ob_2hz = openbsr(eeg)
        ob_1hz = ob_2hz[::2]
        n = len(ob_1hz)
        pieces.append(pd.DataFrame({
            "case_id": int(cid),
            "time_sec": np.arange(n, dtype=np.int32),
            "openbsr_v2": ob_1hz.astype(np.float32),
        }))
        if i % 25 == 0:
            print(f"  {i}/{len(cids)}  elapsed {time.time()-t0:.0f}s")
    add = pd.concat(pieces, ignore_index=True)
    return val_df.merge(add, on=["case_id", "time_sec"], how="left")


def upper_envelope(x: np.ndarray, y: np.ndarray, n_bins: int = 40,
                   x_min=0.0, x_max=100.0, percentile: float = 99.0):
    """Return (bin centers, p-th percentile of y in each bin)."""
    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    env = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1]) & np.isfinite(y)
        if m.sum() >= 30:
            env[i] = float(np.percentile(y[m], percentile))
    return centers, env


def panel(ax, x, y, label_x, ell_x=None, ell_y=None, title="",
          subsample=80_000):
    m = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[m], y[m]
    if len(xv) > subsample:
        idx = np.random.default_rng(0).choice(len(xv), subsample, replace=False)
        xv_p, yv_p = xv[idx], yv[idx]
    else:
        xv_p, yv_p = xv, yv

    ax.scatter(xv_p, yv_p, s=0.5, color="black", alpha=0.10)

    # Ellerkmann's line
    bsr_grid = np.linspace(0, 100, 200)
    ell = np.clip(44.1 - bsr_grid / 2.25, 0, 100)
    ax.plot(bsr_grid, ell, color="tab:red", lw=2.0,
            label="Ellerkmann 2004  BIS = 44.1 − BSR/2.25")

    # Data-driven upper envelope (99-th percentile)
    centers, env99 = upper_envelope(xv, yv, n_bins=40)
    ok = np.isfinite(env99)
    ax.plot(centers[ok], env99[ok], color="tab:orange", lw=1.6,
            marker="o", ms=3, label="P99 envelope of actual BIS")

    # Linear fit to envelope (slope, intercept)
    if ok.sum() >= 4:
        slope, intercept = np.polyfit(centers[ok], env99[ok], 1)
        ax.plot(bsr_grid, np.clip(slope * bsr_grid + intercept, 0, 100),
                color="tab:blue", lw=1.5, ls="--",
                label=f"fit  BIS_max = {intercept:.1f} + {slope:+.3f}·{label_x}")

    ax.set_xlabel(label_x)
    ax.set_ylabel("actual BIS")
    ax.set_xlim(-2, 100)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)


def main():
    val = pd.read_parquet(RESULTS / "features_val_n100_v3.parquet")
    val = add_openbsr_column(val)
    val = val.dropna(subset=["target"])
    val = val[val["sqi"] >= 80]
    print(f"\nTotal scoring rows (SQI ≥ 80): {len(val):,}")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
    panel(axes[0],
          val["bis_sr_oracle"].values, val["target"].values,
          "Vista BIS/SR (oracle)",
          title=f"(a) Vista oracle BSR vs actual BIS  (N={len(val):,})")
    panel(axes[1],
          val["openbsr_v2"].values, val["target"].values,
          "openbsr (Table 1 port)",
          title=f"(b) New openbsr (r ≈ 0.93 vs Vista) vs actual BIS")
    panel(axes[2],
          val["bsr_quazi"].values, val["target"].values,
          "bsr_quazi",
          title=f"(c) bsr_quazi (legacy detector) vs actual BIS")
    plt.tight_layout()
    out = RESULTS / "bsr_bis_envelope.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved: {out.name}")

    # Numerical fits
    print(f"\n=== P99 envelope linear fits ===")
    for col, label in [("bis_sr_oracle", "Vista BSR (oracle)"),
                        ("openbsr_v2",   "new openbsr"),
                        ("bsr_quazi",    "bsr_quazi (legacy)")]:
        x = val[col].values; y = val["target"].values
        m = np.isfinite(x) & np.isfinite(y)
        c, e = upper_envelope(x[m], y[m], n_bins=40)
        ok = np.isfinite(e)
        if ok.sum() >= 4:
            slope, intercept = np.polyfit(c[ok], e[ok], 1)
            mask_high_bsr = (c[ok] >= 20)
            if mask_high_bsr.sum() >= 4:
                slope2, intercept2 = np.polyfit(c[ok][mask_high_bsr], e[ok][mask_high_bsr], 1)
                print(f"  {label:<22s}  P99 fit (full):  intercept={intercept:5.1f}  slope={slope:+.4f}  "
                      f"|  BSR≥20 only:  intercept={intercept2:5.1f}  slope={slope2:+.4f}")

    # Test Ellerkmann as a hard ceiling: how often does actual BIS exceed it?
    print(f"\n=== Does actual BIS ever exceed Ellerkmann ceiling? (BIS > 44.1 − BSR/2.25) ===")
    for col, label in [("bis_sr_oracle", "Vista BSR"),
                        ("openbsr_v2",   "new openbsr")]:
        x = val[col].values; y = val["target"].values
        m = np.isfinite(x) & np.isfinite(y) & (x > 5)  # only meaningful when BSR is engaged
        ceiling = 44.1 - x[m] / 2.25
        over = (y[m] > ceiling + 5).sum()
        print(f"  {label:<22s}  N(BSR>5)={int(m.sum()):>7,d}  BIS exceeds ceiling by >5 in "
              f"{over:>6,d} epochs ({100*over/max(m.sum(),1):.1f}%)")


if __name__ == "__main__":
    main()
