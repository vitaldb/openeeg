"""Phase 3j-2: extract regression-coefficient evidence from sub-parameter
scatter boundaries on our cohort.

User insights validated here:

(1) **BIS histogram peaks at 40 / 60 imply regime OVERLAP, not strict
    boundaries.** If different regression formulas were applied in
    each regime with hard clipping, the boundary values should pile
    sharply. Smooth peaks at 40 / 60 in the actual data look more
    like the overlap signature.

(2) **EMG raises BIS in apparently anaesthetised patients** — when
    EMG climbs while the spectrum still looks "asleep", BIS shoots
    past 60. Visible as a positive coefficient on EMG inside the
    BIS aggregation rule.

(3) **The diagonal boundary in the BIS vs SEF scatter reveals SEF's
    coefficient.** ``BIS = const + a·SEF + …`` means certain
    (SEF, BIS) combinations are physically impossible: the scatter
    envelope slope is a direct estimate of ``a``.

Outputs:
  results/bis_subparam_boundaries.png  — 4-panel figure
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    val = pd.read_parquet(RESULTS / "features_val_n100_v3.parquet")
    val = val.dropna(subset=["target"])
    val = val[val["sqi"] >= 80]
    print(f"Loaded {len(val):,} scoring rows over {val['case_id'].nunique()} cases")

    bis = val["target"].values
    sef = val["bis_sef_oracle"].values     # Vista oracle (clean)
    emg = val["bis_emg_oracle"].values
    bsr = val["bis_sr_oracle"].values
    br  = val["beta_ratio"].values
    sef_r = val["sef95"].values            # our raw SEF

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # ----- (a) BIS histogram — look for peaks at 40 and 60 -----
    ax = axes[0, 0]
    bins_edges = np.arange(0, 101, 1)
    counts, _ = np.histogram(bis, bins=bins_edges)
    centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
    ax.bar(centers, counts, width=1.0, color="black", alpha=0.7)
    # Mark candidate boundary peaks
    for v in [21, 41, 61, 78]:
        ax.axvline(v, color="tab:red", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel("BIS")
    ax.set_ylabel("epoch count")
    ax.set_title("(a) Histogram of actual BIS  — peaks at 40/60 = regime overlap (user)")
    ax.set_xlim(0, 100)
    # Find local maxima around 40 and 60
    for target in (40, 60):
        i_lo, i_hi = target - 3, target + 3
        peak_count = counts[i_lo:i_hi + 1].max()
        peak_pos = i_lo + int(np.argmax(counts[i_lo:i_hi + 1]))
        ax.text(peak_pos, peak_count, f"  peak ≈ {peak_pos}",
                color="tab:red", fontsize=9, va="bottom")

    # ----- (b) BIS vs SEF (Vista) — extract diagonal boundary -----
    ax = axes[0, 1]
    m = np.isfinite(sef) & np.isfinite(bis)
    # Subsample for plotting
    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(m)[0], min(80000, m.sum()), replace=False)
    ax.scatter(sef[idx], bis[idx], s=0.5, color="black", alpha=0.1)

    # Upper envelope: 99-th percentile of BIS at each SEF bin
    sef_edges = np.linspace(0, 30, 31)
    centers_s = 0.5 * (sef_edges[:-1] + sef_edges[1:])
    env_lo = np.full(30, np.nan); env_hi = np.full(30, np.nan); env_med = np.full(30, np.nan)
    for i in range(30):
        bm = (sef >= sef_edges[i]) & (sef < sef_edges[i + 1]) & np.isfinite(bis)
        if bm.sum() >= 50:
            env_hi[i] = float(np.percentile(bis[bm], 99))
            env_lo[i] = float(np.percentile(bis[bm], 1))
            env_med[i] = float(np.median(bis[bm]))
    ok = np.isfinite(env_hi)

    # Fit slope on the upper envelope (the diagonal the user drew)
    if ok.sum() >= 4:
        # Use only the rising part — SEF 4 to 20 typically
        rising = ok & (centers_s >= 4) & (centers_s <= 20)
        if rising.sum() >= 4:
            slope_hi, intercept_hi = np.polyfit(centers_s[rising], env_hi[rising], 1)
            ax.plot(centers_s, np.clip(slope_hi * centers_s + intercept_hi, 0, 100),
                    color="tab:red", lw=2.0,
                    label=f"P99 upper envelope: BIS ≤ {intercept_hi:.1f} + {slope_hi:+.2f}·SEF")
        if (ok & (centers_s <= 20)).sum() >= 4:
            slope_md, intercept_md = np.polyfit(centers_s[ok], env_med[ok], 1)
            ax.plot(centers_s, slope_md * centers_s + intercept_md,
                    color="tab:orange", lw=1.5, ls="--",
                    label=f"median: BIS ≈ {intercept_md:.1f} + {slope_md:+.2f}·SEF")

    # Morimoto 2004 BcSEF reference: BIS = 2.3·BcSEF + 12 (at BSR=0, BcSEF=SEF)
    sef_grid = np.linspace(0, 30, 100)
    ax.plot(sef_grid, np.clip(2.3 * sef_grid + 12, 0, 100),
            color="tab:blue", lw=1.5, ls="-",
            label="Morimoto 2004: BIS = 2.3·SEF + 12  (BSR=0)")

    ax.set_xlabel("BIS/SEF (Hz, Vista oracle)")
    ax.set_ylabel("actual BIS")
    ax.set_xlim(0, 30); ax.set_ylim(0, 100)
    ax.set_title("(b) BIS vs SEF: diagonal boundary slope → SEF coefficient")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

    # ----- (c) BIS vs EMG conditioned on SEF (Lee's mid gate variants) -----
    ax = axes[1, 0]
    m_lo_sef = (sef < 20) & np.isfinite(emg) & np.isfinite(bis)
    idx = rng.choice(np.where(m_lo_sef)[0], min(80000, m_lo_sef.sum()), replace=False)
    ax.scatter(emg[idx], bis[idx], s=0.5, color="black", alpha=0.10)

    # Local mean & 95p of BIS at each EMG bin (proves "EMG ↑ → BIS ↑")
    emg_edges = np.linspace(20, 60, 41)
    centers_e = 0.5 * (emg_edges[:-1] + emg_edges[1:])
    mean_bis = np.full(40, np.nan); p95_bis = np.full(40, np.nan)
    for i in range(40):
        bm = (emg >= emg_edges[i]) & (emg < emg_edges[i + 1]) & m_lo_sef
        if bm.sum() >= 50:
            mean_bis[i] = float(np.mean(bis[bm]))
            p95_bis[i]  = float(np.percentile(bis[bm], 95))
    ok = np.isfinite(mean_bis)
    if ok.sum() >= 4:
        slope_e, intercept_e = np.polyfit(centers_e[ok], mean_bis[ok], 1)
        ax.plot(centers_e[ok], mean_bis[ok], color="tab:orange", lw=2.0,
                label=f"mean BIS | SEF<20: {intercept_e:.1f} + {slope_e:+.2f}·EMG")
        ax.plot(centers_e[ok], p95_bis[ok], color="tab:red", lw=1.5, ls="--",
                label="P95 BIS | SEF<20")
    ax.axhline(60, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("BIS/EMG (dB, Vista oracle)")
    ax.set_ylabel("actual BIS")
    ax.set_xlim(20, 60); ax.set_ylim(0, 100)
    ax.set_title("(c) BIS vs EMG conditioned on SEF<20 — EMG positive coefficient")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

    # ----- (d) Histogram comparison: actual vs predict_bis output -----
    ax = axes[1, 1]
    # predict_bis_v2 was just shipped — load its predictions on val parquet
    import lightgbm as lgb
    base15 = ['openibis_paper','openibis_quazi','openibis_quazi_30s','bsr_paper','bsr_quazi','sef95','bcsef','beta_ratio','emg_proxy','p_delta','p_theta','p_alpha','p_beta','p_lowgamma','spectral_entropy']
    extra = ['openibis_quazi_5s','openibis_quazi_10s','openibis_quazi_60s','openibis_quazi_dt','openibis_quazi_30s_dt','sef95_dt','emg_proxy_dt']
    feat22 = base15 + extra
    booster = lgb.Booster(model_file="openeeg/models/predict_bis_v2.txt")
    pred = np.clip(booster.predict(val[feat22].values), 0, 100)

    bins_edges = np.arange(0, 101, 1)
    a_counts, _ = np.histogram(bis, bins=bins_edges)
    p_counts, _ = np.histogram(pred, bins=bins_edges)
    centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
    ax.bar(centers - 0.2, a_counts, width=0.4, color="black", alpha=0.7,
           label="actual BIS")
    ax.bar(centers + 0.2, p_counts, width=0.4, color="tab:red", alpha=0.6,
           label="predict_bis_v2 raw")
    for v in [21, 41, 61, 78]:
        ax.axvline(v, color="tab:gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("BIS")
    ax.set_ylabel("epoch count")
    ax.set_xlim(0, 100)
    ax.set_title("(d) Histogram comparison: actual vs predict_bis_v2")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = RESULTS / "bis_subparam_boundaries.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved: {out.name}")

    # Numerical reporting
    print(f"\n=== Histogram peak analysis on actual BIS ===")
    for tgt in [21, 41, 61, 78]:
        i_lo, i_hi = max(0, tgt - 3), min(100, tgt + 3)
        peak_pos = i_lo + int(np.argmax(a_counts[i_lo:i_hi + 1]))
        print(f"  near {tgt}: peak at BIS={peak_pos}  count={a_counts[peak_pos]:,}")
    print(f"\n=== BIS vs SEF boundary slopes ===")
    print(f"  P99 envelope rising part:  slope={slope_hi:+.3f}  intercept={intercept_hi:.1f}")
    print(f"  Median fit (all SEF≤20):   slope={slope_md:+.3f}  intercept={intercept_md:.1f}")
    print(f"  Morimoto 2004 BcSEF:       slope=+2.30  intercept=12.0 (n=20 isoflurane)")
    print(f"\n=== BIS vs EMG (SEF<20 cohort) mean fit ===")
    print(f"  mean BIS = {intercept_e:.1f} + {slope_e:+.3f} · EMG")


if __name__ == "__main__":
    main()
