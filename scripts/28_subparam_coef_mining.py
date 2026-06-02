"""Phase B — Sub-parameter coefficient mining from scatter envelopes.

For each candidate sub-parameter X (BSR variants, SEF variants,
EMG variants, BetaRatio, BcSEF, openibis_quazi_30s), bin actual BIS
by X and fit linear slopes through three envelopes (P1, median, P99)
within the monotone regime of that variable. The slope of an envelope
is a direct, model-free estimate of the per-region regression
coefficient that BIS uses on X.

Cross-check against published constants:
  * Morimoto 2004 : BIS = 2.3·SEF + 12        (BcSEF, BIS<80)
  * Morimoto 2004 : BIS = 20·BetaRatio + 95   (BIS>60)
  * Ellerkmann 2004: BIS = 44.1 − BSR/2.25    (BSR>40)
  * Lee 2019      : per-region coefficients in lee2019_replication_coefs.csv

Outputs
  results/subparam_coef_mining.csv  — one row per (variable, percentile, regime)
  results/subparam_coef_mining.png  — 3×3 small-multiples
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _compute_openbsr_for_case(cid):
    """Worker — needs module-level definition for multiprocessing pickling."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from openeeg.cohort import load_case, preprocess_eeg
    from openeeg import openbsr as _openbsr
    case = load_case(int(cid))
    if case is None:
        return None
    eeg = preprocess_eeg(case["eeg"])
    ob_1hz = _openbsr(eeg)[::2]
    return pd.DataFrame({
        "case_id": int(cid),
        "time_sec": np.arange(len(ob_1hz), dtype=np.int32),
        "openbsr": ob_1hz.astype(np.float32),
    })


# Each row: (column_name, label, x_min, x_max, monotone_window, monotone_dir)
# monotone_dir: +1 means BIS rises with X (e.g. SEF), -1 means BIS falls (e.g. BSR)
VARIABLES = [
    ("bis_sr_oracle",      "Vista BSR (oracle, %)",     0,  80,  (5,  60),  -1),
    ("openbsr",            "openbsr (new, %)",           0,  80,  (5,  60),  -1),
    ("bsr_quazi",          "bsr_quazi (legacy, %)",      0,  80,  (5,  60),  -1),
    ("bis_sef_oracle",     "Vista SEF (oracle, Hz)",     5,  30,  (5,  22),  +1),
    ("sef95",              "sef95 (our raw, Hz)",        5,  30,  (5,  22),  +1),
    ("bis_emg_oracle",     "Vista EMG (oracle, dB)",    25,  60,  (25, 55),  +1),
    ("emg_proxy",          "emg_proxy (47-63 Hz, dB)", -35,  10, (-30, 5),  +1),
    ("beta_ratio",         "BetaRatio (log10)",         -3,   2,  (-2, 1.0), +1),
    ("openibis_quazi_30s", "openibis_quazi_30s",         0, 100,  (10, 90),  +1),
]


def w15_filter(df, oracle_w_csv):
    w = pd.read_csv(oracle_w_csv)
    keep = set(w.loc[w["oracle_W"] == 15, "case_id"].astype(int))
    return df[df["case_id"].isin(keep)].reset_index(drop=True)


def envelope(x, y, x_min, x_max, n_bins=40, percentiles=(1, 50, 99), min_n=200):
    """Return bin centres and percentiles of y per X bin."""
    edges = np.linspace(x_min, x_max, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    out = {p: np.full(n_bins, np.nan) for p in percentiles}
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        bm = (x >= edges[i]) & (x < edges[i + 1]) & np.isfinite(y)
        counts[i] = int(bm.sum())
        if counts[i] >= min_n:
            for p in percentiles:
                out[p][i] = float(np.percentile(y[bm], p))
    return centres, out, counts


def linear_fit(x, y, mask):
    valid = mask & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    pred = slope * x[valid] + intercept
    ss_res = float(np.sum((y[valid] - pred) ** 2))
    ss_tot = float(np.sum((y[valid] - y[valid].mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return float(slope), float(intercept), float(r2)


def main():
    print("Loading W=15 train cohort...")
    train = pd.read_parquet(RESULTS / "features_train_n500_v3.parquet")
    train_w15 = w15_filter(train, RESULTS / "oracle_W_train.csv")
    # openbsr column may not be in v3; cache the augmented parquet under v5 so
    # later phases can reuse it.
    v5_path = RESULTS / "features_train_n500_v5.parquet"
    if "openbsr" not in train_w15.columns:
        if v5_path.exists():
            print(f"  loading cached {v5_path.name}")
            cached = pd.read_parquet(v5_path)
            train_w15 = w15_filter(cached, RESULTS / "oracle_W_train.csv")
        else:
            print("  computing openbsr with multiprocessing (32 workers)...")
            from multiprocessing import Pool
            import time as _time
            t0 = _time.time()
            cids = sorted(train_w15["case_id"].unique())
            with Pool(32) as pool:
                pieces = [p for p in pool.map(_compute_openbsr_for_case, cids) if p is not None]
            aug = pd.concat(pieces, ignore_index=True)
            train_w15 = train_w15.merge(aug, on=["case_id", "time_sec"], how="left")
            # Cache the full (unfiltered) parquet so other scripts can reuse
            # — we have to re-attach openbsr to the full train parquet too.
            full = pd.read_parquet(RESULTS / "features_train_n500_v3.parquet")
            full = full.merge(aug, on=["case_id", "time_sec"], how="left")
            full.to_parquet(v5_path, index=False, compression="zstd")
            print(f"  done {_time.time()-t0:.1f}s   cached {v5_path.name}")

    print(f"  train W=15: {len(train_w15):,} rows / {train_w15['case_id'].nunique()} cases")

    y = train_w15["target"].values
    rows = []
    panels_data = []

    for col, label, x_min, x_max, (m_lo, m_hi), direction in VARIABLES:
        if col not in train_w15.columns:
            print(f"  skipping {col} (not in parquet)")
            continue
        x = train_w15[col].values
        centres, env, counts = envelope(x, y, x_min, x_max, n_bins=40,
                                         percentiles=(1, 50, 99), min_n=500)
        ok = counts >= 500
        # Mask to monotone window
        mono = ok & (centres >= m_lo) & (centres <= m_hi)
        slopes = {}
        for p in (1, 50, 99):
            s, b, r2 = linear_fit(centres, env[p], mono)
            slopes[p] = (s, b, r2)
            rows.append(dict(variable=col, label=label, percentile=p,
                             monotone_lo=m_lo, monotone_hi=m_hi, direction=direction,
                             slope=s, intercept=b, r2=r2,
                             n_bins_used=int(mono.sum())))
        panels_data.append(dict(col=col, label=label, x_min=x_min, x_max=x_max,
                                centres=centres, env=env, counts=counts,
                                mono=mono, slopes=slopes, direction=direction))

    # Save CSV
    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "subparam_coef_mining.csv", index=False)
    print(f"\nSaved {RESULTS/'subparam_coef_mining.csv'}")

    # Print summary
    print("\n=== Slope summary (one row per (variable, percentile)) ===")
    print(df_out[["variable", "percentile", "slope", "intercept", "r2", "n_bins_used"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n=== Cross-check vs literature ===")
    refs = [
        ("Morimoto 2004 BcSEF coef",   "BIS ≈ 12 + 2.30·SEF",      "median fit on sef95 or bis_sef_oracle"),
        ("Morimoto 2004 RBR coef",     "BIS ≈ 95 + 20·BetaRatio",   "median fit on beta_ratio"),
        ("Ellerkmann 2004 BSR coef",   "BIS ≈ 44.1 − 0.444·BSR",   "median fit on bis_sr_oracle"),
        ("Lee 2019 surg SEF coef",     "BIS ≈ +3.84·SEF in surg",   "median fit on bis_sef_oracle"),
    ]
    for label, formula, fit_hint in refs:
        print(f"  {label}: {formula}  ({fit_hint})")
    medians = df_out[df_out["percentile"] == 50].set_index("variable")[["slope", "intercept"]]
    print(f"\n  Our median fits:")
    for v in ["bis_sef_oracle", "sef95", "beta_ratio", "bis_sr_oracle", "openbsr"]:
        if v in medians.index:
            s, b = medians.loc[v, "slope"], medians.loc[v, "intercept"]
            print(f"    {v:<22s}  BIS ≈ {b:+.2f} + {s:+.3f}·X")

    # 3x3 multipanel figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    rng = np.random.default_rng(0)
    for ax, panel in zip(axes.flat, panels_data):
        col = panel["col"]; centres = panel["centres"]; env = panel["env"]
        # Show a scatter subsample
        x = train_w15[col].values
        m = np.isfinite(x) & np.isfinite(y)
        idx = rng.choice(np.where(m)[0], min(40000, m.sum()), replace=False)
        ax.scatter(x[idx], y[idx], s=0.3, color="black", alpha=0.08)
        # Envelopes
        for p, color in zip((1, 50, 99), ("tab:blue", "tab:orange", "tab:red")):
            valid = np.isfinite(env[p])
            ax.plot(centres[valid], env[p][valid], color=color, lw=1.4, marker="o", ms=3,
                    label=f"P{p}")
        # Slope fits within monotone window
        s50, b50, r2_50 = panel["slopes"][50]
        s99, b99, _ = panel["slopes"][99]
        s1, b1, _ = panel["slopes"][1]
        x_line = np.linspace(panel["x_min"], panel["x_max"], 50)
        if np.isfinite(s50):
            ax.plot(x_line, np.clip(s50 * x_line + b50, 0, 100),
                    color="tab:orange", ls="--", lw=1.0)
        if np.isfinite(s99):
            ax.plot(x_line, np.clip(s99 * x_line + b99, 0, 100),
                    color="tab:red", ls="--", lw=1.0)
        ax.set_xlim(panel["x_min"], panel["x_max"])
        ax.set_ylim(0, 100)
        ax.set_xlabel(panel["label"])
        ax.set_ylabel("actual BIS")
        ax.set_title(f"{col}  |  med slope {s50:+.2f}  P99 slope {s99:+.2f}",
                     fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    out = RESULTS / "subparam_coef_mining.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
