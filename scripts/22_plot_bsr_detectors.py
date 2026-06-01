"""Visualise 4 BSR detectors (bsr_paper / bsr_quazi / openbsr / Vista BSR)
on 10 representative val cases.

Picks a mix:
  * top-3 cases by Vista BSR median  (heavy deep-suppression cohort)
  * top-3 cases with the largest BSR dynamic range (transient deep)
  * 4 random cases for contrast

Each subplot overlays the four detectors + the actual BIS on a twin y-axis,
with SQI<80 windows shaded.
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
    df = pd.read_parquet(RESULTS / "features_val_n100_v4.parquet")
    print(f"Loaded {len(df):,} rows over {df['case_id'].nunique()} cases")

    # Pick 10 cases: 4 by Vista BSR strength, 3 by BSR variability, 3 by overall MAE
    case_stats = df.groupby("case_id").agg(
        n=("target", "size"),
        sr_median=("bis_sr_oracle", "median"),
        sr_max=("bis_sr_oracle", "max"),
        sr_std=("bis_sr_oracle", "std"),
        bis_median=("target", "median"),
    ).reset_index()

    # heavy deep:
    heavy = case_stats.sort_values("sr_median", ascending=False).head(4)["case_id"].tolist()
    # transient deep (high std):
    transient = case_stats[~case_stats["case_id"].isin(heavy)].sort_values("sr_std", ascending=False).head(3)["case_id"].tolist()
    # mid surgical (low sr, ~50 BIS median):
    chosen = set(heavy + transient)
    rest = case_stats[~case_stats["case_id"].isin(chosen)]
    mid = rest[(rest["bis_median"] > 35) & (rest["bis_median"] < 55)].sort_values("n", ascending=False).head(3)["case_id"].tolist()
    cases = (heavy + transient + mid)[:10]

    print(f"Selected cases: {cases}")
    for cid in cases:
        s = case_stats[case_stats["case_id"] == cid].iloc[0]
        print(f"  case {cid}: N={s['n']:,}  SR median={s['sr_median']:.1f}  max={s['sr_max']:.1f}  "
              f"BIS median={s['bis_median']:.1f}")

    fig, axes = plt.subplots(5, 2, figsize=(20, 18))
    for ax, cid in zip(axes.flat, cases):
        sub = df[df["case_id"] == cid].sort_values("time_sec").reset_index(drop=True)
        t = sub["time_sec"].values
        # Detectors
        ax.plot(t, sub["bis_sr_oracle"], color="black", lw=1.3, alpha=0.95, label="Vista BSR (oracle)")
        ax.plot(t, sub["bsr_quazi"],     color="tab:green", lw=0.8, alpha=0.75, label="bsr_quazi")
        ax.plot(t, sub["bsr_paper"],     color="tab:red", lw=0.8, alpha=0.75, label="bsr_paper")
        ax.plot(t, sub["openbsr"],       color="tab:blue", lw=0.8, alpha=0.75, label="openbsr")
        ax.set_ylim(-2, 100)
        ax.set_ylabel("BSR/SR %", fontsize=9)

        # BIS on twin axis
        ax2 = ax.twinx()
        ax2.plot(t, sub["target"], color="tab:orange", lw=0.7, alpha=0.85, label="actual BIS")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("BIS", color="tab:orange", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax.set_title(f"case {cid}  ·  SR median={sub['bis_sr_oracle'].median():.1f}  "
                     f"BIS median={sub['target'].median():.1f}", fontsize=10)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.grid(alpha=0.2)

        if cid == cases[0]:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7, framealpha=0.9)

    plt.tight_layout()
    out = RESULTS / "bsr_detector_comparison_10cases.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nSaved: {out.name}")
    return out


if __name__ == "__main__":
    main()
