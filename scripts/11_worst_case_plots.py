"""Visualise the worst-error val cases for predict_bis().

Rank all 100 val cases by per-case MAE under predict_bis(), then plot
the top-N: actual BIS, predicted BIS, EEG snippet, BSR detectors,
EMG, and SEF over time, with SQI<80 windows shaded grey.

Output: results/worst_case_<caseid>.png
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg.cohort import load_case, preprocess_eeg
from openeeg import predict_bis

RESULTS = Path(__file__).resolve().parents[1] / "results"
VAL_PARQUET = RESULTS / "features_val_n100_v2.parquet"


def find_worst(top_n: int):
    df = pd.read_parquet(VAL_PARQUET)
    # We can use the bundled model's predictions by computing them once
    # per case. The fast path: re-use already-stored features and run
    # the LightGBM directly.
    import lightgbm as lgb
    booster = lgb.Booster(
        model_file=str(Path(__file__).resolve().parents[1]
                       / "openeeg" / "models" / "predict_bis_v1.txt"))
    feat_cols = [c for c in df.columns
                 if c not in ("target", "sqi", "case_id", "time_sec")
                 and not c.startswith("bis_")]
    pred = np.clip(booster.predict(df[feat_cols].values), 0.0, 100.0)
    df["pred"] = pred

    rows = []
    for cid, sub in df.groupby("case_id"):
        mae = float(np.mean(np.abs(sub["pred"] - sub["target"])))
        r = float(np.corrcoef(sub["pred"], sub["target"])[0, 1]) if sub["pred"].std() > 0 else float("nan")
        rows.append((int(cid), len(sub), mae, r))
    rows.sort(key=lambda x: x[2], reverse=True)
    print(f"{'rank':>4s}  {'case':>5s}  {'N':>5s}  {'MAE':>6s}  {'r':>6s}")
    for i, (cid, n, mae, r) in enumerate(rows[:top_n], 1):
        print(f"{i:>4d}  {cid:>5d}  {n:>5d}  {mae:6.2f}  {r:6.3f}")
    return [cid for cid, *_ in rows[:top_n]]


def plot_case(caseid: int, outpath: Path):
    case = load_case(caseid)
    if case is None:
        print(f"case {caseid}: load failed")
        return
    eeg = preprocess_eeg(case["eeg"])
    fs = case["fs"]

    pred = predict_bis(eeg)               # 1 Hz
    actual = case["bis"]
    sqi    = case["sqi"]
    sr_v   = case["sr"]
    emg_v  = case["emg"]
    sef_v  = case["sef"]

    n = min(len(pred), len(actual), len(sqi), len(sr_v), len(emg_v), len(sef_v))
    pred, actual, sqi, sr_v, emg_v, sef_v = (
        pred[:n], actual[:n], sqi[:n], sr_v[:n], emg_v[:n], sef_v[:n])
    t = np.arange(n)

    valid_mask = ~np.isnan(actual) & ~np.isnan(pred) & ~np.isnan(sqi) & (sqi >= 80)
    if valid_mask.sum() > 1:
        mae = float(np.mean(np.abs(pred[valid_mask] - actual[valid_mask])))
        r = float(np.corrcoef(pred[valid_mask], actual[valid_mask])[0, 1])
    else:
        mae, r = float("nan"), float("nan")

    # Low-SQI shade intervals
    low_sqi = np.isnan(sqi) | (sqi < 80)
    shade = []
    in_low = False
    start = 0
    for i in range(n):
        if low_sqi[i] and not in_low:
            start, in_low = i, True
        elif not low_sqi[i] and in_low:
            shade.append((start, i)); in_low = False
    if in_low:
        shade.append((start, n))

    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6, 1, 1, 1]})

    # (1) EEG snippet (downsampled visualisation: 1 Hz subsample)
    t_eeg = np.arange(len(eeg)) / fs
    axes[0].plot(t_eeg[::fs], eeg[::fs], lw=0.3, color="steelblue")
    axes[0].set_ylabel("EEG (µV)")
    axes[0].set_title(f"Case {caseid}  —  MAE={mae:.2f}  r={r:.3f}  (SQI≥80 only)")
    axes[0].set_ylim(-60, 60)
    for s0, s1 in shade:
        axes[0].axvspan(s0, s1, color="gray", alpha=0.2)

    # (2) Actual BIS vs predict_bis
    axes[1].plot(t, actual, color="black", lw=1.0, alpha=0.85, label="Actual BIS")
    axes[1].plot(t, pred,  color="tab:red", lw=1.0, alpha=0.75, label="predict_bis")
    for s0, s1 in shade:
        axes[1].axvspan(s0, s1, color="gray", alpha=0.2)
    axes[1].axhspan(0, 21, color="purple", alpha=0.05, label="Lee 0-21 (deep)")
    axes[1].axhspan(78, 98, color="orange", alpha=0.05, label="Lee 78-98 (awake)")
    axes[1].set_ylabel("BIS")
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc="upper right", fontsize=8)

    # (3) Residual (pred − actual)
    resid = pred - actual
    axes[2].plot(t, resid, color="tab:purple", lw=0.8, alpha=0.75)
    axes[2].axhline(0, color="black", lw=0.5)
    axes[2].fill_between(t, 0, resid, where=resid > 0, color="tab:red", alpha=0.25, label="pred > actual")
    axes[2].fill_between(t, 0, resid, where=resid < 0, color="tab:blue", alpha=0.25, label="pred < actual")
    for s0, s1 in shade:
        axes[2].axvspan(s0, s1, color="gray", alpha=0.2)
    axes[2].set_ylabel("pred − actual")
    axes[2].legend(loc="upper right", fontsize=8)

    # (4) BIS Vista SR vs our BSR detectors
    from openeeg.openibis import bsr as openibis_bsr
    bsr_p = openibis_bsr(eeg, kind="paper")[::2][:n]
    bsr_q = openibis_bsr(eeg, kind="quazi")[::2][:n]
    axes[3].plot(t, sr_v, color="black", lw=1.0, alpha=0.85, label="BIS/SR (Vista)")
    axes[3].plot(t, bsr_p, color="tab:red", lw=0.8, alpha=0.6, label="bsr_paper")
    axes[3].plot(t, bsr_q, color="tab:green", lw=0.8, alpha=0.6, label="bsr_quazi")
    axes[3].axhline(49.8, color="purple", lw=0.5, ls="--", label="Lee deep gate (49.8%)")
    for s0, s1 in shade:
        axes[3].axvspan(s0, s1, color="gray", alpha=0.2)
    axes[3].set_ylabel("BSR / SR (%)")
    axes[3].legend(loc="upper right", fontsize=8)

    # (5) BIS/EMG and BIS/SEF
    ax5 = axes[4]
    ax5.plot(t, emg_v, color="tab:orange", lw=0.9, label="BIS/EMG (dB)")
    ax5.axhline(34.2, color="tab:orange", lw=0.5, ls="--", alpha=0.5, label="Lee 34.2 dB")
    ax5.set_ylabel("EMG (dB)")
    ax5b = ax5.twinx()
    ax5b.plot(t, sef_v, color="tab:cyan", lw=0.9, label="BIS/SEF (Hz)")
    ax5b.axhline(20.2, color="tab:cyan", lw=0.5, ls="--", alpha=0.5, label="Lee 20.2 Hz")
    ax5b.set_ylabel("SEF (Hz)")
    for s0, s1 in shade:
        ax5.axvspan(s0, s1, color="gray", alpha=0.2)
    ax5.set_xlabel("Time (s)")
    ax5.legend(loc="upper left", fontsize=8)
    ax5b.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    print(f"  saved {outpath.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5, help="Plot the top-N worst cases.")
    args = ap.parse_args()
    print("Ranking val cases by per-case MAE under predict_bis()...")
    worst = find_worst(args.top)
    print()
    RESULTS.mkdir(exist_ok=True)
    for cid in worst:
        plot_case(cid, RESULTS / f"worst_case_{cid}.png")


if __name__ == "__main__":
    main()
