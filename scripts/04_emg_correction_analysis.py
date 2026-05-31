"""Phase 2A — empirical fit + cohort evaluation of EMG correction.

Two stages on the 100-case val cohort:

  1. Aggregate (residual, EMG) pairs across all valid epochs and fit
     ``residual ≈ α · max(EMG − 34, 0)`` to derive the correction
     slope used by :func:`openeeg.emg_correct`.

  2. Re-evaluate every openibis variant with and without
     ``emg_correct`` applied — per-regime MAE on Lee bins, global
     MAE / r / Lin's rc per case, summarised as cohort means.

Run after the cache is populated (e.g. by 02_cohort_benchmark.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg import openibis, emg_correct
from openeeg.cohort import load_case, preprocess_eeg
from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS, evaluate

CACHE = Path("C:/temp/openeeg_cache")
SQI_THRESH = 80


def collect_cohort():
    caseids = sorted(int(p.stem) for p in CACHE.glob("*.vital"))
    print(f"Loading {len(caseids)} cached cases...")
    cases = []
    for cid in caseids:
        case = load_case(cid, cache_dir=CACHE)
        if case is None:
            continue
        eeg = preprocess_eeg(case["eeg"])
        # Best-performing variant from Phase 1: quazi + paper
        pred = openibis(eeg, bsr="quazi", deep="paper")[::2]
        bis = case["bis"]; sqi = case["sqi"]; emg = case["emg"]
        n = min(len(bis), len(pred), len(sqi), len(emg))
        if n < 60:
            continue
        a, p, s, e = bis[:n], pred[:n], sqi[:n], emg[:n]
        valid = (
            ~np.isnan(a) & ~np.isnan(p) & ~np.isnan(s) & ~np.isnan(e)
            & (s >= SQI_THRESH)
        )
        if valid.sum() < 100:
            continue
        cases.append({"caseid": cid, "a": a, "p": p, "emg": e, "valid": valid})
    print(f"  {len(cases)} cases retained")
    return cases


def fit_emg_correction(cases):
    """Pool epochs across all cases, fit residual vs EMG excess."""
    resid_all, excess_all = [], []
    for c in cases:
        v = c["valid"]
        resid_all.append(c["p"][v] - c["a"][v])
        excess_all.append(np.maximum(c["emg"][v] - 34.0, 0.0))
    resid = np.concatenate(resid_all)
    excess = np.concatenate(excess_all)
    mask = excess > 0
    slope, intercept = np.polyfit(excess[mask], resid[mask], 1)
    print(f"\nEMG correction fit (N={mask.sum():,} epochs with EMG>34dB):")
    print(f"  residual = {slope:+.3f} · (EMG − 34) + {intercept:+.3f}")
    print(f"  pearson r(resid, excess) on EMG>34: {np.corrcoef(excess[mask], resid[mask])[0,1]:+.3f}")
    return slope


def per_regime_mae(actual, pred, valid):
    out = {}
    a, p = actual[valid], pred[valid]
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        m = (a >= lo) & (a < hi)
        out[lbl] = float(np.mean(np.abs(p[m] - a[m]))) if m.sum() > 10 else float("nan")
    return out


def evaluate_cohort(cases, *, label, apply_corr):
    per_case = []
    for c in cases:
        if apply_corr:
            corrected = emg_correct(c["p"], c["emg"])
        else:
            corrected = c["p"]
        a, p, v = c["a"], corrected, c["valid"]
        g = evaluate(a, p, v)["global"]
        per_case.append({
            "caseid": c["caseid"],
            "mae": g["mae"],
            "r": g["r"],
            "lin_rc": g["lin_rc"],
            "regime": per_regime_mae(a, p, v),
        })
    mae_arr = np.array([x["mae"] for x in per_case])
    r_arr = np.array([x["r"] for x in per_case if not np.isnan(x["r"])])
    rc_arr = np.array([x["lin_rc"] for x in per_case if not np.isnan(x["lin_rc"])])
    regime_means = {
        lbl: float(np.nanmean([x["regime"][lbl] for x in per_case]))
        for lbl in LEE_BIN_LABELS
    }
    print(f"\n=== {label} ===")
    print(f"  MAE mean={mae_arr.mean():.2f}  median={np.median(mae_arr):.2f}")
    print(f"  r   mean={r_arr.mean():.3f}  median={np.median(r_arr):.3f}")
    print(f"  Lin's rc mean={rc_arr.mean():.3f}")
    print(f"  Per-regime MAE: " + "  ".join(
        f"{lbl}={regime_means[lbl]:.2f}" for lbl in LEE_BIN_LABELS))
    return per_case, regime_means


def main():
    cases = collect_cohort()
    fit_emg_correction(cases)

    base, base_reg = evaluate_cohort(cases, label="Baseline (openibis quazi/paper)", apply_corr=False)
    corr, corr_reg = evaluate_cohort(cases, label="+ emg_correct()", apply_corr=True)

    print("\n=== Improvement summary ===")
    base_mae = np.array([x["mae"] for x in base])
    corr_mae = np.array([x["mae"] for x in corr])
    delta = corr_mae - base_mae
    print(f"  Overall MAE: {base_mae.mean():.2f} → {corr_mae.mean():.2f} "
          f"(Δ={delta.mean():+.2f}, {(delta<0).sum()}/{len(delta)} cases improved)")
    for lbl in LEE_BIN_LABELS:
        d = corr_reg[lbl] - base_reg[lbl]
        print(f"  {lbl:>7s}:  {base_reg[lbl]:5.2f} → {corr_reg[lbl]:5.2f}  (Δ={d:+.2f})")


if __name__ == "__main__":
    main()
