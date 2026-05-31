"""Diagnose per-case bias of openibis predictions vs commercial BIS.

For each cached case, reports:
  * Linear regression slope/intercept of pred vs actual
  * Per-regime mean bias (pred − actual)
  * Correlation of residual with BIS/EMG (tests the EMG-contamination hypothesis)
  * Effect of two candidate Phase 2 interventions:
       - 30-second centred smoothing on the prediction
       - Cohort-mean affine calibration ``corrected = (pred − 12.56) / 0.80``

Reproduces the smoke results that motivate Phase 2 scope.

Usage::

    python scripts/03_bias_analysis.py            # all cached cases
    python scripts/03_bias_analysis.py 8 48 68    # explicit subset
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg.cohort import load_case, preprocess_eeg
from openeeg.openibis import openibis
from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS

CACHE = Path(__file__).resolve().parents[1] / ".cache" / "vital"


def _mavg(x: np.ndarray, w: int) -> np.ndarray:
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def analyse(caseid: int) -> dict | None:
    case = load_case(caseid, cache_dir=CACHE)
    if case is None:
        return None
    eeg = preprocess_eeg(case["eeg"])
    pred = openibis(eeg, bsr="quazi", deep="ellerkmann")[::2]
    a = case["bis"]; s = case["sqi"]; e = case["emg"]
    n = min(len(a), len(pred), len(s), len(e))
    a, p, s, e = a[:n], pred[:n], s[:n], e[:n]
    valid = ~np.isnan(a) & ~np.isnan(p) & ~np.isnan(s) & ~np.isnan(e) & (s >= 80)
    if valid.sum() < 100:
        return None
    av, pv, ev = a[valid], p[valid], e[valid]

    slope, intercept = np.polyfit(av, pv, 1)
    p15 = _mavg(p, 15)[valid]
    p_cal = (pv - 12.56) / 0.80
    residual = pv - av

    per_regime_bias = {}
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        m = (av >= lo) & (av < hi)
        per_regime_bias[lbl] = float((pv[m] - av[m]).mean()) if m.sum() > 10 else float("nan")

    return {
        "caseid": caseid,
        "n": int(valid.sum()),
        "slope": float(slope),
        "intercept": float(intercept),
        "actual_p10_p90": (float(np.percentile(av, 10)), float(np.percentile(av, 90))),
        "pred_p10_p90":   (float(np.percentile(pv, 10)), float(np.percentile(pv, 90))),
        "mae_baseline": float(np.mean(np.abs(pv - av))),
        "mae_smooth15": float(np.mean(np.abs(p15 - av))),
        "mae_calibrated": float(np.mean(np.abs(p_cal - av))),
        "r_emg_residual": float(np.corrcoef(ev, residual)[0, 1]) if ev.std() > 0 else float("nan"),
        "per_regime_bias": per_regime_bias,
    }


def main() -> None:
    if len(sys.argv) > 1:
        cids = [int(x) for x in sys.argv[1:]]
    else:
        cids = sorted(int(p.stem) for p in CACHE.glob("*.vital"))
    print(f"Analysing {len(cids)} cached cases: {cids}\n")

    print(f"{'caseid':>6s}  {'N':>5s}  {'slope':>5s}  {'inter':>5s}  "
          f"{'MAE_base':>8s}  {'MAE_sm15':>8s}  {'MAE_cal':>8s}  {'r_emg_resid':>11s}")
    rows = []
    for cid in cids:
        r = analyse(cid)
        if r is None:
            print(f"{cid:6d}  skipped")
            continue
        rows.append(r)
        print(f"{cid:6d}  {r['n']:5d}  {r['slope']:5.2f}  {r['intercept']:+5.1f}  "
              f"{r['mae_baseline']:8.2f}  {r['mae_smooth15']:8.2f}  {r['mae_calibrated']:8.2f}  "
              f"{r['r_emg_residual']:+11.3f}")

    print(f"\nMean MAE:  baseline={np.mean([r['mae_baseline'] for r in rows]):5.2f}  "
          f"smooth15={np.mean([r['mae_smooth15'] for r in rows]):5.2f}  "
          f"calibrated={np.mean([r['mae_calibrated'] for r in rows]):5.2f}")

    print("\nPer-regime mean bias (pred − actual):")
    print(f"{'caseid':>6s}  " + "  ".join(f"{lbl:>7s}" for lbl in LEE_BIN_LABELS))
    for r in rows:
        print(f"{r['caseid']:6d}  " + "  ".join(
            f"{r['per_regime_bias'][lbl]:+7.2f}" if not np.isnan(r['per_regime_bias'][lbl]) else f"{'nan':>7s}"
            for lbl in LEE_BIN_LABELS))


if __name__ == "__main__":
    main()
