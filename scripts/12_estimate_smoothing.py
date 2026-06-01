"""Estimate per-case BIS Vista smoothing window from the actual BIS track.

BIS Vista lets the operator pick 15-, 30-, or 45-second smoothing of
the index. We don't have that setting recorded in VitalDB, so we
estimate it from the BIS time-series itself:

  * **Cross-correlation between predict_bis (computed from raw EEG)
    and actual BIS.** If actual = causal trailing average of an
    underlying "raw" BIS over W seconds, the peak of cross-correlation
    lag(predict_bis, actual) should sit around W/2.

  * **Maximum-step heuristic.** A trailing uniform mean of window W
    on a signal x[t] caps |y[t] - y[t-1]| at |x[t] - x[t-W]| / W,
    which limits the apparent slew rate. Cases with smaller observed
    slew rates have larger W.

  * **ACF first-zero / e-fold time.** The autocorrelation of a
    trailing average has triangular shape decaying linearly to zero
    at lag W. Estimating the first-zero crossing of ACF(actual)
    gives a direct W estimate.

Outputs per case (printed + parquet):
  caseid, N, lag_xcorr_peak, W_from_lag, W_from_acf_zero,
  W_from_max_step, p99_dBIS_per_s
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg.cohort import load_case, preprocess_eeg
from openeeg import predict_bis

RESULTS = Path(__file__).resolve().parents[1] / "results"


def w_from_max_step(bis: np.ndarray) -> float:
    """Heuristic: if y = trailing-mean(x, W), then
    max|Δy| ≈ (range x) / W ⇒ W ≈ range / p99|Δy|."""
    db = np.diff(bis)
    db = db[np.isfinite(db)]
    if len(db) < 30:
        return float("nan")
    p99 = np.percentile(np.abs(db), 99)
    rng = float(np.nanmax(bis) - np.nanmin(bis))
    if p99 < 0.05 or rng < 20:
        return float("nan")
    return rng / p99


def w_from_acf_zero(bis: np.ndarray, max_lag: int = 80) -> float:
    """Window estimate from the first ACF crossing of 0 (or small value)."""
    x = bis - np.nanmean(bis)
    x = x[np.isfinite(x)]
    if len(x) < max_lag * 4:
        return float("nan")
    var = float(np.var(x))
    if var <= 0:
        return float("nan")
    acf = np.empty(max_lag)
    for k in range(max_lag):
        acf[k] = float(np.mean(x[k:] * x[:len(x) - k])) / var
    below = np.where(acf < 0.1)[0]
    if len(below) == 0:
        return float("nan")
    return float(below[0])


def w_from_xcorr_peak(pred: np.ndarray, actual: np.ndarray, max_lag: int = 80) -> tuple[float, float]:
    """Lag of peak cross-correlation between pred (presumed raw-like)
    and actual (presumed smoothed). Returns (peak_lag, 2 * peak_lag).
    Positive peak_lag means actual lags pred."""
    p = pred - np.nanmean(pred)
    a = actual - np.nanmean(actual)
    n = min(len(p), len(a))
    if n < max_lag * 4:
        return float("nan"), float("nan")
    p, a = p[:n], a[:n]
    # only finite pairs
    m = np.isfinite(p) & np.isfinite(a)
    p, a = p[m], a[m]
    if len(p) < max_lag * 4:
        return float("nan"), float("nan")
    np_, na = p / np.linalg.norm(p), a / np.linalg.norm(a)
    lags = np.arange(-max_lag, max_lag + 1)
    xc = np.empty(len(lags))
    for i, k in enumerate(lags):
        if k >= 0:
            xc[i] = float(np.dot(np_[:len(np_) - k], na[k:]))
        else:
            xc[i] = float(np.dot(np_[-k:], na[:len(na) + k]))
    peak = int(np.argmax(xc))
    return float(lags[peak]), 2.0 * float(lags[peak])


def main():
    df = pd.read_parquet(RESULTS / "features_val_n100_v2.parquet")
    print(f"Loaded val parquet: {len(df):,} rows, "
          f"{df['case_id'].nunique()} cases")

    # Vectorized model prediction on cached features (much faster than
    # re-running predict_bis from raw EEG per case).
    import lightgbm as lgb
    booster = lgb.Booster(
        model_file=str(Path(__file__).resolve().parents[1]
                       / "openeeg" / "models" / "predict_bis_v1.txt"))
    feat_cols = [c for c in df.columns
                 if c not in ("target", "sqi", "case_id", "time_sec")
                 and not c.startswith("bis_")]
    df["pred"] = np.clip(booster.predict(df[feat_cols].values), 0.0, 100.0)
    print(f"  Prediction done over all rows.")

    caseids = sorted(df["case_id"].unique())
    rows = []
    for i, cid in enumerate(caseids, 1):
        sub = df[df["case_id"] == cid]
        # Filter SQI>=80 already applied during feature extraction
        a = sub["target"].values
        p = sub["pred"].values
        if len(a) < 200:
            continue

        db = np.diff(a)
        p99 = float(np.percentile(np.abs(db[np.isfinite(db)]), 99)) if len(db) > 30 else float("nan")
        w_step = w_from_max_step(a)
        w_acf  = w_from_acf_zero(a, max_lag=80)
        peak, w_lag = w_from_xcorr_peak(p, a, max_lag=80)

        rows.append({
            "caseid": int(cid),
            "N": len(sub),
            "p99_dBIS_per_s": p99,
            "W_from_max_step": w_step,
            "W_from_acf_zero": w_acf,
            "lag_xcorr_peak_s": peak,
            "W_from_lag": w_lag,
        })
        if i % 25 == 0:
            print(f"  {i}/{len(caseids)} done")

    out = pd.DataFrame(rows)
    out.to_parquet(RESULTS / "smoothing_estimates.parquet", index=False)
    out.to_csv(RESULTS / "smoothing_estimates.csv", index=False)
    print(f"\nWrote smoothing_estimates.{{parquet,csv}}: {len(out)} cases")

    # Summary
    print("\n=== Distribution of W estimates (seconds) ===")
    for col in ["W_from_max_step", "W_from_acf_zero", "W_from_lag", "lag_xcorr_peak_s", "p99_dBIS_per_s"]:
        v = out[col].dropna()
        if len(v) == 0:
            print(f"  {col}: no data")
            continue
        print(f"  {col:<22s}  N={len(v):3d}   median={v.median():6.1f}   "
              f"p25={v.quantile(0.25):6.1f}   p75={v.quantile(0.75):6.1f}   "
              f"min={v.min():6.1f}   max={v.max():6.1f}")

    # Group: how many cases land near 15 / 30 / 45?
    print("\n=== Cases clustered around 15/30/45 (by W_from_acf_zero) ===")
    w = out["W_from_acf_zero"].dropna()
    bins = [(0, 12), (12, 22), (22, 38), (38, 52), (52, 200)]
    for lo, hi in bins:
        n = ((w >= lo) & (w < hi)).sum()
        print(f"  [{lo:>3d}, {hi:>3d}) s : {n:3d} cases ({100*n/len(w):.0f}%)")


if __name__ == "__main__":
    main()
