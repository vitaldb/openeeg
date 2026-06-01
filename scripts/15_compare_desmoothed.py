"""Compare baseline + 4 desmooth variants (EMA/Wiener × per-case-W / fixed-15s)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS, lin_concordance

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODEL_DIR = RESULTS / "desmooth_models"


def estimate_W(bis, lo=10, hi=60):
    db = np.diff(bis)
    db = db[np.isfinite(db)]
    if len(db) < 30:
        return 30
    p99 = float(np.percentile(np.abs(db), 99))
    rng = float(np.nanmax(bis) - np.nanmin(bis))
    if p99 < 0.05 or rng < 20:
        return 30
    return int(np.clip(round(rng / p99), lo, hi))


def smooth_ema(x, W):
    alpha = 2.0 / (W + 1.0)
    y = np.empty(len(x))
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]
    return y


def smooth_uniform(x, W):
    kernel = np.ones(W) / W
    return np.convolve(x, kernel, mode="full")[: len(x)]


def safe(p, a):
    m = ~np.isnan(p) & ~np.isnan(a)
    if m.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(np.abs(p[m] - a[m]))),
            float(np.corrcoef(p[m], a[m])[0, 1]),
            lin_concordance(p[m], a[m]))


def main():
    val_df = pd.read_parquet(RESULTS / "features_val_n100_v2.parquet")
    feat_cols = [c for c in val_df.columns
                 if c not in ("target", "sqi", "case_id", "time_sec")
                 and not c.startswith("bis_")]
    X_val = val_df[feat_cols].values
    actual = val_df["target"].values

    # Per-case W on val (using same heuristic)
    W_val = {}
    for cid, sub in val_df.groupby("case_id"):
        a = sub["target"].values
        a = np.where(np.isnan(a), float(np.nanmean(a)) if not np.isnan(np.nanmean(a)) else 50.0, a)
        W_val[int(cid)] = estimate_W(a)

    # Load 5 models
    tags = [
        "baseline_smoothed",
        "ema_desmooth", "wiener_desmooth",
        "ema_desmooth_fixed15", "wiener_desmooth_fixed15",
    ]
    models = {t: lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_{t}.txt")) for t in tags}

    # Predictions
    preds = {f"{t}_raw": np.clip(m.predict(X_val), 0, 100) for t, m in models.items()}

    # Re-smooth at per-case W or fixed 15s
    for base in ("ema_desmooth", "wiener_desmooth"):
        smoother = smooth_ema if "ema" in base else smooth_uniform
        out = np.empty_like(preds[f"{base}_raw"])
        for cid, sub in val_df.groupby("case_id"):
            idx = sub.index.to_numpy()
            W = W_val.get(int(cid), 30)
            out[idx] = smoother(preds[f"{base}_raw"][idx], W)
        preds[f"{base}_resmoothed"] = out

    for base in ("ema_desmooth_fixed15", "wiener_desmooth_fixed15"):
        smoother = smooth_ema if "ema" in base else smooth_uniform
        out = np.empty_like(preds[f"{base}_raw"])
        for cid, sub in val_df.groupby("case_id"):
            idx = sub.index.to_numpy()
            out[idx] = smoother(preds[f"{base}_raw"][idx], 15)
        preds[f"{base}_resmoothed_15s"] = out

    order = [
        ("baseline_smoothed_raw", "baseline  (predict_bis_v1 equivalent)"),
        ("ema_desmooth_raw", "EMA  per-case W   (raw pred)"),
        ("ema_desmooth_resmoothed", "EMA  per-case W   (+ re-smoothed)"),
        ("wiener_desmooth_raw", "Wiener per-case W (raw pred)"),
        ("wiener_desmooth_resmoothed", "Wiener per-case W (+ re-smoothed)"),
        ("ema_desmooth_fixed15_raw", "EMA  fixed W=15s  (raw pred)"),
        ("ema_desmooth_fixed15_resmoothed_15s", "EMA  fixed W=15s  (+ re-smoothed 15s)"),
        ("wiener_desmooth_fixed15_raw", "Wiener fixed W=15s (raw pred)"),
        ("wiener_desmooth_fixed15_resmoothed_15s", "Wiener fixed W=15s (+ re-smoothed 15s)"),
    ]

    print("=== Val comparison vs ORIGINAL smoothed actual BIS ===")
    print(f"{'variant':<44s}  {'MAE':>5s}  {'r':>6s}  {'Lin_rc':>7s}")
    for k, lbl in order:
        mae, r, rc = safe(preds[k], actual)
        print(f"  {lbl:<42s}  {mae:5.2f}  {r:6.3f}  {rc:7.3f}")

    print(f"\n=== Per-regime MAE (vs actual BIS) ===")
    print(f"{'variant':<44s}  " + "  ".join(f"{l:>6s}" for l in LEE_BIN_LABELS))
    for k, lbl in order:
        p = preds[k]
        m = ~np.isnan(p) & ~np.isnan(actual)
        a_v, p_v = actual[m], p[m]
        parts = []
        for lab, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
            mm = (a_v >= lo) & (a_v < hi)
            parts.append(f"{np.mean(np.abs(p_v[mm] - a_v[mm])):6.2f}" if mm.sum() > 10
                         else f"{'nan':>6s}")
        print(f"  {lbl:<42s}  " + "  ".join(parts))

    print(f"\nW distribution (val): median={int(np.median(list(W_val.values())))}  "
          f"p25={int(np.percentile(list(W_val.values()), 25))}  "
          f"p75={int(np.percentile(list(W_val.values()), 75))}")


if __name__ == "__main__":
    main()
