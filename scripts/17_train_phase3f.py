"""Phase 3f training — sample weighting + extended feature set.

Trains LightGBM on the v3 parquet (15 raw + 7 short-context / velocity
features) with optional deep-BIS sample weighting. Compares:

  * baseline    — current predict_bis_v1 (15 features, no weighting)
  * extended    — 22 features (the new short-context/velocity columns),
                   no weighting
  * extended+W3 — 22 features + deep epochs ×3
  * extended+W5 — 22 features + deep epochs ×5

Reports per-regime MAE and feature importance.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS, lin_concordance

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODEL_DIR = RESULTS / "phase3f_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _ema(x: np.ndarray, W: float) -> np.ndarray:
    a = 2.0 / (W + 1.0)
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    return y


def safe(p, a):
    m = ~np.isnan(p) & ~np.isnan(a)
    if m.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(np.abs(p[m] - a[m]))),
            float(np.corrcoef(p[m], a[m])[0, 1]),
            lin_concordance(p[m], a[m]))


def per_regime(actual, pred):
    m = ~np.isnan(actual) & ~np.isnan(pred)
    a, p = actual[m], pred[m]
    out = {}
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        mm = (a >= lo) & (a < hi)
        out[lbl] = float(np.mean(np.abs(p[mm] - a[mm]))) if mm.sum() > 10 else float("nan")
    return out


def train_one(X, y, Xv, yv, feat_cols, label, weight_deep, num_threads):
    weight_train = np.where(y < 30, weight_deep, 1.0) if weight_deep != 1.0 else None
    dtrain = lgb.Dataset(X, label=y, weight=weight_train, feature_name=feat_cols)
    dval = lgb.Dataset(Xv, label=yv, feature_name=feat_cols, reference=dtrain)
    params = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=5, verbose=-1,
                  num_threads=num_threads)
    print(f"\n--- training {label} (n_train={len(X):,}, weight_deep={weight_deep}, threads={num_threads}) ---")
    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=2000,
                        valid_sets=[dtrain, dval], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    out = MODEL_DIR / f"lgbm_{label}.txt"
    booster.save_model(str(out))
    print(f"  best_iter={booster.best_iteration}  {time.time()-t0:.1f}s  -> {out.name}")
    return booster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="Tag used for saved model & feature subset.")
    ap.add_argument("--features", choices=["base15", "extended22"], default="extended22")
    ap.add_argument("--weight-deep", type=float, default=1.0,
                    help="Multiplier for training rows with target<30.")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    train_df = pd.read_parquet(RESULTS / "features_train_n500_v3.parquet")
    val_df   = pd.read_parquet(RESULTS / "features_val_n100_v3.parquet")
    print(f"Train: {len(train_df):,} rows  Val: {len(val_df):,} rows")

    base_cols = ["openibis_paper", "openibis_quazi", "openibis_quazi_30s",
                 "bsr_paper", "bsr_quazi",
                 "sef95", "bcsef", "beta_ratio", "emg_proxy",
                 "p_delta", "p_theta", "p_alpha", "p_beta", "p_lowgamma",
                 "spectral_entropy"]
    extra_cols = ["openibis_quazi_5s", "openibis_quazi_10s", "openibis_quazi_60s",
                  "openibis_quazi_dt", "openibis_quazi_30s_dt",
                  "sef95_dt", "emg_proxy_dt"]
    feat_cols = base_cols if args.features == "base15" else base_cols + extra_cols
    print(f"Features ({len(feat_cols)}): {feat_cols}")

    X = train_df[feat_cols].values
    y = train_df["target"].values
    Xv = val_df[feat_cols].values
    yv = val_df["target"].values

    n_deep = int((y < 30).sum())
    print(f"  Deep epochs in train: {n_deep:,} ({100*n_deep/len(y):.2f}%)")

    booster = train_one(X, y, Xv, yv, feat_cols, args.label,
                        args.weight_deep, args.threads)

    # Evaluate (raw + EMA-15 post-smoothed)
    pred_raw = np.clip(booster.predict(Xv), 0, 100)
    pred_sm = np.empty_like(pred_raw)
    for cid, sub in val_df.groupby("case_id"):
        idx = sub.index.to_numpy()
        pred_sm[idx] = _ema(pred_raw[idx], 15.0)

    actual = yv
    print(f"\n=== Val cohort comparison ===")
    print(f"{'variant':<32s}  {'MAE':>6s}  {'r':>6s}  {'Lin_rc':>7s}")
    for name, p in [("raw model output", pred_raw),
                    ("+ EMA(15s) post-smooth", pred_sm)]:
        mae, r, rc = safe(p, actual)
        print(f"  {name:<30s}  {mae:6.2f}  {r:6.3f}  {rc:7.3f}")

    print(f"\n=== Per-regime MAE (vs actual BIS) ===")
    print(f"{'variant':<32s}  " + "  ".join(f"{l:>7s}" for l in LEE_BIN_LABELS))
    for name, p in [("raw", pred_raw), ("EMA(15s)", pred_sm)]:
        r = per_regime(actual, p)
        print(f"  {name:<30s}  " + "  ".join(
            f"{r[k]:>7.2f}" if not np.isnan(r[k]) else f"{'nan':>7s}"
            for k in LEE_BIN_LABELS))

    imp = booster.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1][:15]
    print(f"\nTop 15 feature importance (gain):")
    for i in order:
        print(f"  {feat_cols[i]:<26s}  {imp[i]:>12.1f}")


if __name__ == "__main__":
    main()
