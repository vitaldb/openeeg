"""Phase 3h — train on the W=15 sub-cohort only.

The oracle-W cache showed ~80% of VitalDB BIS cases sit at Vista's
15 s smoothing setting. Training and evaluating on the homogeneous
W=15 sub-cohort isolates that population.

Pipeline:
  1. Filter both train and val parquets to cases whose oracle_W == 15.
  2. EMA-desmooth target at W=15 (one fixed window).
  3. Train LightGBM with optional deep-BIS sample weighting.
  4. Evaluate raw + re-smoothed at 15 s vs the original actual BIS.
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
MODEL_DIR = RESULTS / "w15_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

W = 15  # fixed for this experiment


def ema(x, w):
    a = 2.0 / (w + 1.0)
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    return y


def desmooth_ema(y, w):
    alpha = 2.0 / (w + 1.0)
    x = np.empty_like(y)
    x[0] = y[0]
    x[1:] = (y[1:] - (1.0 - alpha) * y[:-1]) / alpha
    return np.clip(x, 0.0, 100.0)


def filter_and_desmooth(df, w15_cases, target_col="target", desm_col="target_desm"):
    keep = df["case_id"].isin(w15_cases)
    df = df[keep].reset_index(drop=True)
    df[desm_col] = np.nan
    for cid, sub in df.groupby("case_id"):
        idx = sub.index.to_numpy()
        a = df[target_col].values[idx]
        mean_fill = float(np.nanmean(a)) if not np.isnan(np.nanmean(a)) else 50.0
        a_filled = np.where(np.isnan(a), mean_fill, a)
        df.loc[idx, desm_col] = desmooth_ema(a_filled, W)
    return df


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--weight-deep", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    # Read W caches and select cases with oracle_W == 15
    w_train = pd.read_csv(RESULTS / "oracle_W_train.csv")
    w_val   = pd.read_csv(RESULTS / "oracle_W_val.csv")
    w15_train = set(w_train.loc[w_train["oracle_W"] == 15, "case_id"].astype(int))
    w15_val   = set(w_val.loc[w_val["oracle_W"] == 15, "case_id"].astype(int))
    print(f"W=15 cases:  train={len(w15_train)}/{len(w_train)}  val={len(w15_val)}/{len(w_val)}")

    train_df = pd.read_parquet(RESULTS / "features_train_n500_v3.parquet")
    val_df   = pd.read_parquet(RESULTS / "features_val_n100_v3.parquet")
    print(f"Before filter: train rows={len(train_df):,}  val rows={len(val_df):,}")

    train_df = filter_and_desmooth(train_df, w15_train)
    val_df   = filter_and_desmooth(val_df, w15_val)
    print(f"After filter:  train rows={len(train_df):,}  val rows={len(val_df):,}")

    base = ["openibis_paper", "openibis_quazi", "openibis_quazi_30s",
            "bsr_paper", "bsr_quazi", "sef95", "bcsef", "beta_ratio", "emg_proxy",
            "p_delta", "p_theta", "p_alpha", "p_beta", "p_lowgamma", "spectral_entropy"]
    extra = ["openibis_quazi_5s", "openibis_quazi_10s", "openibis_quazi_60s",
             "openibis_quazi_dt", "openibis_quazi_30s_dt", "sef95_dt", "emg_proxy_dt"]
    feat_cols = base + [c for c in extra if c in train_df.columns]
    print(f"Features ({len(feat_cols)})")

    X  = train_df[feat_cols].values
    y  = train_df["target_desm"].values
    Xv = val_df[feat_cols].values
    yv_desm = val_df["target_desm"].values
    actual_orig = val_df["target"].values

    # Sample weighting on deep epochs (using ORIGINAL target for the mask)
    y_orig = train_df["target"].values
    weight = np.where(y_orig < 30, args.weight_deep, 1.0) if args.weight_deep != 1.0 else None
    n_deep = int((y_orig < 30).sum())
    print(f"  Deep epochs in filtered train: {n_deep:,} ({100*n_deep/len(y_orig):.2f}%)")
    if args.weight_deep != 1.0:
        print(f"  Sample weighting: deep × {args.weight_deep}")

    params = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=5, verbose=-1,
                  num_threads=args.threads)
    dtrain = lgb.Dataset(X, label=y, weight=weight, feature_name=feat_cols)
    dval = lgb.Dataset(Xv, label=yv_desm, feature_name=feat_cols, reference=dtrain)
    print(f"\nTraining...")
    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=2000,
                        valid_sets=[dtrain, dval], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    out = MODEL_DIR / f"lgbm_{args.label}.txt"
    booster.save_model(str(out))
    print(f"  best_iter={booster.best_iteration}  {time.time()-t0:.1f}s  -> {out.name}")

    pred_raw = np.clip(booster.predict(Xv), 0, 100)
    pred_sm15 = np.empty_like(pred_raw)
    for cid, sub in val_df.groupby("case_id"):
        idx = sub.index.to_numpy()
        pred_sm15[idx] = ema(pred_raw[idx], 15.0)

    print(f"\n=== Val (W=15 sub-cohort, {val_df['case_id'].nunique()} cases) vs ORIGINAL actual ===")
    print(f"{'variant':<40s}  {'MAE':>5s}  {'r':>6s}  {'Lin_rc':>7s}")
    for name, p in [("raw model output", pred_raw),
                    ("+ EMA(15s) post-smooth", pred_sm15)]:
        mae, r, rc = safe(p, actual_orig)
        print(f"  {name:<38s}  {mae:5.2f}  {r:6.3f}  {rc:7.3f}")

    print(f"\n=== Per-regime MAE ===")
    print(f"{'variant':<40s}  " + "  ".join(f"{l:>7s}" for l in LEE_BIN_LABELS))
    for name, p in [("raw", pred_raw), ("EMA(15s)", pred_sm15)]:
        r = per_regime(actual_orig, p)
        print(f"  {name:<38s}  " + "  ".join(
            f"{r[k]:>7.2f}" if not np.isnan(r[k]) else f"{'nan':>7s}"
            for k in LEE_BIN_LABELS))

    imp = booster.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1][:10]
    print(f"\nTop 10 features:")
    for i in order:
        print(f"  {feat_cols[i]:<26s}  {imp[i]:>12.1f}")


if __name__ == "__main__":
    main()
