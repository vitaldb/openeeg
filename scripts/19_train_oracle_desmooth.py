"""Train LightGBM on per-case oracle-W EMA-desmoothed target.

Pipeline:
  1. Load cached oracle W per case (run 18_oracle_W_cache.py first).
  2. For each case, EMA-desmooth ``target`` with that case's W.
  3. Train LightGBM on the desmoothed target.
  4. Evaluate on val:
     (a) raw pred vs desmoothed target (fidelity)
     (b) raw pred re-smoothed at the case's W vs original actual BIS
         (Vista compatibility — main metric)
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
MODEL_DIR = RESULTS / "oracle_desmooth_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def ema(x: np.ndarray, W: float) -> np.ndarray:
    a = 2.0 / (W + 1.0)
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    return y


def desmooth_ema(y: np.ndarray, W: int) -> np.ndarray:
    alpha = 2.0 / (W + 1.0)
    x = np.empty_like(y)
    x[0] = y[0]
    x[1:] = (y[1:] - (1.0 - alpha) * y[:-1]) / alpha
    return np.clip(x, 0.0, 100.0)


def augment(df: pd.DataFrame, W_map: dict) -> pd.DataFrame:
    out = df.copy()
    out["target_desm"] = np.nan
    out["oracle_W"] = 15
    for cid, sub in df.groupby("case_id"):
        idx = sub.index.to_numpy()
        W = int(W_map.get(int(cid), 15))
        a = df["target"].values[idx]
        if np.isnan(a).all():
            continue
        mean_fill = float(np.nanmean(a)) if not np.isnan(np.nanmean(a)) else 50.0
        a_filled = np.where(np.isnan(a), mean_fill, a)
        out.loc[idx, "target_desm"] = desmooth_ema(a_filled, W).astype(np.float64)
        out.loc[idx, "oracle_W"] = W
    return out


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
    ap.add_argument("--label", default="oracle_desmooth")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    # Load oracle W caches
    W_train = pd.read_csv(RESULTS / "oracle_W_train.csv").set_index("case_id")["oracle_W"].to_dict()
    W_val = pd.read_csv(RESULTS / "oracle_W_val.csv").set_index("case_id")["oracle_W"].to_dict()
    print(f"Oracle W cache loaded: {len(W_train)} train cases, {len(W_val)} val cases")
    print(f"  train W distribution: {pd.Series(list(W_train.values())).value_counts().sort_index().to_dict()}")
    print(f"  val W distribution:   {pd.Series(list(W_val.values())).value_counts().sort_index().to_dict()}")

    train_df = pd.read_parquet(RESULTS / "features_train_n500_v3.parquet")
    val_df   = pd.read_parquet(RESULTS / "features_val_n100_v3.parquet")
    print(f"\nTrain rows: {len(train_df):,}  Val rows: {len(val_df):,}")

    print("\nDesmoothing targets with cached oracle W...")
    train_aug = augment(train_df, W_train)
    val_aug   = augment(val_df, W_val)

    # 22-feature extended set from Phase 3f (no oracle bis_* columns)
    base = ["openibis_paper", "openibis_quazi", "openibis_quazi_30s",
            "bsr_paper", "bsr_quazi", "sef95", "bcsef", "beta_ratio", "emg_proxy",
            "p_delta", "p_theta", "p_alpha", "p_beta", "p_lowgamma",
            "spectral_entropy"]
    extra = ["openibis_quazi_5s", "openibis_quazi_10s", "openibis_quazi_60s",
             "openibis_quazi_dt", "openibis_quazi_30s_dt",
             "sef95_dt", "emg_proxy_dt"]
    feat_cols = base + [c for c in extra if c in train_df.columns]
    print(f"Features ({len(feat_cols)}): {feat_cols}")

    X = train_aug[feat_cols].values
    y = train_aug["target_desm"].values
    Xv = val_aug[feat_cols].values
    yv_desm = val_aug["target_desm"].values

    params = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=5, verbose=-1,
                  num_threads=args.threads)
    dtrain = lgb.Dataset(X, label=y, feature_name=feat_cols)
    dval = lgb.Dataset(Xv, label=yv_desm, feature_name=feat_cols, reference=dtrain)
    print(f"\n--- Training on per-case oracle-W desmoothed target ---")
    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=2000,
                        valid_sets=[dtrain, dval], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    out = MODEL_DIR / f"lgbm_{args.label}.txt"
    booster.save_model(str(out))
    print(f"  best_iter={booster.best_iteration}  {time.time()-t0:.1f}s  -> {out.name}")

    # Predict
    pred_raw = np.clip(booster.predict(Xv), 0, 100)

    # Re-smooth pred at each case's cached W
    pred_resm = np.empty_like(pred_raw)
    for cid, sub in val_aug.groupby("case_id"):
        idx = sub.index.to_numpy()
        W = int(W_val.get(int(cid), 15))
        pred_resm[idx] = ema(pred_raw[idx], W)

    # Also re-smooth at fixed 15 s for comparison
    pred_sm15 = np.empty_like(pred_raw)
    for cid, sub in val_aug.groupby("case_id"):
        idx = sub.index.to_numpy()
        pred_sm15[idx] = ema(pred_raw[idx], 15.0)

    actual = val_df["target"].values

    print(f"\n=== Val cohort vs ORIGINAL smoothed actual ===")
    print(f"{'variant':<44s}  {'MAE':>5s}  {'r':>6s}  {'Lin_rc':>7s}")
    for name, p in [
        ("raw model output (predicts desmoothed)",  pred_raw),
        ("re-smoothed at per-case oracle W",         pred_resm),
        ("re-smoothed at fixed 15 s",                pred_sm15),
    ]:
        mae, r, rc = safe(p, actual)
        print(f"  {name:<42s}  {mae:5.2f}  {r:6.3f}  {rc:7.3f}")

    print(f"\n=== Val cohort vs DESMOOTHED target (fidelity) ===")
    mae, r, rc = safe(pred_raw, yv_desm)
    print(f"  raw pred vs target_desm                    {mae:5.2f}  {r:6.3f}  {rc:7.3f}")

    print(f"\n=== Per-regime MAE (vs original actual BIS) ===")
    print(f"{'variant':<44s}  " + "  ".join(f"{l:>7s}" for l in LEE_BIN_LABELS))
    for name, p in [
        ("raw model output",       pred_raw),
        ("re-smoothed @ oracle W", pred_resm),
        ("re-smoothed @ 15 s",     pred_sm15),
    ]:
        r = per_regime(actual, p)
        print(f"  {name:<42s}  " + "  ".join(
            f"{r[k]:>7.2f}" if not np.isnan(r[k]) else f"{'nan':>7s}"
            for k in LEE_BIN_LABELS))

    imp = booster.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1][:15]
    print(f"\nTop 15 features (gain):")
    for i in order:
        print(f"  {feat_cols[i]:<26s}  {imp[i]:>12.1f}")


if __name__ == "__main__":
    main()
