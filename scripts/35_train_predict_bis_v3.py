"""Train predict_bis_v3 — 18-feature LightGBM + Ellerkmann deep rule.

v3 differs from v2 in three ways:
  1. Drop the four lowest-gain v2 features
     (emg_proxy_dt, sef95_dt, openibis_quazi_dt, spectral_entropy).
  2. Bundle the Ellerkmann deep rule at inference time
     (openbsr > 49.8 → BIS = clip(44.1 − openbsr/2.25, 0, 100)).
  3. Train on the same W=15 sub-cohort with desmoothed target.

Save to openeeg/models/predict_bis_v3.txt.

Reported val (W=15) numbers from scripts/34_v2_feature_ablation.py:
  MAE = 3.67  (+0.05 BIS vs v2 22-feature baseline)
  0-21 = 3.17 (better than v2's 6.15 thanks to the rule)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS, lin_concordance

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODEL_OUT = Path(__file__).resolve().parents[1] / "openeeg" / "models" / "predict_bis_v3.txt"

V3_FEATURES = [
    # Kept (gain order, top→bottom — high-importance first):
    "openibis_paper", "openibis_quazi", "openibis_quazi_30s",
    "bsr_paper", "bsr_quazi", "sef95", "bcsef", "beta_ratio", "emg_proxy",
    "p_theta", "p_alpha", "p_beta", "p_lowgamma",
    "openibis_quazi_5s", "openibis_quazi_10s", "openibis_quazi_60s",
    "openibis_quazi_30s_dt",
    "p_delta",
    # Dropped (vs v2):
    #   emg_proxy_dt, sef95_dt, openibis_quazi_dt, spectral_entropy
]


def w15_filter(df, oracle_w_csv):
    w = pd.read_csv(oracle_w_csv)
    keep = set(w.loc[w["oracle_W"] == 15, "case_id"].astype(int))
    return df[df["case_id"].isin(keep)].reset_index(drop=True)


def ema(x, W=15.0):
    a = 2.0 / (W + 1.0)
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    return y


def desmooth_ema(y, W=15.0):
    a = 2.0 / (W + 1.0)
    x = np.empty_like(y, dtype=float)
    x[0] = y[0]
    x[1:] = (y[1:] - (1.0 - a) * y[:-1]) / a
    return np.clip(x, 0, 100)


def smooth_by_case(p, df, W=15.0):
    out = np.empty_like(p, dtype=float)
    for cid, sub in df.groupby("case_id"):
        idx = sub.index.to_numpy()
        out[idx] = ema(p[idx], W)
    return out


def desmooth_target_by_case(df, W=15.0):
    out = np.full(len(df), np.nan, dtype=float)
    for cid, sub in df.groupby("case_id"):
        idx = sub.index.to_numpy()
        a = df["target"].values[idx]
        mu = float(np.nanmean(a)) if not np.isnan(np.nanmean(a)) else 50.0
        a_f = np.where(np.isnan(a), mu, a)
        out[idx] = desmooth_ema(a_f, W)
    return out


def per_regime(actual, pred):
    m = np.isfinite(actual) & np.isfinite(pred)
    a, p = actual[m], pred[m]
    out = {}
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        mm = (a >= lo) & (a < hi)
        out[lbl] = float(np.mean(np.abs(p[mm] - a[mm]))) if mm.sum() > 10 else float("nan")
    return out


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "64")

    print("Loading W=15 train & val (v5)...")
    train = pd.read_parquet(RESULTS / "features_train_n500_v5.parquet")
    val   = pd.read_parquet(RESULTS / "features_val_n100_v5.parquet")
    tr = w15_filter(train, RESULTS / "oracle_W_train.csv").reset_index(drop=True)
    vl = w15_filter(val,   RESULTS / "oracle_W_val.csv").reset_index(drop=True)
    print(f"  train: {len(tr):,} rows / {tr['case_id'].nunique()} cases")
    print(f"  val:   {len(vl):,} rows / {vl['case_id'].nunique()} cases")

    print("Desmoothing targets at W=15...")
    tr["target_desm"] = desmooth_target_by_case(tr, W=15.0)
    vl["target_desm"] = desmooth_target_by_case(vl, W=15.0)

    # Ensure all V3_FEATURES are present
    missing = [f for f in V3_FEATURES if f not in tr.columns]
    assert not missing, f"missing features in train: {missing}"

    print(f"\nTraining v3 with {len(V3_FEATURES)} features...")
    print(f"  features: {V3_FEATURES}")
    X_tr = tr[V3_FEATURES].values
    y_tr = tr["target_desm"].values
    X_v  = vl[V3_FEATURES].values
    y_v  = vl["target_desm"].values
    actual = vl["target"].values

    params = dict(
        objective="regression_l1", metric="l1", learning_rate=0.05,
        num_leaves=63, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
        verbose=-1, num_threads=64,
    )
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=V3_FEATURES)
    dval   = lgb.Dataset(X_v,  label=y_v,  feature_name=V3_FEATURES, reference=dtrain)
    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=2000,
                        valid_sets=[dtrain, dval], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
    booster.save_model(str(MODEL_OUT))
    print(f"  trained in {time.time()-t0:.1f}s   best_iter={booster.best_iteration}")
    print(f"  saved → {MODEL_OUT}")
    print(f"  file size: {MODEL_OUT.stat().st_size / 1024:.1f} kB")

    # Evaluate (raw, +EMA, +EMA+rule)
    pred_raw = np.clip(booster.predict(X_v), 0, 100)
    obsr = vl["openbsr"].values
    deep_mask = np.where(np.isnan(obsr), False, obsr > 49.8)
    ellerk = np.clip(44.1 - obsr / 2.25, 0, 100)

    variants = [
        ("raw model", pred_raw),
        ("+ EMA(15s)", smooth_by_case(pred_raw, vl, W=15.0)),
        ("+ EMA(15s) + Ellerkmann (SHIPPED)",
         smooth_by_case(np.where(deep_mask, ellerk, pred_raw), vl, W=15.0)),
    ]
    print(f"\n=== Val (W=15, SQI≥80) ===")
    print(f"{'variant':<40s}  {'MAE':>5s}  {'r':>6s}  {'rc':>6s}")
    for name, p in variants:
        m = np.isfinite(actual) & np.isfinite(p)
        mae = float(np.mean(np.abs(p[m] - actual[m])))
        r = float(np.corrcoef(p[m], actual[m])[0, 1])
        rc = lin_concordance(p[m], actual[m])
        print(f"  {name:<38s}  {mae:5.2f}  {r:6.3f}  {rc:6.3f}")

    print(f"\n=== Per-regime MAE ===")
    print(f"{'variant':<40s}  " + "  ".join(f"{l:>7s}" for l in LEE_BIN_LABELS))
    for name, p in variants:
        r = per_regime(actual, p)
        print(f"  {name:<38s}  " + "  ".join(
            f"{r[k]:>7.2f}" if not np.isnan(r[k]) else f"{'nan':>7s}"
            for k in LEE_BIN_LABELS))


if __name__ == "__main__":
    main()
