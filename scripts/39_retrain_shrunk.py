"""Retrain shrunk GBM and Lee piecewise models, bundle them as the new ship.

Configuration chosen from scripts/38 grid search:
  GBM:  n_estimators=500, num_leaves=31
        (was 2000/63 → 11.7 MB / MAE 3.67; now → ~1.5 MB / MAE 3.77)
  Lee:  max_depth=3, top_k=8, min_samples_leaf=10_000
        (was 4/6 / 16 leaves / 112 params / MAE 4.41;
         now → 3/8 /  8 leaves /  72 params / MAE 4.39)

Outputs (overwrite):
  openeeg/models/predict_bis_v3.txt        — new ~1.5 MB GBM
  openeeg/models/piecewise_raw_data.json   — new 8-leaf piecewise
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS, lin_concordance

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS = Path(__file__).resolve().parents[1] / "openeeg" / "models"

GBM_FEATURES = [
    "openibis_paper", "openibis_quazi", "openibis_quazi_30s",
    "bsr_paper", "bsr_quazi", "sef95", "bcsef", "beta_ratio", "emg_proxy",
    "p_theta", "p_alpha", "p_beta", "p_lowgamma",
    "openibis_quazi_5s", "openibis_quazi_10s", "openibis_quazi_60s",
    "openibis_quazi_30s_dt", "p_delta",
]
LEE_FEATURES = [
    "openibis_paper", "openibis_quazi", "openibis_quazi_30s",
    "bsr_paper", "bsr_quazi", "sef95", "bcsef", "beta_ratio", "emg_proxy",
    "p_delta", "p_theta", "p_alpha", "p_beta", "p_lowgamma", "spectral_entropy",
    "openibis_quazi_5s", "openibis_quazi_10s", "openibis_quazi_60s",
    "openibis_quazi_dt", "openibis_quazi_30s_dt", "sef95_dt", "emg_proxy_dt",
    "openbsr",
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


def per_regime_mae(actual, pred):
    m = np.isfinite(actual) & np.isfinite(pred)
    a, p = actual[m], pred[m]
    out = {}
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        mm = (a >= lo) & (a < hi)
        out[lbl] = float(np.mean(np.abs(p[mm] - a[mm]))) if mm.sum() > 10 else float("nan")
    return out


def apply_deep_rule(pred, openbsr, threshold=49.8):
    p = pred.copy()
    deep = np.where(np.isnan(openbsr), False, openbsr > threshold)
    p[deep] = np.clip(44.1 - openbsr[deep] / 2.25, 0, 100)
    return p


def retrain_gbm(tr, vl, actual, openbsr_v):
    print("\n=== Retraining GBM (500 trees, 31 leaves) ===")
    X_tr = tr[GBM_FEATURES].values
    y_tr = tr["target_desm"].values
    X_v  = vl[GBM_FEATURES].values
    params = dict(
        objective="regression_l1", metric="l1", learning_rate=0.05,
        num_leaves=31, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
        verbose=-1, num_threads=64,
    )
    t0 = time.time()
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=GBM_FEATURES)
    booster = lgb.train(params, dtrain, num_boost_round=500,
                        callbacks=[lgb.log_evaluation(0)])
    out_path = MODELS / "predict_bis_v3.txt"
    booster.save_model(str(out_path))
    size_kb = out_path.stat().st_size / 1024
    train_sec = time.time() - t0
    print(f"  trained in {train_sec:.1f}s   saved → {out_path}")
    print(f"  file size: {size_kb:.1f} kB")

    pred_raw = np.clip(booster.predict(X_v), 0, 100)
    pred_rule = apply_deep_rule(pred_raw, openbsr_v)
    pred_sm = smooth_by_case(pred_rule, vl, W=15.0)
    m = np.isfinite(actual) & np.isfinite(pred_sm)
    mae = float(np.mean(np.abs(pred_sm[m] - actual[m])))
    r = float(np.corrcoef(pred_sm[m], actual[m])[0, 1])
    reg = per_regime_mae(actual, pred_sm)
    print(f"\n  GBM val MAE = {mae:.3f}  r = {r:.3f}")
    for k, v in reg.items():
        print(f"    {k}: {v:.2f}")
    return dict(method="gbm", mae=mae, r=r, size_kb=size_kb, **reg)


def retrain_lee(tr, vl, actual, openbsr_v):
    print("\n=== Refitting Lee piecewise (depth=5, top_k=8) ===")
    X_tr = tr[LEE_FEATURES].values
    y_tr = tr["target"].values
    X_v  = vl[LEE_FEATURES].values
    keep = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
    t0 = time.time()
    tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10_000,
                                  random_state=0)
    tree.fit(X_tr[keep], y_tr[keep])
    leaves_tr = tree.apply(np.where(np.isnan(X_tr), 0.0, X_tr))
    leaves_v  = tree.apply(np.where(np.isnan(X_v), 0.0, X_v))
    unique_leaves = sorted(set(leaves_tr.tolist()) | set(leaves_v.tolist()))
    print(f"  tree built ({len(unique_leaves)} leaves)")

    leaf_specs = []
    pred_v = np.full(len(vl), np.nan, dtype=float)
    for lid in unique_leaves:
        mtr = (leaves_tr == lid) & ~np.isnan(y_tr) & ~np.isnan(X_tr).any(axis=1)
        mv  = (leaves_v == lid) & ~np.isnan(X_v).any(axis=1)
        if mtr.sum() < 500:
            mean_v = float(np.nanmean(y_tr[leaves_tr == lid])) if (leaves_tr == lid).sum() else 50.0
            if mv.any():
                pred_v[mv] = mean_v
            leaf_specs.append(dict(leaf=int(lid), n_train=int(mtr.sum()),
                                    n_val=int(mv.sum()), features=[],
                                    intercept=mean_v, coefs=[], note="fallback_mean"))
            continue
        Xl = X_tr[mtr]; yl = y_tr[mtr]
        corrs = np.array([
            float(np.corrcoef(Xl[:, j], yl)[0, 1]) if Xl[:, j].std() > 1e-9 else 0.0
            for j in range(Xl.shape[1])
        ])
        top = np.argsort(np.abs(corrs))[::-1][:8]
        m = LinearRegression().fit(Xl[:, top], yl)
        if mv.any():
            pred_v[mv] = m.predict(X_v[mv][:, top])
        leaf_specs.append(dict(
            leaf=int(lid),
            n_train=int(mtr.sum()), n_val=int(mv.sum()),
            features=[LEE_FEATURES[j] for j in top],
            intercept=float(m.intercept_),
            coefs=[float(c) for c in m.coef_],
        ))

    # Apply Ellerkmann + EMA
    pred_v = np.clip(pred_v, 0, 100)
    pred_rule = apply_deep_rule(pred_v, openbsr_v)
    pred_sm = smooth_by_case(pred_rule, vl, W=15.0)
    m_f = np.isfinite(actual) & np.isfinite(pred_sm)
    mae = float(np.mean(np.abs(pred_sm[m_f] - actual[m_f])))
    r = float(np.corrcoef(pred_sm[m_f], actual[m_f])[0, 1])
    reg = per_regime_mae(actual, pred_sm)

    # Save model JSON
    feat_cols = LEE_FEATURES
    out_path = MODELS / "piecewise_raw_data.json"
    with open(out_path, "w") as f:
        json.dump(dict(
            variant="raw", partition="data",
            feature_cols=feat_cols,
            deep_rule=dict(feature="openbsr", threshold=49.8,
                            formula="BIS = 44.1 - openbsr/2.25"),
            leaves=leaf_specs,
            tree_thresholds=[
                dict(node=int(i),
                     feature=feat_cols[tree.tree_.feature[i]] if tree.tree_.feature[i] >= 0 else None,
                     threshold=float(tree.tree_.threshold[i]),
                     left=int(tree.tree_.children_left[i]),
                     right=int(tree.tree_.children_right[i]))
                for i in range(tree.tree_.node_count)
            ]), f, indent=2)
    size_kb = out_path.stat().st_size / 1024
    print(f"  saved → {out_path}")
    print(f"  file size: {size_kb:.1f} kB")
    print(f"  fit in {time.time()-t0:.1f}s")

    print(f"\n  Lee val MAE = {mae:.3f}  r = {r:.3f}")
    for k, v in reg.items():
        print(f"    {k}: {v:.2f}")
    return dict(method="lee", mae=mae, r=r, size_kb=size_kb, **reg)


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "64")
    print("Loading W=15 train & val (v5)...")
    train = pd.read_parquet(RESULTS / "features_train_n500_v5.parquet")
    val   = pd.read_parquet(RESULTS / "features_val_n100_v5.parquet")
    tr = w15_filter(train, RESULTS / "oracle_W_train.csv").reset_index(drop=True)
    vl = w15_filter(val,   RESULTS / "oracle_W_val.csv").reset_index(drop=True)
    print(f"  train: {len(tr):,} rows / {tr['case_id'].nunique()} cases")
    print(f"  val:   {len(vl):,} rows / {vl['case_id'].nunique()} cases")
    print("Desmoothing target for GBM...")
    tr["target_desm"] = desmooth_target_by_case(tr, W=15.0)

    actual = vl["target"].values
    openbsr_v = vl["openbsr"].values

    gbm_result = retrain_gbm(tr, vl, actual, openbsr_v)
    lee_result = retrain_lee(tr, vl, actual, openbsr_v)

    print("\n" + "=" * 60)
    print("Ship-ready models")
    print("=" * 60)
    print(f"  GBM (v3, 500/31):   {gbm_result['size_kb']:>8.1f} kB   MAE {gbm_result['mae']:.3f}")
    print(f"  Lee (3/8, 8 leaves): {lee_result['size_kb']:>8.1f} kB   MAE {lee_result['mae']:.3f}")


if __name__ == "__main__":
    main()
