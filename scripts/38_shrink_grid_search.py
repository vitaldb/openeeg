"""Shrink GBM and Lee piecewise models — grid search.

User hypothesis: BIS is fundamentally rule-based, so fewer trees and
shallower depth should achieve similar accuracy. Two parallel grids:

  1. GBM (LightGBM) grid: n_estimators × num_leaves
       n_estimators ∈ {50, 100, 200, 500, 1000, 2000}
       num_leaves   ∈ {7, 15, 31, 63}
  2. Lee piecewise grid: max_depth × min_samples_leaf × top_k
       max_depth        ∈ {2, 3, 4, 5}
       min_samples_leaf ∈ {5000, 10_000, 20_000}
       top_k            ∈ {3, 4, 6, 8}

For each config, train on the 397-case W=15 train fold (target_desm),
evaluate on the 80-case W=15 val fold with mandatory Ellerkmann +
EMA(15s). All training uses 64 threads.

Outputs
  results/shrink_grid_search_gbm.csv
  results/shrink_grid_search_lee.csv
  results/shrink_grid_search.png  — Pareto-front trade-off
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


def grid_gbm(tr, vl, actual, openbsr_v):
    rows = []
    grid_n = [50, 100, 200, 500, 1000, 2000]
    grid_l = [7, 15, 31, 63]
    print(f"\n=== GBM grid ({len(grid_n) * len(grid_l)} configs) ===")
    X_tr = tr[GBM_FEATURES].values
    y_tr = tr["target_desm"].values
    X_v  = vl[GBM_FEATURES].values
    base_params = dict(
        objective="regression_l1", metric="l1", learning_rate=0.05,
        min_data_in_leaf=200, feature_fraction=0.9,
        bagging_fraction=0.8, bagging_freq=5, verbose=-1, num_threads=64,
    )
    for n_est in grid_n:
        for n_leaves in grid_l:
            params = dict(base_params, num_leaves=n_leaves)
            t0 = time.time()
            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=GBM_FEATURES)
            booster = lgb.train(params, dtrain, num_boost_round=n_est,
                                callbacks=[lgb.log_evaluation(0)])
            train_time = time.time() - t0
            pred_raw = np.clip(booster.predict(X_v), 0, 100)
            pred_rule = apply_deep_rule(pred_raw, openbsr_v)
            pred_sm = smooth_by_case(pred_rule, vl, W=15.0)
            mae_full = float(np.mean(np.abs(pred_sm[np.isfinite(pred_sm) & np.isfinite(actual)]
                                              - actual[np.isfinite(pred_sm) & np.isfinite(actual)])))
            reg = per_regime_mae(actual, pred_sm)
            # Save model size proxy (number of trees × leaves × ~8 bytes per leaf)
            model_text = booster.model_to_string()
            n_trees_actual = booster.num_trees()
            model_bytes = len(model_text)
            rows.append(dict(
                n_estimators=n_est, num_leaves=n_leaves,
                n_trees_actual=n_trees_actual,
                model_kb=model_bytes / 1024,
                train_sec=train_time,
                mae=mae_full, **reg,
            ))
            print(f"  n_est={n_est:>4d}  leaves={n_leaves:>2d}  "
                  f"trees={n_trees_actual:>4d}  MAE={mae_full:.3f}  "
                  f"size={model_bytes/1024:>6.1f} kB  train={train_time:.1f}s")
    return pd.DataFrame(rows)


def grid_lee(tr, vl, actual, openbsr_v):
    rows = []
    grid_depth = [2, 3, 4, 5]
    grid_min   = [5_000, 10_000, 20_000]
    grid_topk  = [3, 4, 6, 8]
    print(f"\n=== Lee grid ({len(grid_depth) * len(grid_min) * len(grid_topk)} configs) ===")
    X_tr = tr[LEE_FEATURES].values
    y_tr = tr["target"].values  # use SMOOTHED target (matches Phase D2)
    X_v  = vl[LEE_FEATURES].values
    keep_tr_mask = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
    for depth in grid_depth:
        for min_samples in grid_min:
            # Build a single tree at this depth (data-driven)
            tree = DecisionTreeRegressor(max_depth=depth,
                                          min_samples_leaf=min_samples,
                                          random_state=0)
            tree.fit(X_tr[keep_tr_mask], y_tr[keep_tr_mask])
            leaves_tr = tree.apply(np.where(np.isnan(X_tr), 0.0, X_tr))
            leaves_v  = tree.apply(np.where(np.isnan(X_v), 0.0, X_v))
            unique_leaves = sorted(set(leaves_tr.tolist()) | set(leaves_v.tolist()))

            for top_k in grid_topk:
                pred_v = np.full(len(vl), np.nan, dtype=float)
                n_params_total = 0
                for lid in unique_leaves:
                    mtr = (leaves_tr == lid) & ~np.isnan(y_tr) & ~np.isnan(X_tr).any(axis=1)
                    mv  = (leaves_v  == lid) & ~np.isnan(X_v).any(axis=1)
                    if mtr.sum() < 500:
                        if mv.any():
                            fallback = float(np.nanmean(y_tr[leaves_tr == lid])) if (leaves_tr == lid).sum() else float(np.nanmean(y_tr))
                            pred_v[mv] = fallback
                            n_params_total += 1
                        continue
                    Xl = X_tr[mtr]
                    yl = y_tr[mtr]
                    corrs = np.array([
                        float(np.corrcoef(Xl[:, j], yl)[0, 1])
                        if Xl[:, j].std() > 1e-9 else 0.0
                        for j in range(Xl.shape[1])
                    ])
                    top = np.argsort(np.abs(corrs))[::-1][:top_k]
                    m = LinearRegression().fit(Xl[:, top], yl)
                    if mv.any():
                        pred_v[mv] = m.predict(X_v[mv][:, top])
                    n_params_total += 1 + top_k  # intercept + coefs

                pred_v = np.clip(pred_v, 0, 100)
                pred_rule = apply_deep_rule(pred_v, openbsr_v)
                pred_sm = smooth_by_case(pred_rule, vl, W=15.0)
                m_finite = np.isfinite(pred_sm) & np.isfinite(actual)
                mae = float(np.mean(np.abs(pred_sm[m_finite] - actual[m_finite])))
                reg = per_regime_mae(actual, pred_sm)
                rows.append(dict(
                    max_depth=depth, min_samples_leaf=min_samples,
                    top_k=top_k, n_leaves=len(unique_leaves),
                    n_params=n_params_total,
                    mae=mae, **reg,
                ))
                print(f"  depth={depth} min={min_samples:>6d} top_k={top_k}  "
                      f"leaves={len(unique_leaves):>3d}  params={n_params_total:>4d}  "
                      f"MAE={mae:.3f}")
    return pd.DataFrame(rows)


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

    actual = vl["target"].values
    openbsr_v = vl["openbsr"].values

    df_gbm = grid_gbm(tr, vl, actual, openbsr_v)
    df_gbm.to_csv(RESULTS / "shrink_grid_search_gbm.csv", index=False)

    df_lee = grid_lee(tr, vl, actual, openbsr_v)
    df_lee.to_csv(RESULTS / "shrink_grid_search_lee.csv", index=False)

    # Find Pareto-efficient configs
    print("\n=== Best GBM configs (sorted by MAE) ===")
    print(df_gbm.sort_values("mae").head(10)[
        ["n_estimators", "num_leaves", "n_trees_actual", "model_kb", "mae",
         "0-21", "21-41", "41-61", "61-78", "78-98"]
    ].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Best Lee configs (sorted by MAE) ===")
    print(df_lee.sort_values("mae").head(10)[
        ["max_depth", "min_samples_leaf", "top_k", "n_leaves", "n_params", "mae",
         "0-21", "21-41", "41-61", "61-78", "78-98"]
    ].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ---- Pareto plot ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    sc = ax.scatter(df_gbm["model_kb"], df_gbm["mae"],
                     c=df_gbm["num_leaves"], cmap="viridis",
                     s=df_gbm["n_estimators"] / 5, alpha=0.8)
    for _, row in df_gbm.iterrows():
        ax.annotate(f"{int(row['n_estimators'])}/{int(row['num_leaves'])}",
                     (row["model_kb"], row["mae"]), fontsize=7, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="num_leaves")
    ax.set_xscale("log")
    ax.set_xlabel("model file size (kB)")
    ax.set_ylabel("val MAE (BIS)")
    ax.set_title("(a) GBM: complexity vs accuracy  "
                  f"(current ship: 2000/63, MAE 3.82)")
    ax.axhline(3.82, color="gray", ls="--", lw=0.6, label="current v3 (3.82)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sc = ax.scatter(df_lee["n_params"], df_lee["mae"],
                     c=df_lee["max_depth"], cmap="viridis",
                     s=df_lee["top_k"] * 20, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="max_depth")
    ax.set_xlabel("total parameters")
    ax.set_ylabel("val MAE (BIS)")
    ax.set_title(f"(b) Lee piecewise: parameters vs accuracy  "
                  f"(current ship: 4/6, MAE 4.46)")
    ax.axhline(4.46, color="gray", ls="--", lw=0.6, label="current Lee K=6 (4.46)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS / "shrink_grid_search.png", dpi=110)
    plt.close(fig)
    print(f"\nSaved results/shrink_grid_search.{{csv,png}}")


if __name__ == "__main__":
    main()
