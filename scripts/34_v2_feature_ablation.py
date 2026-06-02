"""Phase 3k — feature ablation on predict_bis_v2.

Procedure:
  1. Read v2's gain-based feature importance to rank candidates.
  2. Train LightGBM (same hyperparameters as Phase 3h) on progressively
     smaller feature subsets, dropping the lowest-gain features first.
  3. Evaluate each retrained model on the W=15 val cohort.
  4. Repeat with the Ellerkmann deep-rule override (openbsr > 49.8 →
     BIS = 44.1 − openbsr/2.25) so we know whether feature trimming
     interacts with the rule.

Outputs
  results/v2_feature_ablation.csv     one row per (drop_n, with_rule)
  results/v2_feature_ablation.txt     human-readable summary
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS

RESULTS = Path(__file__).resolve().parents[1] / "results"
V2_MODEL = Path(__file__).resolve().parents[1] / "openeeg" / "models" / "predict_bis_v2.txt"

V2_FEATURES = [
    "openibis_paper", "openibis_quazi", "openibis_quazi_30s",
    "bsr_paper", "bsr_quazi", "sef95", "bcsef", "beta_ratio", "emg_proxy",
    "p_delta", "p_theta", "p_alpha", "p_beta", "p_lowgamma", "spectral_entropy",
    "openibis_quazi_5s", "openibis_quazi_10s", "openibis_quazi_60s",
    "openibis_quazi_dt", "openibis_quazi_30s_dt", "sef95_dt", "emg_proxy_dt",
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


def metrics_for(actual, pred):
    m = np.isfinite(actual) & np.isfinite(pred)
    mae = float(np.mean(np.abs(pred[m] - actual[m])))
    r = float(np.corrcoef(pred[m], actual[m])[0, 1])
    return mae, r


def train_subset(X_tr, y_tr, X_v, y_v, feat_names, num_threads=32):
    params = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=5, verbose=-1,
                  num_threads=num_threads)
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names)
    dval   = lgb.Dataset(X_v,  label=y_v,  feature_name=feat_names, reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=2000,
                        valid_sets=[dtrain, dval], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    return booster


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "64")
    print("Loading W=15 train & val (v5)...")
    train = pd.read_parquet(RESULTS / "features_train_n500_v5.parquet")
    val   = pd.read_parquet(RESULTS / "features_val_n100_v5.parquet")
    tr = w15_filter(train, RESULTS / "oracle_W_train.csv").reset_index(drop=True)
    vl = w15_filter(val,   RESULTS / "oracle_W_val.csv").reset_index(drop=True)
    print(f"  train: {len(tr):,} rows / {tr['case_id'].nunique()} cases")
    print(f"  val:   {len(vl):,} rows / {vl['case_id'].nunique()} cases")

    # Desmooth targets (Phase 3h convention)
    print("Desmoothing targets at W=15...")
    tr["target_desm"] = desmooth_target_by_case(tr, W=15.0)
    vl["target_desm"] = desmooth_target_by_case(vl, W=15.0)

    # ---- Step 1: read v2's importance ranking (gain) ----
    booster_v2 = lgb.Booster(model_file=str(V2_MODEL))
    fnames = booster_v2.feature_name()
    imp = booster_v2.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1]  # high → low
    rank = pd.DataFrame({"feature": [fnames[i] for i in order],
                          "gain":    [int(imp[i]) for i in order]})
    print("\n=== predict_bis_v2 feature ranking (gain, top→bottom) ===")
    print(rank.to_string(index=False))
    bottom_to_top = [fnames[i] for i in np.argsort(imp)]  # low → high
    print(f"\nDrop order (low gain first): {bottom_to_top[:8]}")

    # ---- Step 2: train ablated models ----
    actual = vl["target"].values
    y_tr = tr["target_desm"].values
    obsr_v = vl["openbsr"].values
    deep_mask_v = np.where(np.isnan(obsr_v), False, obsr_v > 49.8)
    ellerk_v = np.clip(44.1 - obsr_v / 2.25, 0, 100)

    drop_levels = [0, 4, 8, 12, 16]
    rows = []
    for drop_n in drop_levels:
        if drop_n >= len(V2_FEATURES):
            continue
        keep_feats = [f for f in V2_FEATURES if f not in bottom_to_top[:drop_n]]
        print(f"\n----- Training with {len(keep_feats)} features (dropped bottom {drop_n}) -----")
        print(f"  kept: {keep_feats}")
        X_tr = tr[keep_feats].values
        X_v  = vl[keep_feats].values
        booster = train_subset(X_tr, y_tr, X_v, vl["target_desm"].values, keep_feats)
        pred_raw = np.clip(booster.predict(X_v), 0, 100)
        pred_sm = smooth_by_case(pred_raw, vl, W=15.0)
        mae_no, r_no = metrics_for(actual, pred_sm)
        per_no = per_regime(actual, pred_sm)
        rows.append(dict(drop_n=drop_n, n_features=len(keep_feats),
                          with_rule=False, mae=mae_no, r=r_no, **per_no))
        # With Ellerkmann rule
        hybrid = pred_raw.copy()
        hybrid[deep_mask_v] = ellerk_v[deep_mask_v]
        hyb_sm = smooth_by_case(hybrid, vl, W=15.0)
        mae_yes, r_yes = metrics_for(actual, hyb_sm)
        per_yes = per_regime(actual, hyb_sm)
        rows.append(dict(drop_n=drop_n, n_features=len(keep_feats),
                          with_rule=True, mae=mae_yes, r=r_yes, **per_yes))
        print(f"  no rule : MAE={mae_no:.3f}  r={r_no:.3f}")
        print(f"  +rule   : MAE={mae_yes:.3f}  r={r_yes:.3f}")
        print(f"  best_iter={booster.best_iteration}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "v2_feature_ablation.csv", index=False)

    # ---- Step 3: report ----
    print("\n" + "=" * 70)
    print("Summary table (val W=15 cohort, EMA(15s) post-smoothed)")
    print("=" * 70)
    cols = ["drop_n", "n_features", "with_rule", "mae", "r"] + list(LEE_BIN_LABELS)
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    base_mae = float(df[(df["drop_n"] == 0) & (df["with_rule"] == False)].iloc[0]["mae"])
    base_mae_rule = float(df[(df["drop_n"] == 0) & (df["with_rule"] == True)].iloc[0]["mae"])
    report = ["v2 feature-ablation summary",
              "============================",
              f"Baseline (22 features, no rule):  MAE = {base_mae:.3f}",
              f"Baseline + Ellerkmann rule:       MAE = {base_mae_rule:.3f}",
              ""]
    report.append("Drop bottom-N features (by gain) and retrain:\n")
    for _, row in df.iterrows():
        tag = "+rule" if row["with_rule"] else "raw  "
        delta = row["mae"] - (base_mae_rule if row["with_rule"] else base_mae)
        report.append(f"  drop_n={int(row['drop_n']):>2d}  n_feat={int(row['n_features']):>2d}  {tag}  "
                      f"MAE={row['mae']:.3f}  Δ={delta:+.3f}")
    out_txt = RESULTS / "v2_feature_ablation.txt"
    out_txt.write_text("\n".join(report))
    print(f"\nSaved {out_txt}")


if __name__ == "__main__":
    main()
