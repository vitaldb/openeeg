"""Phase 3e — Hybrid Lee-2019-rule + per-region LightGBM.

Structure:

    if BSR > 49.8%:                 -> 0-21 range, use Ellerkmann (44.1 - BSR/2.25)
    elif EMG < 34.2 AND SEF < 20.2: -> 21-61 range, use LightGBM trained on mid
    else:                            -> 61-98 range, use LightGBM trained on awake

Two variants compared:
  * ``--gate-oracle`` (default): uses BIS Vista's published BSR/SEF/EMG
    tracks for gating (= Lee 2019's actual inputs). Research ceiling.
  * ``--gate-raw``: uses our raw-EEG-derived BSR (quazi)/SEF95/EMG-proxy
    for gating. Library-deployable variant.

In both variants the per-region LightGBM gets every feature column
(including the oracle ones if present in the parquet); only the gating
inputs differ.

Outputs go to ``results/lgbm_hybrid_<gate>.txt`` per region; eval table
plus per-regime MAE is printed to stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS, lin_concordance

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def gate_features(df: pd.DataFrame, oracle: bool):
    if oracle:
        return (df["bis_sr_oracle"].values,
                df["bis_sef_oracle"].values,
                df["bis_emg_oracle"].values)
    return (df["bsr_quazi"].values,
            df["sef95"].values,
            df["emg_proxy"].values)


def partition(df: pd.DataFrame, oracle: bool):
    bsr, sef, emg = gate_features(df, oracle)
    deep_mask  = bsr > 49.8
    mid_mask   = (~deep_mask) & (emg < 34.2) & (sef < 20.2)
    awake_mask = (~deep_mask) & (~mid_mask)
    return {"deep": deep_mask, "mid": mid_mask, "awake": awake_mask}


def train_region_lgbm(X: np.ndarray, y: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      feature_names: list[str], region: str):
    params = {
        "objective": "regression_l1",
        "metric": "l1",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_names)
    if len(X_val) >= 10:
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)
        booster = lgb.train(
            params, dtrain, num_boost_round=2000,
            valid_sets=[dtrain, dval], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
    else:
        # too few val rows in this region (the deep gate sometimes carves
        # out ~all of one region) — fixed 500 trees, no early stop
        booster = lgb.train(params, dtrain, num_boost_round=500)
    print(f"  region={region!r}: trained on {len(X):,} rows, "
          f"val rows={len(X_val):,}, best_iter={booster.best_iteration}")
    return booster


def hybrid_predict(df_eval: pd.DataFrame, models: dict, feature_cols: list[str],
                   oracle: bool) -> np.ndarray:
    n = len(df_eval)
    pred = np.zeros(n)
    masks = partition(df_eval, oracle)
    X = df_eval[feature_cols].values

    # Deep: Ellerkmann formula on the oracle BSR (or raw quazi if --gate-raw)
    bsr_used, _, _ = gate_features(df_eval, oracle)
    pred[masks["deep"]] = 44.1 - bsr_used[masks["deep"]] / 2.25

    for r in ("mid", "awake"):
        if masks[r].any():
            pred[masks[r]] = models[r].predict(X[masks[r]])
    return np.clip(pred, 0.0, 100.0)


def per_regime_mae(actual: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        m = (actual >= lo) & (actual < hi)
        out[lbl] = float(np.mean(np.abs(pred[m] - actual[m]))) if m.sum() > 10 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    gate_group = ap.add_mutually_exclusive_group(required=True)
    gate_group.add_argument("--gate-oracle", action="store_true",
                            help="Use Vista BIS/SR, BIS/SEF, BIS/EMG for gating (Lee's actual inputs).")
    gate_group.add_argument("--gate-raw", action="store_true",
                            help="Use bsr_quazi, sef95, emg_proxy for gating (standalone library).")
    args = ap.parse_args()

    oracle = bool(args.gate_oracle)
    label = "oracle" if oracle else "raw"
    print(f"\n=== Hybrid rule-then-LightGBM (gate={label}) ===")

    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val)
    print(f"Train: {args.train.name}  {len(train_df):,} rows / "
          f"{train_df['case_id'].nunique()} cases")
    print(f"Val:   {args.val.name}  {len(val_df):,} rows / "
          f"{val_df['case_id'].nunique()} cases")

    feature_cols = [c for c in train_df.columns
                    if c not in ("target", "sqi", "case_id", "time_sec")]
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # Partition both train and val
    tparts = partition(train_df, oracle)
    vparts = partition(val_df, oracle)
    print("\nPartition sizes:")
    print(f"  {'region':<8s}  {'train':>10s}  {'val':>10s}  {'rule fires':>15s}")
    for r in ("deep", "mid", "awake"):
        print(f"  {r:<8s}  {int(tparts[r].sum()):10,d}  {int(vparts[r].sum()):10,d}  "
              f"{100*tparts[r].mean():6.2f}% train")

    # Train per-region models (deep uses formula, no model)
    models = {}
    for r in ("mid", "awake"):
        tsub = train_df.loc[tparts[r]]
        vsub = val_df.loc[vparts[r]]
        X = tsub[feature_cols].values
        y = tsub["target"].values
        Xv = vsub[feature_cols].values
        yv = vsub["target"].values
        models[r] = train_region_lgbm(X, y, Xv, yv, feature_cols, r)
        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = RESULTS_DIR / f"lgbm_hybrid_{label}_{r}.txt"
        models[r].save_model(str(out_path))

    # Evaluate
    pred = hybrid_predict(val_df, models, feature_cols, oracle)
    actual = val_df["target"].values
    baseline = val_df["openibis_quazi"].values

    def safe_metrics(p, a):
        m = ~np.isnan(p) & ~np.isnan(a)
        if m.sum() < 2:
            return float("nan"), float("nan"), float("nan")
        return (float(np.mean(np.abs(p[m] - a[m]))),
                float(np.corrcoef(p[m], a[m])[0, 1]),
                lin_concordance(p[m], a[m]))

    print(f"\n=== Val cohort (epoch-weighted) ===")
    print(f"{'variant':<32s}  {'MAE':>6s}  {'r':>6s}  {'Lin_rc':>7s}")
    for name, p in [("openibis(quazi, paper) base", baseline),
                    (f"Hybrid Phase 3e ({label})", pred)]:
        mae, r, rc = safe_metrics(p, actual)
        print(f"  {name:<30s}  {mae:6.2f}  {r:6.3f}  {rc:7.3f}")

    print(f"\n=== Per-regime MAE ===")
    print(f"{'variant':<32s}  " + "  ".join(f"{lbl:>7s}" for lbl in LEE_BIN_LABELS))
    for name, p in [("openibis(quazi, paper) base", baseline),
                    (f"Hybrid Phase 3e ({label})", pred)]:
        reg = per_regime_mae(actual, p)
        print(f"  {name:<30s}  " + "  ".join(
            f"{reg[k]:>7.2f}" if not np.isnan(reg[k]) else f"{'nan':>7s}"
            for k in LEE_BIN_LABELS))

    # Top-5 importance per region
    for r in ("mid", "awake"):
        imp = models[r].feature_importance(importance_type="gain")
        order = np.argsort(imp)[::-1][:5]
        print(f"\nTop-5 importance ({r}):")
        for i in order:
            print(f"  {feature_cols[i]:<22s}  {imp[i]:>10.1f}")


if __name__ == "__main__":
    main()
