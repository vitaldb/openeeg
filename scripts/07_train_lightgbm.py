"""Phase 3 — train a LightGBM regressor on engineered features.

Splits the cached val cohort (caseid % 10 == 8) into sub-train (80 of
100 cases by sorted caseid) and sub-val (20 cases), trains a single
LightGBM model with case-level grouping, and reports per-regime MAE
vs the openibis(quazi, paper) baseline already in the feature set.

The sub-train / sub-val split is a stepping stone — once the pipeline
is validated, we will download the proper train fold (residues 0-7)
and re-train.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS, lin_concordance

PARQUET = Path(__file__).resolve().parents[1] / "results" / "features_val.parquet"
MODEL_OUT = Path(__file__).resolve().parents[1] / "results" / "lgbm_phase3a.txt"


def per_regime_mae(actual: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        m = (actual >= lo) & (actual < hi)
        out[lbl] = float(np.mean(np.abs(pred[m] - actual[m]))) if m.sum() > 10 else float("nan")
    return out


def main():
    df = pd.read_parquet(PARQUET)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
    cases = sorted(df["case_id"].unique())
    print(f"  Cases: {len(cases)}  ({cases[0]} … {cases[-1]})")

    # Sub-split: first 80 cases by sorted id → train, last 20 → val
    train_cases = set(cases[:80])
    val_cases = set(cases[80:])
    train_mask = df["case_id"].isin(train_cases)
    val_mask = df["case_id"].isin(val_cases)
    print(f"  Sub-train: {int(train_mask.sum()):,} rows from {len(train_cases)} cases")
    print(f"  Sub-val:   {int(val_mask.sum()):,} rows from {len(val_cases)} cases")

    feature_cols = [c for c in df.columns if c not in ("target", "sqi", "case_id", "time_sec")]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "target"].values
    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, "target"].values

    # NaN handling: LightGBM accepts NaN; we trust it.
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)

    params = {
        "objective": "regression_l1",  # MAE
        "metric": "l1",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "num_threads": 0,
    }

    print("\nTraining LightGBM...")
    booster = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_OUT))
    print(f"\nSaved model: {MODEL_OUT.name}")

    pred = np.clip(booster.predict(X_val), 0.0, 100.0)
    baseline = df.loc[val_mask, "openibis_quazi"].values

    def safe_metrics(p, a):
        m = ~np.isnan(p) & ~np.isnan(a)
        if m.sum() < 2:
            return float("nan"), float("nan"), float("nan")
        return (
            float(np.mean(np.abs(p[m] - a[m]))),
            float(np.corrcoef(p[m], a[m])[0, 1]),
            lin_concordance(p[m], a[m]),
        )

    print("\n=== Sub-val cohort (20 cases) ===")
    print(f"{'variant':<28s}  {'MAE':>6s}  {'r':>6s}  {'Lin_rc':>7s}  {'N':>9s}")
    for name, p in [("openibis(quazi, paper)", baseline), ("LightGBM Phase 3a", pred)]:
        mae, r, rc = safe_metrics(p, y_val)
        n_valid = int((~np.isnan(p) & ~np.isnan(y_val)).sum())
        print(f"  {name:<26s}  {mae:6.2f}  {r:6.3f}  {rc:7.3f}  {n_valid:>9d}")

    def per_regime_mae_safe(actual, pred):
        m = ~np.isnan(actual) & ~np.isnan(pred)
        a, p = actual[m], pred[m]
        out = {}
        for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
            mm = (a >= lo) & (a < hi)
            out[lbl] = float(np.mean(np.abs(p[mm] - a[mm]))) if mm.sum() > 10 else float("nan")
        return out

    print("\n=== Per-regime MAE ===")
    print(f"{'variant':<28s}  " + "  ".join(f"{lbl:>7s}" for lbl in LEE_BIN_LABELS))
    for name, p in [("openibis(quazi, paper)", baseline), ("LightGBM Phase 3a", pred)]:
        reg = per_regime_mae_safe(y_val, p)
        print(f"  {name:<26s}  " + "  ".join(
            f"{reg[k]:>7.2f}" if not np.isnan(reg[k]) else f"{'nan':>7s}"
            for k in LEE_BIN_LABELS))

    # Per-case MAE table for a quick sanity scan
    print("\n=== Per-val-case MAE: LightGBM vs baseline ===")
    val_df = df.loc[val_mask].copy()
    val_df["pred_lgbm"] = pred
    print(f"  {'case':>5s}  {'N':>5s}  {'base':>5s}  {'lgbm':>5s}  {'Δ':>5s}")
    for cid in sorted(val_cases):
        sub = val_df[val_df["case_id"] == cid]
        b = float(np.mean(np.abs(sub["openibis_quazi"] - sub["target"])))
        l = float(np.mean(np.abs(sub["pred_lgbm"] - sub["target"])))
        print(f"  {cid:>5d}  {len(sub):>5d}  {b:5.2f}  {l:5.2f}  {l-b:+5.2f}")

    # Feature importance
    print("\n=== Top 10 feature importance (gain) ===")
    imp = booster.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1][:10]
    for i in order:
        print(f"  {feature_cols[i]:<22s}  {imp[i]:>10.1f}")


if __name__ == "__main__":
    main()
