"""Phase 3 — train a LightGBM regressor on engineered features.

Usage::

    # Phase 3a/3b — sub-split a single val parquet (legacy):
    python scripts/07_train_lightgbm.py --self-split results/features_val.parquet

    # Phase 3c — proper train/val split:
    python scripts/07_train_lightgbm.py \
        --train results/features_train_n500.parquet \
        --val   results/features_val_n100.parquet
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

MODEL_OUT = Path(__file__).resolve().parents[1] / "results" / "lgbm.txt"


def per_regime_mae(actual: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        m = (actual >= lo) & (actual < hi)
        out[lbl] = float(np.mean(np.abs(pred[m] - actual[m]))) if m.sum() > 10 else float("nan")
    return out


def _split_args(ap: argparse.ArgumentParser):
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-split", type=Path,
                   help="Single parquet: first 80%% of cases → train, last 20%% → val.")
    g.add_argument("--train", type=Path, help="Train parquet (use with --val).")
    ap.add_argument("--val", type=Path, help="Val parquet.")
    ap.add_argument("--weight-deep", type=float, default=1.0,
                    help="Multiplier applied to training rows with target<30. Default 1.0 (no weighting).")


def main():
    ap = argparse.ArgumentParser()
    _split_args(ap)
    args = ap.parse_args()

    if args.self_split is not None:
        df = pd.read_parquet(args.self_split)
        cases = sorted(df["case_id"].unique())
        n_train = int(len(cases) * 0.8)
        train_cases = set(cases[:n_train])
        val_cases = set(cases[n_train:])
        train_df = df[df["case_id"].isin(train_cases)].reset_index(drop=True)
        val_df = df[df["case_id"].isin(val_cases)].reset_index(drop=True)
        print(f"Self-split {args.self_split.name}: {len(cases)} cases → "
              f"{n_train} train + {len(cases)-n_train} val")
    else:
        if args.val is None:
            ap.error("--val is required when --train is given.")
        train_df = pd.read_parquet(args.train)
        val_df = pd.read_parquet(args.val)
        print(f"Train: {args.train.name}  ({train_df['case_id'].nunique()} cases, "
              f"{len(train_df):,} rows)")
        print(f"Val:   {args.val.name}  ({val_df['case_id'].nunique()} cases, "
              f"{len(val_df):,} rows)")

    feature_cols = [c for c in train_df.columns
                    if c not in ("target", "sqi", "case_id", "time_sec")]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["target"].values

    # Sample weighting: deep BIS epochs (BIS<30) are rare; optionally
    # up-weight them so the L1 loss can't ignore the deep regime.
    weight_train = np.where(y_train < 30, args.weight_deep, 1.0)
    n_deep = int((y_train < 30).sum())
    print(f"  Sample weighting: {n_deep:,} deep (target<30) epochs × {args.weight_deep:g}, "
          f"{len(y_train)-n_deep:,} normal × 1.0")

    # NaN handling: LightGBM accepts NaN; we trust it.
    dtrain = lgb.Dataset(X_train, label=y_train, weight=weight_train, feature_name=feature_cols)
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
    baseline = val_df["openibis_quazi"].values

    # 2-stage: hard-switch to Ellerkmann when the *paper* BSR detector
    # fires (very low false-positive rate per Phase 0 analysis), AND the
    # quazi BSR exceeds 40% (confirms substantial suppression).
    # Ellerkmann 2004: BIS = 44.1 - BSR/2.25, R²=0.99 for BSR>40%.
    bsr_p = val_df["bsr_paper"].values
    bsr_q = val_df["bsr_quazi"].values
    ellerkmann = 44.1 - bsr_q / 2.25
    gate = (bsr_p > 5.0) & (bsr_q > 40.0)
    pred_2stage = np.clip(np.where(gate, ellerkmann, pred), 0.0, 100.0)
    print(f"  2-stage gate fires on {int(gate.sum()):,} / {len(gate):,} epochs "
          f"({100*gate.mean():.1f}%)")

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
    for name, p in [
        ("openibis(quazi, paper)", baseline),
        ("LightGBM Phase 3a", pred),
        ("LightGBM + Ellerkmann (3b)", pred_2stage),
    ]:
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
    for name, p in [
        ("openibis(quazi, paper)", baseline),
        ("LightGBM Phase 3a", pred),
        ("LightGBM + Ellerkmann (3b)", pred_2stage),
    ]:
        reg = per_regime_mae_safe(y_val, p)
        print(f"  {name:<26s}  " + "  ".join(
            f"{reg[k]:>7.2f}" if not np.isnan(reg[k]) else f"{'nan':>7s}"
            for k in LEE_BIN_LABELS))

    # Per-case MAE table for a quick sanity scan
    print("\n=== Per-val-case MAE: baseline / LightGBM / 2-stage ===")
    val_df_ext = val_df.copy()
    val_df_ext["pred_lgbm"] = pred
    val_df_ext["pred_2stage"] = pred_2stage
    print(f"  {'case':>5s}  {'N':>5s}  {'base':>5s}  {'lgbm':>5s}  {'2stg':>5s}  {'Δlgbm':>6s}")
    rows = []
    for cid in sorted(val_df_ext["case_id"].unique()):
        sub = val_df_ext[val_df_ext["case_id"] == cid]
        m_b = ~sub["openibis_quazi"].isna()
        b = float(np.mean(np.abs(sub.loc[m_b, "openibis_quazi"] - sub.loc[m_b, "target"]))) if m_b.any() else float("nan")
        l = float(np.mean(np.abs(sub["pred_lgbm"] - sub["target"])))
        t = float(np.mean(np.abs(sub["pred_2stage"] - sub["target"])))
        rows.append((cid, len(sub), b, l, t))
    # Print only the worst 10 and best 10 by LightGBM MAE for compactness
    rows.sort(key=lambda r: r[3], reverse=True)
    for cid, n, b, l, t in rows[:10] + [("...", "", "", "", "")] + rows[-10:]:
        if isinstance(cid, str):
            print(f"  {cid:>5s}  {n:>5s}  {'':>5s}  {'':>5s}  {'':>5s}  {'':>6s}")
        else:
            print(f"  {cid:>5d}  {n:>5d}  {b:5.2f}  {l:5.2f}  {t:5.2f}  {l-b:+6.2f}")

    # Feature importance
    print("\n=== Top 10 feature importance (gain) ===")
    imp = booster.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1][:10]
    for i in order:
        print(f"  {feature_cols[i]:<22s}  {imp[i]:>10.1f}")


if __name__ == "__main__":
    main()
