"""Phase 3g — train LightGBM on EMA- and Wiener-desmoothed targets.

Modes:

* ``--flavour {baseline,ema,wiener}`` (single-flavour, ~1 model
  output) trains and saves a single model. Run three of these in
  parallel to use multiple CPU sockets effectively.
* ``--flavour eval`` loads all three already-trained models and
  prints the full comparison table (no training).

Multi-process strategy: each training process pins itself to a
slice of OMP threads via ``OMP_NUM_THREADS`` / ``num_threads``,
so three concurrent jobs on a 64-core box stay below total
contention.

Baseline = model on the original smoothed actual (= existing
predict_bis_v1). EMA / Wiener targets are case-W-specific
desmoothings.

Outputs: ``results/desmooth_models/lgbm_<flavour>.txt``.
"""
from __future__ import annotations

import argparse
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
MODEL_DIR = RESULTS / "desmooth_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------- per-case smoothing-window estimation ----------

def estimate_W(bis: np.ndarray, lo: int = 10, hi: int = 60) -> int:
    """Return an integer W in [lo, hi] inferred from the BIS slew rate."""
    db = np.diff(bis)
    db = db[np.isfinite(db)]
    if len(db) < 30:
        return 30
    p99 = float(np.percentile(np.abs(db), 99))
    rng = float(np.nanmax(bis) - np.nanmin(bis))
    if p99 < 0.05 or rng < 20:
        return 30
    W = rng / p99
    return int(np.clip(round(W), lo, hi))


# ---------- desmoothers ----------

def desmooth_ema(y: np.ndarray, W: int) -> np.ndarray:
    alpha = 2.0 / (W + 1.0)
    x = np.empty(len(y))
    x[0] = y[0]
    x[1:] = (y[1:] - (1.0 - alpha) * y[:-1]) / alpha
    return np.clip(x, 0.0, 100.0)


def desmooth_wiener(y: np.ndarray, W: int, lam: float = 0.05) -> np.ndarray:
    n = len(y)
    if n < W * 4:
        return y.copy()
    h = np.zeros(n)
    h[:W] = 1.0 / W
    Y = np.fft.fft(y)
    H = np.fft.fft(h)
    X_hat = Y * np.conj(H) / (np.abs(H) ** 2 + lam)
    return np.clip(np.real(np.fft.ifft(X_hat)), 0.0, 100.0)


# ---------- re-smoothing (forward) ----------

def smooth_ema(x: np.ndarray, W: int) -> np.ndarray:
    alpha = 2.0 / (W + 1.0)
    y = np.empty(len(x))
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]
    return y


def smooth_uniform(x: np.ndarray, W: int) -> np.ndarray:
    """Causal trailing uniform mean."""
    kernel = np.ones(W) / W
    return np.convolve(x, kernel, mode="full")[: len(x)]


# ---------- per-case target augmentation ----------

def augment_targets(df: pd.DataFrame, flavour: str,
                    fixed_W: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Add case-specific W and a desmoothed target column.

    If ``fixed_W`` is given, use that window for every case instead of
    the per-case slew-rate estimate.
    """
    base = flavour.split("_fixed")[0]  # ema_fixed15 -> ema
    out = df.copy()
    out["target_desm"] = np.nan
    out["W_est"] = 0
    W_map = {}
    for cid, sub in df.groupby("case_id"):
        t = sub["target"].values
        if np.isnan(t).all():
            continue
        a = np.where(np.isnan(t), float(np.nanmean(t)), t)
        W = fixed_W if fixed_W is not None else estimate_W(a)
        W_map[int(cid)] = W
        if base == "ema":
            d = desmooth_ema(a, W)
        elif base == "wiener":
            d = desmooth_wiener(a, W, lam=0.05)
        else:
            raise ValueError(flavour)
        out.loc[sub.index, "target_desm"] = d
        out.loc[sub.index, "W_est"] = W
    return out, W_map


# ---------- training ----------

def _lgb_params(num_threads: int) -> dict:
    return dict(
        objective="regression_l1", metric="l1", learning_rate=0.05,
        num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
        bagging_fraction=0.8, bagging_freq=5, verbose=-1,
        num_threads=num_threads,
    )


def train_one(train_df: pd.DataFrame, val_df: pd.DataFrame,
              feat_cols: list[str], target_col: str, label: str,
              num_threads: int):
    X = train_df[feat_cols].values
    y = train_df[target_col].values
    Xv = val_df[feat_cols].values
    yv = val_df[target_col].values

    dtrain = lgb.Dataset(X, label=y, feature_name=feat_cols)
    dval = lgb.Dataset(Xv, label=yv, feature_name=feat_cols, reference=dtrain)
    print(f"\n--- training {label} (target='{target_col}', n_train={len(X):,}, threads={num_threads}) ---")
    t0 = time.time()
    booster = lgb.train(
        _lgb_params(num_threads), dtrain, num_boost_round=2000,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    out = MODEL_DIR / f"lgbm_{label}.txt"
    booster.save_model(str(out))
    print(f"  best_iter={booster.best_iteration}  {time.time()-t0:.1f}s  -> {out.name}")
    return booster


# ---------- evaluation ----------

def safe_metrics(p, a):
    m = ~np.isnan(p) & ~np.isnan(a)
    if m.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(np.abs(p[m] - a[m]))),
        float(np.corrcoef(p[m], a[m])[0, 1]),
        lin_concordance(p[m], a[m]),
    )


def per_regime_mae(actual: np.ndarray, pred: np.ndarray):
    out = {}
    m = ~np.isnan(actual) & ~np.isnan(pred)
    a, p = actual[m], pred[m]
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        mm = (a >= lo) & (a < hi)
        out[lbl] = float(np.mean(np.abs(p[mm] - a[mm]))) if mm.sum() > 10 else float("nan")
    return out


def load_features():
    train_df = pd.read_parquet(RESULTS / "features_train_n500_v2.parquet")
    val_df   = pd.read_parquet(RESULTS / "features_val_n100_v2.parquet")
    feat_cols = [c for c in train_df.columns
                 if c not in ("target", "sqi", "case_id", "time_sec")
                 and not c.startswith("bis_")]
    return train_df, val_df, feat_cols


def cmd_train(flavour: str, num_threads: int, fixed_W: int | None = None):
    print(f"== Single-flavour training: {flavour}  fixed_W={fixed_W} ==")
    train_df, val_df, feat_cols = load_features()
    print(f"  train rows: {len(train_df):,}  val rows: {len(val_df):,}")
    if flavour == "baseline":
        train_one(train_df, val_df, feat_cols, "target", "baseline_smoothed", num_threads)
        return
    label_suffix = f"_fixed{fixed_W}" if fixed_W is not None else ""
    print(f"\nEstimating per-case W ({flavour}) for train + val...")
    train_aug, W_train = augment_targets(train_df, flavour, fixed_W=fixed_W)
    val_aug,   W_val   = augment_targets(val_df, flavour, fixed_W=fixed_W)
    pd.DataFrame({"case_id": list(W_val.keys()),
                  "W_est": list(W_val.values())}).to_csv(
        RESULTS / f"W_est_{flavour}{label_suffix}_val.csv", index=False)
    W_vals = list(W_train.values())
    print(f"  W (train): median={int(np.median(W_vals))}  "
          f"range=[{min(W_vals)}, {max(W_vals)}]")
    train_one(train_aug, val_aug, feat_cols, "target_desm",
              f"{flavour}_desmooth{label_suffix}", num_threads)


def cmd_eval():
    print("== Evaluation only ==")
    train_df, val_df, feat_cols = load_features()
    print(f"  val rows: {len(val_df):,}")

    booster_base = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_baseline_smoothed.txt"))
    booster_ema  = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_ema_desmooth.txt"))
    booster_wn   = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_wiener_desmooth.txt"))

    # Per-case W on val (recompute, cheap)
    _, W_val_ema = augment_targets(val_df, "ema")

    X_val = val_df[feat_cols].values
    pred_baseline = np.clip(booster_base.predict(X_val), 0, 100)
    pred_ema_raw  = np.clip(booster_ema.predict(X_val),  0, 100)
    pred_wn_raw   = np.clip(booster_wn.predict(X_val),   0, 100)

    pred_ema_resmoothed = np.empty_like(pred_ema_raw)
    pred_wn_resmoothed  = np.empty_like(pred_wn_raw)
    for cid, sub in val_df.groupby("case_id"):
        idx = sub.index.to_numpy()
        W = W_val_ema.get(int(cid), 30)
        pred_ema_resmoothed[idx] = smooth_ema(pred_ema_raw[idx], W)
        pred_wn_resmoothed[idx]  = smooth_uniform(pred_wn_raw[idx], W)

    actual = val_df["target"].values

    print("\n=== Val cohort comparison (vs ORIGINAL smoothed actual) ===")
    print(f"{'variant':<34s}  {'MAE':>6s}  {'r':>6s}  {'Lin_rc':>7s}")
    for name, p in [
        ("baseline (current predict_bis)", pred_baseline),
        ("EMA-desmooth raw pred",          pred_ema_raw),
        ("EMA-desmooth re-smoothed @ Wcs", pred_ema_resmoothed),
        ("Wiener-desmooth raw pred",       pred_wn_raw),
        ("Wiener-desmooth re-smoothed",    pred_wn_resmoothed),
    ]:
        mae, r, rc = safe_metrics(p, actual)
        print(f"  {name:<32s}  {mae:6.2f}  {r:6.3f}  {rc:7.3f}")

    print("\n=== Per-regime MAE (vs ORIGINAL smoothed actual) ===")
    print(f"{'variant':<34s}  " + "  ".join(f"{lbl:>7s}" for lbl in LEE_BIN_LABELS))
    for name, p in [
        ("baseline",                pred_baseline),
        ("EMA-desmooth (raw)",      pred_ema_raw),
        ("EMA-desmooth (resmooth)", pred_ema_resmoothed),
        ("Wiener (raw)",            pred_wn_raw),
        ("Wiener (resmooth)",       pred_wn_resmoothed),
    ]:
        r = per_regime_mae(actual, p)
        print(f"  {name:<32s}  " + "  ".join(
            f"{r[k]:>7.2f}" if not np.isnan(r[k]) else f"{'nan':>7s}"
            for k in LEE_BIN_LABELS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flavour", required=True,
                    choices=["baseline", "ema", "wiener", "eval"])
    ap.add_argument("--threads", type=int, default=0,
                    help="num_threads for LightGBM (0 = use all available).")
    ap.add_argument("--fixed-W", type=int, default=None,
                    help="Fixed smoothing window (s) for every case; "
                         "overrides per-case estimation.")
    args = ap.parse_args()
    if args.flavour == "eval":
        cmd_eval()
    else:
        cmd_train(args.flavour, args.threads, fixed_W=args.fixed_W)


if __name__ == "__main__":
    main()
