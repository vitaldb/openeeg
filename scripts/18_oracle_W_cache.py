"""Cache the per-case oracle smoothing window W ∈ {15, 30, 45}.

For each case, picks the W that minimises MAE between
``EMA(predict_bis_raw, W)`` and the original Vista actual BIS.
Uses the already-trained baseline model (predict_bis_v1 equivalent)
to produce the raw signal.

Output:
  results/oracle_W_train.csv
  results/oracle_W_val.csv

Each row: ``case_id, oracle_W, mae_W15, mae_W30, mae_W45, oracle_mae``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODEL = RESULTS / "desmooth_models" / "lgbm_baseline_smoothed.txt"
CANDIDATES = [15, 30, 45]


def ema(x: np.ndarray, W: float) -> np.ndarray:
    a = 2.0 / (W + 1.0)
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    return y


def cache_W_for(parquet: Path, out_csv: Path):
    print(f"\n=== {parquet.name} → {out_csv.name} ===")
    df = pd.read_parquet(parquet)
    feat = [c for c in df.columns
            if c not in ("target", "sqi", "case_id", "time_sec")
            and not c.startswith("bis_")
            and not c.endswith("_dt")
            and not c.endswith("_5s")
            and not c.endswith("_10s")
            and not c.endswith("_60s")]
    print(f"  rows: {len(df):,}  features for predict: {len(feat)} ({feat[:3]}…)")
    model = lgb.Booster(model_file=str(MODEL))
    pred_raw = np.clip(model.predict(df[feat].values), 0.0, 100.0)
    actual = df["target"].values

    rows = []
    for cid, sub in df.groupby("case_id"):
        idx = sub.index.to_numpy()
        a = actual[idx]
        p = pred_raw[idx]
        mae_per_W = {}
        for W in CANDIDATES:
            ps = ema(p, W)
            m = ~np.isnan(ps) & ~np.isnan(a)
            if m.sum() < 2:
                mae_per_W[W] = float("nan")
            else:
                mae_per_W[W] = float(np.mean(np.abs(ps[m] - a[m])))
        oracle_W = min(CANDIDATES, key=lambda W: (mae_per_W[W] if not np.isnan(mae_per_W[W]) else 1e9))
        rows.append({
            "case_id": int(cid),
            "N": len(sub),
            "oracle_W": oracle_W,
            "oracle_mae": mae_per_W[oracle_W],
            **{f"mae_W{W}": mae_per_W[W] for W in CANDIDATES},
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    counts = out["oracle_W"].value_counts().sort_index().to_dict()
    print(f"  W distribution: {counts}")
    print(f"  saved {out_csv.name}: {len(out)} cases")
    return out


def main():
    cache_W_for(RESULTS / "features_val_n100_v3.parquet",   RESULTS / "oracle_W_val.csv")
    cache_W_for(RESULTS / "features_train_n500_v3.parquet", RESULTS / "oracle_W_train.csv")


if __name__ == "__main__":
    main()
