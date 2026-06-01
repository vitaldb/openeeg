"""Phase 3f — augment train/val parquets with short-context + velocity features.

Adds per-case-causal columns on top of the existing 15-feature parquet
without re-running PSDs:

  * ``openibis_quazi_5s``, ``openibis_quazi_10s``, ``openibis_quazi_60s``
    — trailing means at three additional windows (the current parquet
    only has the 30 s window, which is the dominant feature).
  * ``openibis_quazi_dt`` — first difference of openibis_quazi (per-second
    velocity / transition rate).
  * ``openibis_quazi_30s_dt`` — first difference of the existing 30 s
    rolling mean (smoothed transition rate; for emergence detection).
  * ``sef95_dt``, ``emg_proxy_dt`` — first differences of two other
    informative features.

These additions cost ~5 columns × float32 × 4 M rows = ~80 MB extra
on train, ~16 MB extra on val. Run augment_oracle separately before
this if you also want the BIS/SR / BIS/SEF columns.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def trailing_mean_causal(x: np.ndarray, W: int) -> np.ndarray:
    """Strictly causal trailing mean over the last W samples (NaN-safe)."""
    x_f = np.where(np.isnan(x), 0.0, x).astype(np.float64)
    valid = (~np.isnan(x)).astype(np.float64)
    cs_x = np.concatenate([[0.0], np.cumsum(x_f)])
    cs_v = np.concatenate([[0.0], np.cumsum(valid)])
    out = np.full(len(x), np.nan, dtype=np.float32)
    for t in range(len(x)):
        lo = max(0, t + 1 - W)
        num = cs_x[t + 1] - cs_x[lo]
        den = cs_v[t + 1] - cs_v[lo]
        if den > 0:
            out[t] = num / den
    return out


def add_velocity(x: np.ndarray) -> np.ndarray:
    """First-difference velocity (Δ per 1-Hz sample); first element = 0."""
    out = np.empty_like(x, dtype=np.float32)
    out[0] = 0.0
    out[1:] = (x[1:] - x[:-1]).astype(np.float32)
    return out


def augment_case(sub: pd.DataFrame) -> pd.DataFrame:
    pq = sub["openibis_quazi"].values.astype(np.float64)
    pq30 = sub["openibis_quazi_30s"].values.astype(np.float64)
    sef = sub["sef95"].values.astype(np.float64)
    emg = sub["emg_proxy"].values.astype(np.float64)

    out = sub.copy()
    out["openibis_quazi_5s"]  = trailing_mean_causal(pq, 5)
    out["openibis_quazi_10s"] = trailing_mean_causal(pq, 10)
    out["openibis_quazi_60s"] = trailing_mean_causal(pq, 60)
    out["openibis_quazi_dt"]  = add_velocity(pq)
    out["openibis_quazi_30s_dt"] = add_velocity(pq30)
    out["sef95_dt"]      = add_velocity(sef)
    out["emg_proxy_dt"]  = add_velocity(emg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"Loading {args.in_path.name} ...")
    df = pd.read_parquet(args.in_path)
    print(f"  {len(df):,} rows × {len(df.columns)} cols")

    pieces = []
    t0 = time.time()
    cids = sorted(df["case_id"].unique())
    for i, cid in enumerate(cids, 1):
        sub = df[df["case_id"] == cid].sort_values("time_sec").reset_index(drop=True)
        pieces.append(augment_case(sub))
        if i % 50 == 0:
            print(f"  {i}/{len(cids)}  elapsed {time.time()-t0:.1f}s")
    out = pd.concat(pieces, ignore_index=True)

    new_cols = ["openibis_quazi_5s", "openibis_quazi_10s", "openibis_quazi_60s",
                "openibis_quazi_dt", "openibis_quazi_30s_dt",
                "sef95_dt", "emg_proxy_dt"]
    for c in new_cols:
        out[c] = out[c].astype(np.float32)

    out.to_parquet(args.out, index=False, compression="zstd")
    print(f"\nWrote {args.out.name}: {len(out):,} rows × {len(out.columns)} cols, "
          f"{args.out.stat().st_size/1e6:.1f} MB, {time.time()-t0:.1f}s")
    print(f"New columns: {new_cols}")


if __name__ == "__main__":
    main()
