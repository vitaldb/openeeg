"""Augment an existing features parquet with oracle BIS/* tracks.

Adds:
  * ``bis_sef_oracle``    — BIS Vista SEF95 (Lee 2019 input)
  * ``bis_sr_oracle``     — BIS Vista Suppression Ratio (= Lee 2019 "BSR")
  * ``bis_totpow_oracle`` — BIS Vista total power, dB

Joined by (``case_id``, ``time_sec``) so existing rows / target / openibis
columns are unchanged. Much faster than re-extracting because no PSDs
are recomputed — each case is just a numpy slice off disk.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg.cohort import load_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.in_path)
    print(f"Loaded {args.in_path.name}: {len(df):,} rows × {len(df.columns)} cols")

    pieces = []
    cids = sorted(df["case_id"].unique())
    t0 = time.time()
    for i, cid in enumerate(cids, 1):
        case = load_case(int(cid))
        if case is None:
            print(f"  case {cid}: load failed, skipping")
            continue
        n = min(len(case["sef"]), len(case["sr"]), len(case["totpow"]))
        pieces.append(pd.DataFrame({
            "case_id": int(cid),
            "time_sec": np.arange(n, dtype=np.int32),
            "bis_sef_oracle":    case["sef"][:n].astype(np.float32),
            "bis_sr_oracle":     case["sr"][:n].astype(np.float32),
            "bis_totpow_oracle": case["totpow"][:n].astype(np.float32),
        }))
        if i % 50 == 0:
            print(f"  {i}/{len(cids)} cases   elapsed {time.time()-t0:.1f}s")
    aug = pd.concat(pieces, ignore_index=True)
    print(f"Augment table: {len(aug):,} rows")

    merged = df.merge(aug, on=["case_id", "time_sec"], how="left")
    print(f"Merged: {len(merged):,} rows × {len(merged.columns)} cols")
    new_cols = ["bis_sef_oracle", "bis_sr_oracle", "bis_totpow_oracle"]
    for c in new_cols:
        n_nan = int(merged[c].isna().sum())
        print(f"  {c}: NaN={n_nan:,} ({100*n_nan/len(merged):.2f}%)")
    merged.to_parquet(args.out, index=False, compression="zstd")
    print(f"Wrote {args.out.name}: {args.out.stat().st_size/1e6:.1f} MB, "
          f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
