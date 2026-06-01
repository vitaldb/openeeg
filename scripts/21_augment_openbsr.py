"""Augment val parquet with openbsr (Connor 2024) per epoch.

For each val case, compute openbsr from raw EEG and join on
(case_id, time_sec). Lets us test openbsr as a gate against the
BIS bins.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg import openbsr
from openeeg.cohort import load_case, preprocess_eeg

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    IN = RESULTS / "features_val_n100_v3.parquet"
    OUT = RESULTS / "features_val_n100_v4.parquet"
    df = pd.read_parquet(IN)
    print(f"Loaded {IN.name}: {len(df):,} rows")

    pieces = []
    cids = sorted(df["case_id"].unique())
    t0 = time.time()
    for i, cid in enumerate(cids, 1):
        case = load_case(int(cid))
        if case is None:
            print(f"  case {cid}: load failed")
            continue
        eeg = preprocess_eeg(case["eeg"])
        bsr_2hz = openbsr(eeg)
        # Downsample to 1 Hz (every 2nd, matching parquet stride)
        openbsr_1hz = bsr_2hz[::2]
        n_full = len(openbsr_1hz)
        pieces.append(pd.DataFrame({
            "case_id": int(cid),
            "time_sec": np.arange(n_full, dtype=np.int32),
            "openbsr": openbsr_1hz.astype(np.float32),
        }))
        if i % 20 == 0:
            print(f"  {i}/{len(cids)}  elapsed {time.time()-t0:.1f}s")

    aug = pd.concat(pieces, ignore_index=True)
    merged = df.merge(aug, on=["case_id", "time_sec"], how="left")
    n_nan = int(merged["openbsr"].isna().sum())
    print(f"Merge done: {len(merged):,} rows, openbsr NaN={n_nan:,} ({100*n_nan/len(merged):.2f}%)")
    merged.to_parquet(OUT, index=False, compression="zstd")
    print(f"Wrote {OUT.name}: {OUT.stat().st_size/1e6:.1f} MB, total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
