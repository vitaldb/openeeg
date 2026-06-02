"""Final per-function performance report on the W=15 val cohort.

Every predict_bis_* function in the public API is called on raw EEG
(case by case, via multiprocessing pool) and the 1 Hz output is
compared against actual Vista BIS.

All functions now route their raw output through the canonical
``openeeg.postprocess.apply_ellerkmann_and_smooth`` wrapper, so the
Ellerkmann deep override is applied identically to every model.

Output: results/function_performance.{csv,txt}
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS, lin_concordance

RESULTS = Path(__file__).resolve().parents[1] / "results"


def w15_filter(df, oracle_w_csv):
    w = pd.read_csv(oracle_w_csv)
    keep = set(w.loc[w["oracle_W"] == 15, "case_id"].astype(int))
    return df[df["case_id"].isin(keep)].reset_index(drop=True)


def _run_all_funcs_for_case(cid: int) -> pd.DataFrame:
    """Worker: run every public predict_bis_* function on one case."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from openeeg.cohort import load_case, preprocess_eeg
    from openeeg import (
        predict_bis, predict_bis_rules,
        bis_morimoto_betaratio, bis_morimoto_bcsef, bis_morimoto_combined,
        bis_cusenza_fdsr, bis_sleigh_betaratio_gated,
    )
    case = load_case(int(cid))
    if case is None:
        return None
    eeg = preprocess_eeg(case["eeg"])
    funcs = [
        ("predict_bis", predict_bis),
        ("predict_bis_rules", predict_bis_rules),
        ("bis_morimoto_betaratio", bis_morimoto_betaratio),
        ("bis_morimoto_bcsef", bis_morimoto_bcsef),
        ("bis_morimoto_combined", bis_morimoto_combined),
        ("bis_cusenza_fdsr", bis_cusenza_fdsr),
        ("bis_sleigh_betaratio_gated", bis_sleigh_betaratio_gated),
    ]
    out = {}
    for name, fn in funcs:
        try:
            p = fn(eeg)
            out[name] = np.asarray(p, dtype=np.float32)
        except Exception as e:
            print(f"  [case {cid}] {name} failed: {e}", flush=True)
            out[name] = None
    n_max = max(len(v) for v in out.values() if v is not None)
    df = pd.DataFrame(dict(
        case_id=np.full(n_max, int(cid), dtype=np.int32),
        time_sec=np.arange(n_max, dtype=np.int32),
    ))
    for name, p in out.items():
        if p is None:
            df[name] = np.nan
        else:
            n = min(len(p), n_max)
            arr = np.full(n_max, np.nan, dtype=np.float32)
            arr[:n] = p[:n]
            df[name] = arr
    return df


def metrics(actual, pred):
    m = np.isfinite(actual) & np.isfinite(pred)
    if m.sum() < 20:
        return float("nan"), float("nan"), float("nan")
    a, p = actual[m], pred[m]
    mae = float(np.mean(np.abs(p - a)))
    r = float(np.corrcoef(p, a)[0, 1])
    rc = lin_concordance(p, a)
    return mae, r, rc


def per_regime_mae(actual, pred):
    m = np.isfinite(actual) & np.isfinite(pred)
    a, p = actual[m], pred[m]
    out = {}
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        mm = (a >= lo) & (a < hi)
        out[lbl] = float(np.mean(np.abs(p[mm] - a[mm]))) if mm.sum() > 10 else float("nan")
    return out


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "64")
    print("Loading val W=15 (v5) targets...")
    val = pd.read_parquet(RESULTS / "features_val_n100_v5.parquet")
    vl = w15_filter(val, RESULTS / "oracle_W_val.csv").reset_index(drop=True)
    print(f"  {len(vl):,} rows / {vl['case_id'].nunique()} cases")

    cache = RESULTS / "function_predictions_val.parquet"
    if cache.exists():
        print(f"  loading cached {cache.name}")
        out = pd.read_parquet(cache)
    else:
        cids = sorted(vl["case_id"].unique())
        print(f"\nRunning all 7 predict_bis_* functions on {len(cids)} cases (32 workers)...")
        t0 = time.time()
        with Pool(32) as pool:
            pieces = [p for p in pool.map(_run_all_funcs_for_case, [int(c) for c in cids])
                      if p is not None]
        out = pd.concat(pieces, ignore_index=True)
        out.to_parquet(cache, index=False, compression="zstd")
        print(f"  done {time.time()-t0:.1f}s — cached {cache.name}")

    # Merge predictions with targets
    print("\nMerging predictions ↔ targets...")
    merged = vl[["case_id", "time_sec", "target"]].merge(
        out, on=["case_id", "time_sec"], how="left")
    print(f"  coverage: {merged.notna().mean().min()*100:.1f}% on lowest column")

    actual = merged["target"].values
    func_names = [
        "predict_bis",
        "predict_bis_rules",
        "bis_morimoto_bcsef",
        "bis_morimoto_combined",
        "bis_morimoto_betaratio",
        "bis_cusenza_fdsr",
        "bis_sleigh_betaratio_gated",
    ]

    rows = []
    for name in func_names:
        pred = merged[name].values
        mae, r, rc = metrics(actual, pred)
        reg = per_regime_mae(actual, pred)
        rows.append(dict(function=name, mae=mae, r=r, rc=rc, **reg))
    summary = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    summary.to_csv(RESULTS / "function_performance.csv", index=False)
    cols = ["function", "mae", "r", "rc"] + list(LEE_BIN_LABELS)
    print("\n=== Per-function performance (val W=15, SQI ≥ 80, EMA(15s)) ===")
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    txt = ["openeeg public-API function performance report",
            "=" * 50,
            f"Cohort: {len(vl):,} epochs, {vl['case_id'].nunique()} val cases "
            f"(W=15, SQI≥80, EMA(15s))",
            "All functions route through "
            "openeeg.postprocess.apply_ellerkmann_and_smooth.",
            ""]
    txt.append(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    (RESULTS / "function_performance.txt").write_text("\n".join(txt))
    print(f"\nSaved results/function_performance.csv and .txt")


if __name__ == "__main__":
    main()
