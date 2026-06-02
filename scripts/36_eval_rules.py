"""Evaluate the literature-cited rule-based BIS surrogates on the W=15 val cohort.

Rules tested
------------
* Morimoto 2004 BetaRatio:  BIS = 19.3·BetaRatio + 93.3
* Morimoto 2004 BcSEF:      BIS = 2.3·BcSEF + 12.0
* Morimoto 2004 combined:   sigmoid blend at BIS ≈ 70
* Cusenza 2013 FDSR:        BIS = a·(FD_Higuchi · (1 − 0.8·BSR/100)) + b
* Sleigh-gated Morimoto:    SE50d gate → BcSEF or BetaRatio per epoch

We also calibrate the Cusenza affine constants ``(a, b)`` on the train
cohort and persist them so the deployable rule uses the right scale.

For each rule, we report:
  overall MAE (full val) / 0-21 / 21-41 / 41-61 / 61-78 / 78-98
  also: MAE + Ellerkmann override (openbsr > 49.8)

Outputs
  results/rules_eval.csv
  results/rules_eval.txt
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


def ema(x, W=15.0):
    a = 2.0 / (W + 1.0)
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    return y


def smooth_by_case(pred, df, W=15.0):
    out = np.empty_like(pred, dtype=float)
    for cid, sub in df.groupby("case_id"):
        idx = sub.index.to_numpy()
        out[idx] = ema(pred[idx], W)
    return out


def per_regime_mae(actual, pred):
    m = np.isfinite(actual) & np.isfinite(pred)
    a, p = actual[m], pred[m]
    out = {}
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        mm = (a >= lo) & (a < hi)
        out[lbl] = float(np.mean(np.abs(p[mm] - a[mm]))) if mm.sum() > 10 else float("nan")
    return out


def metrics(actual, pred):
    m = np.isfinite(actual) & np.isfinite(pred)
    if m.sum() < 10:
        return float("nan"), float("nan"), float("nan")
    a, p = actual[m], pred[m]
    return float(np.mean(np.abs(p - a))), float(np.corrcoef(p, a)[0, 1]), lin_concordance(p, a)


# ------ Per-case workers ---------------------------------------------------

def _eval_one_case(args):
    """Compute the NEW rule features (Higuchi FD, SE50d) at 1 Hz for one case.

    Existing features (beta_ratio, bcsef, sef95, openbsr) live in
    ``features_*_v5.parquet`` already, so we pull them from there rather
    than recomputing — guarantees the merge key matches.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    cid = int(args)
    from openeeg.cohort import load_case, preprocess_eeg
    from openeeg.features import higuchi_fd, se50d

    case = load_case(cid)
    if case is None:
        return None
    eeg = preprocess_eeg(case["eeg"])

    fd_2hz = higuchi_fd(eeg)
    se_2hz = se50d(eeg)
    n_2hz = min(len(fd_2hz), len(se_2hz))
    # Each 2 Hz epoch = 0.5 s; take every other → 1 Hz, time_sec = 0, 1, 2, ...
    n_1hz = n_2hz // 2
    fd_1hz = fd_2hz[: n_1hz * 2 : 2]
    se_1hz = se_2hz[: n_1hz * 2 : 2]
    return pd.DataFrame(dict(
        case_id=np.full(n_1hz, cid, dtype=np.int32),
        time_sec=np.arange(n_1hz, dtype=np.int32),
        higuchi_fd=fd_1hz.astype(np.float32),
        se50d_v=se_1hz.astype(np.float32),
    ))


def compute_rule_features(cohort_df, label, n_workers=32):
    """Compute (or load) per-case rule features for a cohort."""
    cache = RESULTS / f"rule_features_{label}.parquet"
    if cache.exists():
        print(f"  loading cached {cache.name}")
        return pd.read_parquet(cache)
    cids = sorted(cohort_df["case_id"].unique())
    print(f"  computing rule features for {len(cids)} cases ({n_workers} workers)...")
    t0 = time.time()
    with Pool(n_workers) as pool:
        pieces = [p for p in pool.map(_eval_one_case, [int(c) for c in cids]) if p is not None]
    df = pd.concat(pieces, ignore_index=True)
    df.to_parquet(cache, index=False, compression="zstd")
    print(f"  done {time.time() - t0:.1f}s — cached {cache.name}")
    return df


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "64")

    print("Loading W=15 train & val targets...")
    train = pd.read_parquet(RESULTS / "features_train_n500_v5.parquet")
    val   = pd.read_parquet(RESULTS / "features_val_n100_v5.parquet")
    tr = w15_filter(train, RESULTS / "oracle_W_train.csv").reset_index(drop=True)
    vl = w15_filter(val,   RESULTS / "oracle_W_val.csv").reset_index(drop=True)
    print(f"  train: {len(tr):,} rows / {tr['case_id'].nunique()} cases")
    print(f"  val:   {len(vl):,} rows / {vl['case_id'].nunique()} cases")

    # ---- Compute NEW rule features (Higuchi FD, SE50d) for both cohorts ----
    print("\nComputing rule features for train (one-time cache)...")
    tr_rf = compute_rule_features(tr, "train")
    print("\nComputing rule features for val (one-time cache)...")
    vl_rf = compute_rule_features(vl, "val")

    # Merge with parquet (which already has beta_ratio, bcsef, sef95, openbsr)
    tr_m = tr[["case_id", "time_sec", "target", "beta_ratio", "bcsef",
                "sef95", "openbsr"]].merge(tr_rf, on=["case_id", "time_sec"], how="left")
    vl_m = vl[["case_id", "time_sec", "target", "beta_ratio", "bcsef",
                "sef95", "openbsr"]].merge(vl_rf, on=["case_id", "time_sec"], how="left")
    print(f"  merge coverage train: {tr_m['higuchi_fd'].notna().mean()*100:.1f}%   "
          f"val: {vl_m['higuchi_fd'].notna().mean()*100:.1f}%")

    # ---- Calibrate Cusenza FDSR affine constants on train ----
    print("\nCalibrating Cusenza FDSR a, b on train fold...")
    fdsr_tr = tr_m["higuchi_fd"].values * (1.0 - 0.8 * tr_m["openbsr"].values / 100.0)
    y_tr = tr_m["target"].values
    mask = np.isfinite(fdsr_tr) & np.isfinite(y_tr)
    a_cal, b_cal = np.polyfit(fdsr_tr[mask], y_tr[mask], 1)
    print(f"  fitted: a={a_cal:+.3f}  b={b_cal:+.3f}")

    # ---- Build rule predictions on val (Ellerkmann ALWAYS applied on RAW) ----
    # Pipeline (corrected): raw rule output → Ellerkmann override (raw values)
    # → EMA(15s) once. Previously the override was applied AFTER smoothing and
    # then re-smoothed, which diluted the deep correction.
    obsr_v = vl_m["openbsr"].values
    deep_mask_v = np.where(np.isnan(obsr_v), False, obsr_v > 49.8)
    ellerk_v = np.clip(44.1 - obsr_v / 2.25, 0, 100)
    actual = vl_m["target"].values

    def apply_ellerkmann_then_smooth(raw_pred):
        """Mandatory Ellerkmann override on raw values, then single smooth."""
        p = np.clip(raw_pred, 0, 100).copy()
        p[deep_mask_v] = ellerk_v[deep_mask_v]
        return smooth_by_case(p, vl, W=15.0)

    preds = {}
    # Morimoto BetaRatio
    pred_br_rule = 19.3 * vl_m["beta_ratio"].values + 93.3
    preds["morimoto_betaratio"] = apply_ellerkmann_then_smooth(pred_br_rule)
    # Morimoto BcSEF
    pred_bc_rule = 2.3 * vl_m["bcsef"].values + 12.0
    preds["morimoto_bcsef"] = apply_ellerkmann_then_smooth(pred_bc_rule)
    # Morimoto combined (sigmoid blend)
    bc_c = np.clip(pred_bc_rule, 0, 100)
    br_c = np.clip(pred_br_rule, 0, 100)
    w = 1.0 / (1.0 + np.exp(-(bc_c - 70.0) / 5.0))
    pred_comb = (1 - w) * bc_c + w * br_c
    preds["morimoto_combined"] = apply_ellerkmann_then_smooth(pred_comb)
    # Cusenza FDSR
    fdsr_v = vl_m["higuchi_fd"].values * (1.0 - 0.8 * vl_m["openbsr"].values / 100.0)
    pred_fdsr = a_cal * fdsr_v + b_cal
    preds["cusenza_fdsr"] = apply_ellerkmann_then_smooth(pred_fdsr)
    # Sleigh-gated Morimoto
    se = vl_m["se50d_v"].values
    w_awake = np.where(np.isnan(se), 0.5,
                        np.where(se < 17, 0.0, np.where(se < 21, 0.5, 1.0)))
    pred_sleigh = (1 - w_awake) * bc_c + w_awake * br_c
    preds["sleigh_gated_morimoto"] = apply_ellerkmann_then_smooth(pred_sleigh)

    # ---- Build summary ----
    rows = []
    for name, p in preds.items():
        mae, r, rc = metrics(actual, p)
        per = per_regime_mae(actual, p)
        rows.append(dict(rule=name, mae=mae, r=r, rc=rc, **per))
    summary = pd.DataFrame(rows).sort_values("mae")
    summary.to_csv(RESULTS / "rules_eval.csv", index=False)

    cols = ["rule", "mae", "r", "rc"] + list(LEE_BIN_LABELS)
    print("\n=== Val (W=15, SQI ≥ 80, EMA(15s)) — rule-based BIS surrogates ===")
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ---- Compare to current ship baselines ----
    print("\nReference numbers for comparison:")
    print(f"  openibis(quazi,paper) + EMA(15s):   MAE 5.08")
    print(f"  Lee 2019 oracle reproduction:       MAE 6.16")
    print(f"  predict_bis_rules (Phase D2):       MAE 4.39")
    print(f"  predict_bis (v3, +Ellerkmann):      MAE 3.67")
    print(f"  predict_bis_v2 (LightGBM):          MAE 3.63")

    # Text report
    rep = ["Rule-based BIS surrogate evaluation",
           "===================================",
           f"Cohort: {len(vl):,} epochs, {vl['case_id'].nunique()} cases (val W=15, SQI≥80)",
           f"Cusenza FDSR calibration constants:  a={a_cal:+.3f}  b={b_cal:+.3f}",
           ""]
    rep.append(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    out_txt = RESULTS / "rules_eval.txt"
    out_txt.write_text("\n".join(rep))
    print(f"\nSaved {out_txt}")

    # Also write the calibrated Cusenza constants back to openeeg/rules.py?
    # Leave that as a separate manual decision — print them so user can update.
    print(f"\nTo update openeeg/rules.py, set:")
    print(f"  _CUSENZA_A = {a_cal:+.3f}")
    print(f"  _CUSENZA_B = {b_cal:+.3f}")


if __name__ == "__main__":
    main()
