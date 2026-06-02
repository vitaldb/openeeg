"""Phase A — head-to-head reproduction of Lee 2019 (Anesth Analg)
``BIS = piecewise linear regression with rule-gated 5 regions``.

Two variants compared on the W=15 / SQI≥80 sub-cohort:

  --gate oracle  : Vista tracks (bis_sr/sef/emg_oracle) + beta_ratio
  --gate raw     : new openbsr (Connor 2024 line-by-line port) + sef95
                    + emg_proxy + beta_ratio  (deployable, raw EEG only)

Outputs
  results/lee2019_replication_coefs.csv      — per-leaf OLS coefficients
                                                + Lee Table 2 reference rows
  results/lee2019_replication_scatter.png    — actual vs predicted scatter
  results/lee2019_replication_perbin.csv     — per-Lee-bin MAE
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS, lin_concordance

RESULTS = Path(__file__).resolve().parents[1] / "results"

LEE_TABLE2 = [
    # Lee 2019 Anesth Analg Table 2 reproduced from Agent 1 brief
    dict(variant="Lee2019", region="deep",  intercept=42.1, coef_BSR= 0.00, coef_EMG=0.00, coef_SEF=0.00, coef_RBR= 0.00),  # Lee deep formula has no per-feature coefs printed; placeholder zero
    dict(variant="Lee2019", region="light", intercept=29.9, coef_BSR=-0.42, coef_EMG=0.00, coef_SEF=0.00, coef_RBR= 0.01),
    dict(variant="Lee2019", region="surg",  intercept=65.2, coef_BSR=-3.01, coef_EMG=0.96, coef_SEF=3.84, coef_RBR=-8.70),
    dict(variant="Lee2019", region="trans", intercept=-57.6,coef_BSR=-0.42, coef_EMG=0.04, coef_SEF=0.91, coef_RBR= 3.06),
    dict(variant="Lee2019", region="awake", intercept= 5.3, coef_BSR=-1.43, coef_EMG=0.41, coef_SEF=2.55, coef_RBR= 4.26),
]


def w15_filter(df, oracle_w_csv):
    w_df = pd.read_csv(oracle_w_csv)
    keep = set(w_df.loc[w_df["oracle_W"] == 15, "case_id"].astype(int))
    return df[df["case_id"].isin(keep)].reset_index(drop=True)


# --- Lee's decision tree applied to four columns ---
def lee_partition(bsr, sef, emg, rbr):
    deep = bsr > 49.8
    mid_gate = (~deep) & (emg < 34.2) & (sef < 20.2)
    light = mid_gate & ((bsr > 2.1) | (sef < 14.8))
    surg  = mid_gate & ~light
    not_mid = (~deep) & (~mid_gate)
    trans = not_mid & (rbr < -0.7)
    awake = not_mid & (rbr >= -0.7)
    return {"deep": deep, "light": light, "surg": surg, "trans": trans, "awake": awake}


# --- Multiprocessing worker for new openbsr augmentation ---
def _augment_one_case(cid):
    """Worker — load case, compute new openbsr at 1 Hz."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from openeeg.cohort import load_case, preprocess_eeg
    from openeeg import openbsr as _openbsr
    case = load_case(int(cid))
    if case is None:
        return None
    eeg = preprocess_eeg(case["eeg"])
    ob_2hz = _openbsr(eeg)
    ob_1hz = ob_2hz[::2]
    n = len(ob_1hz)
    return pd.DataFrame({
        "case_id":   int(cid),
        "time_sec":  np.arange(n, dtype=np.int32),
        "openbsr":   ob_1hz.astype(np.float32),
    })


def augment_openbsr(df, n_workers=32):
    """Add the 'openbsr' column by recomputing per case in parallel."""
    cids = sorted(df["case_id"].unique())
    t0 = time.time()
    print(f"  Augmenting openbsr on {len(cids)} cases using {n_workers} workers...")
    with Pool(n_workers) as pool:
        pieces = pool.map(_augment_one_case, cids)
    pieces = [p for p in pieces if p is not None]
    aug = pd.concat(pieces, ignore_index=True)
    out = df.merge(aug, on=["case_id", "time_sec"], how="left")
    print(f"  done {time.time()-t0:.1f}s  NaN={out['openbsr'].isna().sum():,}")
    return out


# --- Per-region OLS fitting ---
def fit_region(X_tr, y_tr, mask_tr, X_v, y_v, mask_v):
    keep_tr = mask_tr & ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
    if keep_tr.sum() < 100:
        return None
    m = LinearRegression()
    m.fit(X_tr[keep_tr], y_tr[keep_tr])
    keep_v = mask_v & ~np.isnan(X_v).any(axis=1) & ~np.isnan(y_v)
    pred_tr = m.predict(X_tr[keep_tr])
    pred_v = m.predict(X_v[keep_v]) if keep_v.sum() > 0 else np.empty(0)
    return dict(
        model=m,
        intercept=float(m.intercept_),
        coef=m.coef_.copy(),
        n_train=int(keep_tr.sum()),
        n_val=int(keep_v.sum()),
        mae_train=float(np.mean(np.abs(pred_tr - y_tr[keep_tr]))),
        mae_val=float(np.mean(np.abs(pred_v - y_v[keep_v]))) if keep_v.sum() > 0 else float("nan"),
        pred_v_indices=np.where(keep_v)[0],
        pred_v=pred_v,
    )


def run_variant(name, bsr_c, sef_c, emg_c, rbr_c, train_w15, val_w15):
    """Apply Lee partition + per-region OLS on a single column choice."""
    print(f"\n=== Variant '{name}': BSR={bsr_c}, SEF={sef_c}, EMG={emg_c}, RBR={rbr_c} ===")

    bsr_t = train_w15[bsr_c].values; sef_t = train_w15[sef_c].values
    emg_t = train_w15[emg_c].values; rbr_t = train_w15[rbr_c].values
    bsr_v = val_w15[bsr_c].values;   sef_v = val_w15[sef_c].values
    emg_v = val_w15[emg_c].values;   rbr_v = val_w15[rbr_c].values

    parts_t = lee_partition(bsr_t, sef_t, emg_t, rbr_t)
    parts_v = lee_partition(bsr_v, sef_v, emg_v, rbr_v)

    print("  partition (train | val):")
    for r in ("deep", "light", "surg", "trans", "awake"):
        nt = int(parts_t[r].sum()); nv = int(parts_v[r].sum())
        print(f"    {r:<6s}  {nt:>8,d} | {nv:>7,d}  ({100*nt/len(train_w15):.2f}% | {100*nv/len(val_w15):.2f}%)")

    X_tr = np.column_stack([bsr_t, emg_t, sef_t, rbr_t])
    X_v  = np.column_stack([bsr_v, emg_v, sef_v, rbr_v])
    y_tr = train_w15["target"].values
    y_v  = val_w15["target"].values

    region_results = {}
    coef_rows = []
    pred_v_full = np.full(len(val_w15), np.nan)

    for region in ("deep", "light", "surg", "trans", "awake"):
        res = fit_region(X_tr, y_tr, parts_t[region], X_v, y_v, parts_v[region])
        if res is None:
            print(f"  {region:<6s}: n_train too small, skipped")
            continue
        region_results[region] = res
        pred_v_full[res["pred_v_indices"]] = np.clip(res["pred_v"], 0, 100)
        c = res["coef"]
        print(f"  {region:<6s}: n_tr={res['n_train']:>7,d}  n_v={res['n_val']:>6,d}  "
              f"intercept={res['intercept']:+7.2f}  "
              f"BSR={c[0]:+7.3f}  EMG={c[1]:+7.3f}  SEF={c[2]:+7.3f}  RBR={c[3]:+7.3f}  "
              f"MAE_v={res['mae_val']:5.2f}")
        coef_rows.append(dict(
            variant=name, region=region,
            intercept=res["intercept"],
            coef_BSR=float(c[0]), coef_EMG=float(c[1]), coef_SEF=float(c[2]), coef_RBR=float(c[3]),
            n_train=res["n_train"], n_val=res["n_val"],
            mae_train=res["mae_train"], mae_val=res["mae_val"],
        ))

    # Overall (combined per-region) MAE & r against actual BIS
    overall_mask = ~np.isnan(pred_v_full) & ~np.isnan(y_v)
    if overall_mask.sum() > 10:
        overall_mae = float(np.mean(np.abs(pred_v_full[overall_mask] - y_v[overall_mask])))
        overall_r = float(np.corrcoef(pred_v_full[overall_mask], y_v[overall_mask])[0, 1])
        overall_rc = lin_concordance(pred_v_full[overall_mask], y_v[overall_mask])
    else:
        overall_mae = float("nan"); overall_r = float("nan"); overall_rc = float("nan")
    print(f"  ---- overall val: MAE={overall_mae:.2f}  r={overall_r:.3f}  Lin's rc={overall_rc:.3f}")

    # Per-Lee-bin MAE
    perbin_rows = []
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        mm = overall_mask & (y_v >= lo) & (y_v < hi)
        if mm.sum() > 10:
            mae = float(np.mean(np.abs(pred_v_full[mm] - y_v[mm])))
            perbin_rows.append(dict(variant=name, bin=lbl, n=int(mm.sum()), mae=mae))

    return dict(
        coef_rows=coef_rows, perbin_rows=perbin_rows,
        pred_v=pred_v_full, overall_mae=overall_mae, overall_r=overall_r, overall_rc=overall_rc,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["oracle", "raw", "mixed", "all"], default="all",
                    help="oracle = Vista tracks; raw = our raw-EEG estimates; "
                         "mixed = raw EEG (openbsr/sef95/beta_ratio) + Vista EMG "
                         "(BIS sensor hardware provides 70-110 Hz EMG natively, "
                         "so this is the realistic 'deployable on BIS Vista' input set).")
    ap.add_argument("--workers", type=int, default=32,
                    help="multiprocessing workers for openbsr augmentation (only when --gate=raw/both).")
    args = ap.parse_args()

    print("Loading parquets...")
    train = pd.read_parquet(RESULTS / "features_train_n500_v3.parquet")
    val   = pd.read_parquet(RESULTS / "features_val_n100_v3.parquet")
    train_w15 = w15_filter(train, RESULTS / "oracle_W_train.csv")
    val_w15   = w15_filter(val,   RESULTS / "oracle_W_val.csv")
    print(f"  train W=15: {len(train_w15):,} rows / {train_w15['case_id'].nunique()} cases")
    print(f"  val   W=15: {len(val_w15):,} rows / {val_w15['case_id'].nunique()} cases")

    # If any variant needs openbsr, ensure it exists and cache the augmented
    # FULL parquets (v5) so later phases can reuse without recomputing.
    needs_openbsr = args.gate in ("raw", "mixed", "all")
    if needs_openbsr:
        train_v5 = RESULTS / "features_train_n500_v5.parquet"
        val_v5   = RESULTS / "features_val_n100_v5.parquet"
        if "openbsr" not in train_w15.columns:
            if train_v5.exists():
                print(f"\nLoading cached train openbsr from {train_v5.name}")
                train_w15 = w15_filter(pd.read_parquet(train_v5), RESULTS / "oracle_W_train.csv")
            else:
                print("\nAugmenting train W=15 with new openbsr...")
                aug = pd.concat(
                    [p for p in __import__("multiprocessing").Pool(args.workers).map(
                        _augment_one_case, sorted(train_w15["case_id"].unique())) if p is not None],
                    ignore_index=True)
                train_w15 = train_w15.merge(aug, on=["case_id", "time_sec"], how="left")
                full = pd.read_parquet(RESULTS / "features_train_n500_v3.parquet")
                full = full.merge(aug, on=["case_id", "time_sec"], how="left")
                full.to_parquet(train_v5, index=False, compression="zstd")
                print(f"  cached {train_v5.name} ({full.shape})")
        if "openbsr" not in val_w15.columns:
            if val_v5.exists():
                print(f"Loading cached val openbsr from {val_v5.name}")
                val_w15 = w15_filter(pd.read_parquet(val_v5), RESULTS / "oracle_W_val.csv")
            else:
                print("Augmenting val W=15 with new openbsr...")
                val_w15 = augment_openbsr(val_w15, n_workers=args.workers)
                full_v = pd.read_parquet(RESULTS / "features_val_n100_v3.parquet")
                full_v = full_v.merge(val_w15[["case_id", "time_sec", "openbsr"]],
                                      on=["case_id", "time_sec"], how="left")
                full_v.to_parquet(val_v5, index=False, compression="zstd")
                print(f"  cached {val_v5.name} ({full_v.shape})")

    variants_to_run = []
    if args.gate in ("oracle", "all"):
        variants_to_run.append(("oracle", "bis_sr_oracle", "bis_sef_oracle", "bis_emg_oracle", "beta_ratio"))
    if args.gate in ("mixed", "all"):
        # raw EEG features + Vista EMG (BIS sensor's high-frequency channel
        # we cannot replicate from 128 Hz EEG).
        variants_to_run.append(("mixed",  "openbsr",       "sef95",          "bis_emg_oracle", "beta_ratio"))
    if args.gate in ("raw", "all"):
        variants_to_run.append(("raw",    "openbsr",       "sef95",          "emg_proxy",      "beta_ratio"))

    all_coef_rows = list(LEE_TABLE2)
    all_perbin_rows = []
    predictions = {}
    overall_metrics = {}

    for name, bsr_c, sef_c, emg_c, rbr_c in variants_to_run:
        res = run_variant(name, bsr_c, sef_c, emg_c, rbr_c, train_w15, val_w15)
        all_coef_rows.extend(res["coef_rows"])
        all_perbin_rows.extend(res["perbin_rows"])
        predictions[name] = res["pred_v"]
        overall_metrics[name] = dict(mae=res["overall_mae"], r=res["overall_r"], rc=res["overall_rc"])

    # Write outputs
    pd.DataFrame(all_coef_rows).to_csv(RESULTS / "lee2019_replication_coefs.csv", index=False)
    pd.DataFrame(all_perbin_rows).to_csv(RESULTS / "lee2019_replication_perbin.csv", index=False)
    print(f"\nSaved {RESULTS/'lee2019_replication_coefs.csv'}")
    print(f"Saved {RESULTS/'lee2019_replication_perbin.csv'}")

    # Scatter plot
    if predictions:
        import matplotlib.pyplot as plt
        y_v = val_w15["target"].values
        nplots = len(predictions)
        fig, axes = plt.subplots(1, nplots, figsize=(7 * nplots, 7), sharey=True)
        if nplots == 1:
            axes = [axes]
        for ax, (name, pred) in zip(axes, predictions.items()):
            m = ~np.isnan(pred) & ~np.isnan(y_v)
            ax.scatter(y_v[m], pred[m], s=0.4, color="black", alpha=0.10)
            ax.plot([0, 100], [0, 100], color="tab:red", lw=1.0)
            om = overall_metrics[name]
            ax.set_title(f"{name}  MAE={om['mae']:.2f}  r={om['r']:.3f}  rc={om['rc']:.3f}")
            ax.set_xlabel("actual BIS"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
            ax.set_ylabel("predicted BIS")
            ax.grid(alpha=0.3)
        plt.tight_layout()
        out = RESULTS / "lee2019_replication_scatter.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"Saved {out}")

    # Final coefficient comparison printout
    print("\n=== Coefficient comparison (Lee 2019 vs our variants) ===")
    df_print = pd.DataFrame(all_coef_rows)
    print(df_print[["variant", "region", "intercept", "coef_BSR", "coef_EMG", "coef_SEF", "coef_RBR"]].to_string(
        index=False, float_format=lambda x: f"{x:+7.3f}"))

    print("\n=== Per-bin val MAE ===")
    pb = pd.DataFrame(all_perbin_rows)
    if not pb.empty:
        print(pb.pivot(index="bin", columns="variant", values="mae").reindex(LEE_BIN_LABELS).to_string(
            float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    # Set thread caps before any heavy imports inside workers
    os.environ.setdefault("OMP_NUM_THREADS", "2")  # per-worker; with 32 workers ≈ 64 total
    main()
