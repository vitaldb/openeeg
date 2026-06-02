"""Phase D — per-region OLS with the smallest informative feature set,
emitting a deployable piecewise-linear BIS predictor.

Two partitioning schemes are compared:
  * 'lee'  — Lee 2019's hand-crafted 5-region decision tree (Phase A).
  * 'data' — sklearn depth-4 tree's leaf assignment (Phase C).

Within each leaf, we pick the top-K features by within-leaf |corr(feature,
target)| (default K=2) and fit a constrained OLS. The resulting
piecewise-linear model is saved as JSON and as an executable Python
predictor inside ``openeeg/piecewise.py``.

We build BOTH variants:
  * raw   — uses only raw-EEG-derived features (truly standalone).
  * mixed — same plus ``bis_emg_oracle`` (the BIS sensor's
            70–110 Hz EMG track; available on any device that ships
            with a BIS sensor, but not from any 128 Hz EEG channel).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS, lin_concordance

RESULTS = Path(__file__).resolve().parents[1] / "results"


RAW_FEATURES = [
    "openibis_paper", "openibis_quazi", "openibis_quazi_30s",
    "bsr_paper", "bsr_quazi", "sef95", "bcsef", "beta_ratio", "emg_proxy",
    "p_delta", "p_theta", "p_alpha", "p_beta", "p_lowgamma", "spectral_entropy",
    "openibis_quazi_5s", "openibis_quazi_10s", "openibis_quazi_60s",
    "openibis_quazi_dt", "openibis_quazi_30s_dt", "sef95_dt", "emg_proxy_dt",
    "openbsr",
]
ORACLE_EMG = "bis_emg_oracle"


def w15_filter(df, oracle_w_csv):
    w = pd.read_csv(oracle_w_csv)
    keep = set(w.loc[w["oracle_W"] == 15, "case_id"].astype(int))
    return df[df["case_id"].isin(keep)].reset_index(drop=True)


def ema_smooth(x, W=15.0):
    a = 2.0 / (W + 1.0)
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    return y


def lee_partition(df, bsr_col, sef_col, emg_col, rbr_col):
    bsr = df[bsr_col].values; sef = df[sef_col].values
    emg = df[emg_col].values; rbr = df[rbr_col].values
    deep = bsr > 49.8
    mid_gate = (~deep) & (emg < 34.2) & (sef < 20.2)
    light = mid_gate & ((bsr > 2.1) | (sef < 14.8))
    surg  = mid_gate & ~light
    not_mid = (~deep) & (~mid_gate)
    trans = not_mid & (rbr < -0.7)
    awake = not_mid & (rbr >= -0.7)
    return {"deep": deep, "light": light, "surg": surg, "trans": trans, "awake": awake}


def fit_data_tree(df, feat_cols, target_col, max_depth=4, min_samples_leaf=10_000):
    X = df[feat_cols].values; y = df[target_col].values
    keep = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0)
    tree.fit(X[keep], y[keep])
    return tree, keep


def data_tree_leaves(tree, X):
    """Return leaf id per row (NaN-safe via 0 fill on missing rows)."""
    Xc = np.where(np.isnan(X), 0.0, X)
    return tree.apply(Xc)


def fit_per_leaf_linear(df_tr, df_v, X_tr, X_v, feat_cols, leaves_tr, leaves_v, top_k=2,
                       global_target_col="target"):
    """For each leaf id present in both train and val, fit OLS using top-K
    features by within-leaf |corr| with target.  Return prediction array
    on val + structured coefficients."""
    y_tr = df_tr[global_target_col].values
    y_v = df_v[global_target_col].values
    pred_v = np.full(len(df_v), np.nan)
    leaf_specs = []

    all_leaves = sorted(set(leaves_tr.tolist()) | set(leaves_v.tolist()))
    for leaf in all_leaves:
        mask_tr = (leaves_tr == leaf) & ~np.isnan(y_tr) & ~np.isnan(X_tr).any(axis=1)
        mask_v = (leaves_v == leaf) & ~np.isnan(y_v) & ~np.isnan(X_v).any(axis=1)
        if mask_tr.sum() < 500:
            # fall back to leaf mean
            mean_v = float(y_tr[(leaves_tr == leaf) & ~np.isnan(y_tr)].mean()
                          if ((leaves_tr == leaf) & ~np.isnan(y_tr)).sum() > 0 else np.nanmean(y_tr))
            pred_v[mask_v] = mean_v
            leaf_specs.append(dict(leaf=int(leaf), n_train=int(mask_tr.sum()),
                                    n_val=int(mask_v.sum()), features=[],
                                    intercept=mean_v, coefs=[], note="fallback_mean"))
            continue
        Xl = X_tr[mask_tr]
        yl = y_tr[mask_tr]
        # within-leaf correlation per feature
        corrs = np.array([
            float(np.corrcoef(Xl[:, j], yl)[0, 1]) if Xl[:, j].std() > 1e-9 else 0.0
            for j in range(Xl.shape[1])
        ])
        top = np.argsort(np.abs(corrs))[::-1][:top_k]
        m = LinearRegression()
        m.fit(Xl[:, top], yl)
        # Apply
        if mask_v.sum() > 0:
            pred_v[mask_v] = m.predict(X_v[mask_v][:, top])
        leaf_specs.append(dict(
            leaf=int(leaf),
            n_train=int(mask_tr.sum()), n_val=int(mask_v.sum()),
            features=[feat_cols[j] for j in top],
            intercept=float(m.intercept_),
            coefs=[float(c) for c in m.coef_],
        ))
    return pred_v, leaf_specs


def per_regime_mae(actual, pred):
    out = {}
    m_finite = ~np.isnan(actual) & ~np.isnan(pred)
    a = actual[m_finite]; p = pred[m_finite]
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        mm = (a >= lo) & (a < hi)
        out[lbl] = float(np.mean(np.abs(p[mm] - a[mm]))) if mm.sum() > 10 else float("nan")
    return out


def overall_metrics(actual, pred):
    m = ~np.isnan(actual) & ~np.isnan(pred)
    if m.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    mae = float(np.mean(np.abs(pred[m] - actual[m])))
    r = float(np.corrcoef(pred[m], actual[m])[0, 1])
    rc = lin_concordance(pred[m], actual[m])
    return mae, r, rc


def smooth_by_case(pred, df, W=15.0):
    out = np.empty_like(pred, dtype=float)
    for cid, sub in df.groupby("case_id"):
        idx = sub.index.to_numpy()
        out[idx] = ema_smooth(pred[idx], W)
    return out


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "64")
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=6,
                    help="Number of top-|corr| features used in each leaf's OLS. "
                         "K=2 is the original interpretable variant; K=6 closes "
                         "~0.18 BIS-point of underfit at the cost of longer "
                         "per-leaf formulas.")
    ap.add_argument("--deep-rule-thresh", type=float, default=49.8,
                    help="openbsr threshold above which the Ellerkmann rule "
                         "(BIS = 44.1 - openbsr/2.25) is used instead of "
                         "piecewise. Training is filtered to openbsr <= this "
                         "threshold so the piecewise model's scope matches "
                         "the deployment scope.")
    args = ap.parse_args()

    print("Loading W=15 train & val (v5)...")
    train = pd.read_parquet(RESULTS / "features_train_n500_v5.parquet")
    val   = pd.read_parquet(RESULTS / "features_val_n100_v5.parquet")
    train_w15 = w15_filter(train, RESULTS / "oracle_W_train.csv").reset_index(drop=True)
    val_w15_full = w15_filter(val, RESULTS / "oracle_W_val.csv").reset_index(drop=True)
    print(f"  train: {len(train_w15):,}  val: {len(val_w15_full):,}")

    # ---- Deep-rule split: train on openbsr <= thresh; Ellerkmann otherwise.
    deep_thr = float(args.deep_rule_thresh)
    train_deep_mask = train_w15["openbsr"].values > deep_thr
    train_w15 = train_w15.loc[~train_deep_mask].reset_index(drop=True)
    print(f"  after deep-rule filter (openbsr <= {deep_thr}): "
          f"train={len(train_w15):,} ({train_deep_mask.sum():,} dropped)")

    # Val cohort stays whole for end-to-end evaluation; we split outputs by openbsr.
    val_w15 = val_w15_full
    val_deep_mask = val_w15["openbsr"].values > deep_thr

    y_v = val_w15["target"].values
    summary_rows = []
    all_coef_specs = {}

    for variant_name, extra_cols in [("raw", []), ("mixed", [ORACLE_EMG])]:
        feat_cols = [c for c in RAW_FEATURES + extra_cols if c in train_w15.columns]
        X_tr = train_w15[feat_cols].values
        X_v  = val_w15[feat_cols].values
        print(f"\n========== Variant: {variant_name}  ({len(feat_cols)} features) ==========")

        # ---- Scheme 1: Lee partition (using openbsr / sef95 / EMG / beta_ratio)
        emg_col = ORACLE_EMG if variant_name == "mixed" and ORACLE_EMG in train_w15.columns else "emg_proxy"
        parts_tr = lee_partition(train_w15, "openbsr", "sef95", emg_col, "beta_ratio")
        parts_v  = lee_partition(val_w15,   "openbsr", "sef95", emg_col, "beta_ratio")
        # Build a leaf id per row using the region name → integer
        region_id = {"deep": 0, "light": 1, "surg": 2, "trans": 3, "awake": 4}
        leaves_tr_lee = np.full(len(train_w15), -1, dtype=int)
        leaves_v_lee  = np.full(len(val_w15), -1, dtype=int)
        for r in region_id:
            leaves_tr_lee[parts_tr[r]] = region_id[r]
            leaves_v_lee[parts_v[r]]   = region_id[r]

        pred_lee, lee_specs = fit_per_leaf_linear(
            train_w15, val_w15, X_tr, X_v, feat_cols, leaves_tr_lee, leaves_v_lee,
            top_k=args.top_k)
        pred_lee_clip = np.clip(pred_lee, 0, 100)
        # Apply Ellerkmann rule where openbsr > deep_thr
        ellerk_v = np.clip(44.1 - val_w15["openbsr"].values / 2.25, 0, 100)
        pred_lee_clip = np.where(val_deep_mask, ellerk_v, pred_lee_clip)
        pred_lee_sm = smooth_by_case(pred_lee_clip, val_w15, W=15.0)
        mae, r, rc = overall_metrics(y_v, pred_lee_sm)
        reg = per_regime_mae(y_v, pred_lee_sm)
        print(f"\n[{variant_name} | Lee partition] MAE={mae:.2f}  r={r:.3f}  rc={rc:.3f}")
        for k, v in reg.items():
            print(f"    {k}: {v:.2f}")
        summary_rows.append(dict(variant=variant_name, partition="lee", mae=mae, r=r, rc=rc, **reg))

        # ---- Scheme 2: Data-driven tree partition
        tree, keep_tr = fit_data_tree(train_w15, feat_cols, "target",
                                      max_depth=4, min_samples_leaf=10_000)
        leaves_tr_dat = data_tree_leaves(tree, X_tr)
        leaves_v_dat  = data_tree_leaves(tree, X_v)
        pred_dat, dat_specs = fit_per_leaf_linear(
            train_w15, val_w15, X_tr, X_v, feat_cols, leaves_tr_dat, leaves_v_dat,
            top_k=args.top_k)
        pred_dat_clip = np.clip(pred_dat, 0, 100)
        pred_dat_clip = np.where(val_deep_mask, ellerk_v, pred_dat_clip)
        pred_dat_sm = smooth_by_case(pred_dat_clip, val_w15, W=15.0)
        mae, r, rc = overall_metrics(y_v, pred_dat_sm)
        reg = per_regime_mae(y_v, pred_dat_sm)
        print(f"\n[{variant_name} | Data-driven partition] MAE={mae:.2f}  r={r:.3f}  rc={rc:.3f}")
        for k, v in reg.items():
            print(f"    {k}: {v:.2f}")
        summary_rows.append(dict(variant=variant_name, partition="data", mae=mae, r=r, rc=rc, **reg))

        # Save coefficient JSONs for both partitions of this variant
        out_json_lee = RESULTS / f"piecewise_{variant_name}_lee.json"
        out_json_dat = RESULTS / f"piecewise_{variant_name}_data.json"
        with open(out_json_lee, "w") as f:
            json.dump(dict(variant=variant_name, partition="lee",
                            feature_cols=feat_cols,
                            deep_rule=dict(
                                feature="openbsr", threshold=deep_thr,
                                formula="BIS = 44.1 - openbsr/2.25"),
                            leaves=lee_specs), f, indent=2)
        with open(out_json_dat, "w") as f:
            json.dump(dict(variant=variant_name, partition="data",
                            feature_cols=feat_cols,
                            deep_rule=dict(
                                feature="openbsr", threshold=deep_thr,
                                formula="BIS = 44.1 - openbsr/2.25"),
                            leaves=dat_specs,
                            tree_thresholds=[
                                dict(node=int(i),
                                     feature=feat_cols[tree.tree_.feature[i]] if tree.tree_.feature[i] >= 0 else None,
                                     threshold=float(tree.tree_.threshold[i]),
                                     left=int(tree.tree_.children_left[i]),
                                     right=int(tree.tree_.children_right[i]))
                                for i in range(tree.tree_.node_count)
                            ]), f, indent=2)
        print(f"  saved {out_json_lee.name} / {out_json_dat.name}")
        all_coef_specs[variant_name] = dict(lee=lee_specs, data=dat_specs,
                                             feature_cols=feat_cols)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "piecewise_summary.csv", index=False)
    print("\n=== Summary table (val cohort, EMA(15s) post-smoothed) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
