"""Phase C — discover Lee-2019-style decision tree topology from data.

Two shallow (depth 4) sklearn DecisionTreeRegressors trained on the
W=15 train fold:
  Tree-RAW     : 23 raw-EEG-derived features only (deployable standalone).
  Tree-MIXED   : 23 raw + bis_emg_oracle (deployable on a BIS sensor —
                 BIS Vista provides 70–110 Hz EMG that we cannot
                 reproduce from 128 Hz EEG, so this is the realistic
                 production input set).

For each tree:
  * `sklearn.tree.export_text` dump to results/data_driven_tree.txt
  * top-K feature importances
  * sklearn split thresholds compared head-to-head with Lee 2019
    canonical cuts: BSR > 49.8 / EMG < 34.2 / SEF < 20.2 / RBR < -0.7
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).resolve().parents[1] / "results"

RAW_FEATURES = [
    # 15 base
    "openibis_paper", "openibis_quazi", "openibis_quazi_30s",
    "bsr_paper", "bsr_quazi", "sef95", "bcsef", "beta_ratio", "emg_proxy",
    "p_delta", "p_theta", "p_alpha", "p_beta", "p_lowgamma", "spectral_entropy",
    # 7 phase-3f extras
    "openibis_quazi_5s", "openibis_quazi_10s", "openibis_quazi_60s",
    "openibis_quazi_dt", "openibis_quazi_30s_dt", "sef95_dt", "emg_proxy_dt",
    # new openbsr (Connor 2024 Table 1)
    "openbsr",
]

MIXED_EXTRA = ["bis_emg_oracle"]


def w15_filter(df, oracle_w_csv):
    w = pd.read_csv(oracle_w_csv)
    keep = set(w.loc[w["oracle_W"] == 15, "case_id"].astype(int))
    return df[df["case_id"].isin(keep)].reset_index(drop=True)


def fit_and_describe(df, feature_cols, target_col, label, *, max_depth=4,
                     min_samples_leaf=10_000):
    Xa = df[feature_cols].values
    y = df[target_col].values
    keep = ~np.isnan(Xa).any(axis=1) & ~np.isnan(y)
    print(f"\n=== {label} — {keep.sum():,} rows, {len(feature_cols)} features ===")
    if keep.sum() < min_samples_leaf:
        print("  too few rows; skipping")
        return None
    tree = DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        random_state=0,
    )
    tree.fit(Xa[keep], y[keep])
    pred = tree.predict(Xa[keep])
    mae = float(np.mean(np.abs(pred - y[keep])))
    print(f"  in-bag MAE = {mae:.2f}")
    text = export_text(tree, feature_names=feature_cols, decimals=2)
    return tree, text, feature_cols


def describe_splits(tree, feature_cols):
    """Walk the tree and list every threshold split with its feature."""
    rows = []
    t = tree.tree_
    for node in range(t.node_count):
        if t.feature[node] < 0:
            continue
        rows.append(dict(
            node=node,
            feature=feature_cols[t.feature[node]],
            threshold=float(t.threshold[node]),
            n_samples=int(t.n_node_samples[node]),
            left_value=float(t.value[t.children_left[node], 0, 0]),
            right_value=float(t.value[t.children_right[node], 0, 0]),
        ))
    return pd.DataFrame(rows)


def compare_to_lee(splits_df, label):
    print(f"\n  splits found in {label}:")
    print(splits_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    LEE = [
        ("BSR-like > 49.8",  "bsr",  49.8),
        ("EMG-like < 34.2",  "emg",  34.2),
        ("SEF-like < 20.2",  "sef",  20.2),
        ("RBR < -0.7",        "beta_ratio", -0.7),
    ]
    print(f"\n  Lee 2019 reference cuts and the closest discovered threshold:")
    for lee_label, family, lee_val in LEE:
        candidate = splits_df[
            splits_df["feature"].str.contains(family, regex=False) |
            (splits_df["feature"] == "openbsr") & (family == "bsr") |
            (splits_df["feature"] == "bis_emg_oracle") & (family == "emg") |
            (splits_df["feature"] == "bis_sr_oracle") & (family == "bsr")
        ]
        if candidate.empty:
            print(f"    {lee_label:<20s}: no match")
            continue
        candidate = candidate.assign(diff=(candidate["threshold"] - lee_val).abs())
        best = candidate.loc[candidate["diff"].idxmin()]
        print(f"    {lee_label:<20s} → "
              f"{best['feature']:<22s} <= {best['threshold']:.2f}  "
              f"(Lee {lee_val:+.2f}, diff {best['diff']:.2f})")


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "64")

    print("Loading W=15 train cohort (v5 with openbsr)...")
    v5 = RESULTS / "features_train_n500_v5.parquet"
    if not v5.exists():
        raise SystemExit(f"Missing {v5} — run scripts/27 first to generate it.")
    train = pd.read_parquet(v5)
    train_w15 = w15_filter(train, RESULTS / "oracle_W_train.csv")
    print(f"  {len(train_w15):,} rows / {train_w15['case_id'].nunique()} cases")
    print(f"  Available columns include: {sorted(set(RAW_FEATURES + MIXED_EXTRA) & set(train_w15.columns))}")

    text_out = []
    importance_out = []

    # ---- Tree-RAW
    raw_cols = [c for c in RAW_FEATURES if c in train_w15.columns]
    r = fit_and_describe(train_w15, raw_cols, "target", "Tree-RAW (23 raw features)")
    if r is not None:
        tree, text, feats = r
        text_out.append("=== Tree-RAW ===\n" + text)
        splits = describe_splits(tree, feats)
        splits.to_csv(RESULTS / "data_driven_tree_raw_splits.csv", index=False)
        compare_to_lee(splits, "Tree-RAW")
        imp = pd.DataFrame({"feature": feats, "importance": tree.feature_importances_})
        imp = imp.sort_values("importance", ascending=False).head(10)
        print(f"\n  Tree-RAW top 10 importances:\n{imp.to_string(index=False, float_format=lambda x: f'{x:.3f}')}")
        importance_out.append(("raw", imp))

    # ---- Tree-MIXED
    mixed_cols = raw_cols + [c for c in MIXED_EXTRA if c in train_w15.columns]
    r2 = fit_and_describe(train_w15, mixed_cols, "target", "Tree-MIXED (23 raw + Vista EMG)")
    if r2 is not None:
        tree2, text2, feats2 = r2
        text_out.append("\n=== Tree-MIXED ===\n" + text2)
        splits2 = describe_splits(tree2, feats2)
        splits2.to_csv(RESULTS / "data_driven_tree_mixed_splits.csv", index=False)
        compare_to_lee(splits2, "Tree-MIXED")
        imp2 = pd.DataFrame({"feature": feats2, "importance": tree2.feature_importances_})
        imp2 = imp2.sort_values("importance", ascending=False).head(10)
        print(f"\n  Tree-MIXED top 10 importances:\n{imp2.to_string(index=False, float_format=lambda x: f'{x:.3f}')}")
        importance_out.append(("mixed", imp2))

    # ---- Tree-RAW on desmoothed target
    # Reuse oracle_W cache for per-case EMA-15s inversion.
    w_cache = pd.read_csv(RESULTS / "oracle_W_train.csv").set_index("case_id")["oracle_W"].to_dict()

    def desmooth_ema(y, W):
        a = 2.0 / (W + 1.0)
        x = np.empty_like(y)
        x[0] = y[0]
        x[1:] = (y[1:] - (1 - a) * y[:-1]) / a
        return np.clip(x, 0, 100)

    train_w15 = train_w15.sort_values(["case_id", "time_sec"]).reset_index(drop=True)
    train_w15["target_desm"] = np.nan
    for cid, sub in train_w15.groupby("case_id"):
        idx = sub.index.to_numpy()
        a = sub["target"].values
        if np.isnan(a).all():
            continue
        a_filled = np.where(np.isnan(a), float(np.nanmean(a)) or 50.0, a)
        W = int(w_cache.get(int(cid), 15))
        train_w15.loc[idx, "target_desm"] = desmooth_ema(a_filled, W)

    r3 = fit_and_describe(train_w15, raw_cols, "target_desm",
                          "Tree-RAW-DESMOOTHED (raw features, EMA-inverse target)")
    if r3 is not None:
        tree3, text3, feats3 = r3
        text_out.append("\n=== Tree-RAW-DESMOOTHED ===\n" + text3)
        splits3 = describe_splits(tree3, feats3)
        splits3.to_csv(RESULTS / "data_driven_tree_desm_splits.csv", index=False)
        compare_to_lee(splits3, "Tree-RAW-DESMOOTHED")

    out_txt = RESULTS / "data_driven_tree.txt"
    out_txt.write_text("\n".join(text_out))
    print(f"\nSaved {out_txt}")


if __name__ == "__main__":
    main()
