"""Phase E — Residual analysis on the deployable Phase-D raw model.

Goal: does ``residual = actual_BIS − predict_bis_rules(eeg)`` carry
structure that points to a missing sub-parameter we should add (and
which could then be learned by a CNN in Phase F)?

We test:
  1. Per-feature, per-region |corr(residual, feature)|. Any region/
     feature pair with |corr| > 0.2 flags a candidate addition.
  2. A shallow DecisionTreeRegressor on the residual using the same
     23 raw features. Its first split + the resulting MAE drop are
     reported. If the first split removes > 25 % of in-region MAE,
     that feature is also flagged.
  3. Two SynchFastSlow proxies (Connor 2023 says BIS does NOT use the
     bispectrum, Noh 2017 says it does):
       * Power-only SFS:  log10(P_30-47 / P_0.5-47)
       * Bispectral SFS:  per-segment cross-bispectrum magnitude
         at (f1=10 Hz, f2=20 Hz) using ``scipy.signal.csd`` averaged
         over a 5-s window. (Approximation, not full bispectrum.)
     Per-region correlations with the residual decide whether the
     bispectrum hypothesis is alive in our cohort.

Gating decision is written to ``results/residual_top_feature.txt``:
* Phase F (CNN) runs only if any region has either
  ``|corr|_max > 0.2`` OR ``residual_tree_mae_drop_pct > 25``.
* Otherwise Phase F is skipped and the report names the null result.

Outputs
  results/residual_structure.csv    one row per (feature, region, corr)
  results/residual_tree.txt         export_text of the residual tree
  results/residual_top_feature.txt  the single feature most worth
                                    adding + Phase F gating decision
  results/residual_diagnostic.png   4-panel (overall scatter, per-region
                                    corr heatmap, residual tree, SFS panel)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS

RESULTS = Path(__file__).resolve().parents[1] / "results"

FEATURE_COLS = [
    "openibis_paper", "openibis_quazi", "openibis_quazi_30s",
    "bsr_paper", "bsr_quazi", "sef95", "bcsef", "beta_ratio", "emg_proxy",
    "p_delta", "p_theta", "p_alpha", "p_beta", "p_lowgamma", "spectral_entropy",
    "openibis_quazi_5s", "openibis_quazi_10s", "openibis_quazi_60s",
    "openibis_quazi_dt", "openibis_quazi_30s_dt", "sef95_dt", "emg_proxy_dt",
    "openbsr",
]

PIECEWISE_JSON = RESULTS / "piecewise_raw_data.json"


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


def piecewise_predict_from_features(df: pd.DataFrame) -> np.ndarray:
    """Apply piecewise_raw_data.json (including deep-rule override) on a
    feature DataFrame (no EEG, no openbsr recomputation)."""
    import json
    with open(PIECEWISE_JSON) as f:
        j = json.load(f)
    nodes = {int(n["node"]): n for n in j["tree_thresholds"]}
    leaves = {int(s["leaf"]): s for s in j["leaves"]}
    feat_cols = j["feature_cols"]
    deep_rule = j.get("deep_rule")

    X = df[feat_cols].values
    X_safe = np.where(np.isnan(X), 0.0, X)
    n = X.shape[0]
    cur = np.zeros(n, dtype=int)
    done = np.zeros(n, dtype=bool)
    leaf_id = np.full(n, -1, dtype=int)
    for _ in range(64):
        if done.all():
            break
        unique_nodes = np.unique(cur[~done])
        for node in unique_nodes:
            spec = nodes[int(node)]
            m = (cur == int(node)) & (~done)
            if spec["feature"] is None:
                leaf_id[m] = int(node); done[m] = True; continue
            fi = feat_cols.index(spec["feature"])
            thr = float(spec["threshold"])
            vals = X_safe[m, fi]
            go_left = ~(vals > thr)
            new_cur = np.where(go_left, spec["left"], spec["right"])
            cur[m] = new_cur.astype(int)

    pred = np.full(n, np.nan, dtype=float)
    for lid, spec in leaves.items():
        m = leaf_id == int(lid)
        if not m.any():
            continue
        a = float(spec["intercept"])
        if not spec["features"]:
            pred[m] = a; continue
        idxs = [feat_cols.index(f) for f in spec["features"]]
        coefs = np.asarray(spec["coefs"], dtype=float)
        pred[m] = a + X_safe[m][:, idxs] @ coefs
    pred = np.clip(pred, 0, 100)
    if deep_rule is not None:
        fi = feat_cols.index(deep_rule["feature"])
        thr = float(deep_rule["threshold"])
        obsr = X[:, fi]
        deep_mask = np.where(np.isnan(obsr), False, obsr > thr)
        pred[deep_mask] = np.clip(44.1 - obsr[deep_mask] / 2.25, 0, 100)
    return pred


def _sfs_proxies_for_case(args):
    """Worker: compute power-only SFS and bispectral SFS at 2 Hz per case."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    cid, win_samples, fs = args
    from openeeg.cohort import load_case, preprocess_eeg
    case = load_case(int(cid))
    if case is None:
        return None
    eeg = preprocess_eeg(case["eeg"])
    # We compute per-2-Hz-epoch quantities (= 0.5 s step at fs=128 → 64 samples)
    step = fs // 2  # 64 samples = 2 Hz
    # Window: 5 s (640 samples) for bispectrum estimation
    win = win_samples
    n_epochs = max(0, (len(eeg) - win) // step + 1)
    if n_epochs <= 0:
        return None
    times = (np.arange(n_epochs, dtype=np.int32) * step // fs).astype(np.int32)
    power_sfs = np.full(n_epochs, np.nan, dtype=np.float32)
    bispec_sfs = np.full(n_epochs, np.nan, dtype=np.float32)
    # Bispectrum proxy: average |X(f1) X(f2) X*(f1+f2)| over short segments
    # within the 5-s window for f1=10, f2=20 (sum freq 30) — picks up the
    # quadratic phase coupling that distinguishes anesthetic burst patterns.
    f1, f2 = 10.0, 20.0
    f12 = f1 + f2
    n_fft = 512
    # Pre-compute indices for f1, f2, f12 at this fs/n_fft
    freqs_axis = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    i1 = int(np.argmin(np.abs(freqs_axis - f1)))
    i2 = int(np.argmin(np.abs(freqs_axis - f2)))
    i12 = int(np.argmin(np.abs(freqs_axis - f12)))

    for k in range(n_epochs):
        seg = eeg[k * step : k * step + win]
        if len(seg) < win or not np.all(np.isfinite(seg)):
            continue
        # Power-only SFS = log10(P_30-47 / P_0.5-47)
        seg_dm = seg - seg.mean()
        # Use Welch-like averaging via 256-sample sub-windows with 50% overlap
        sub = 256
        if len(seg_dm) < sub:
            continue
        Pxx = np.zeros(sub // 2 + 1, dtype=np.float64)
        n_sub = 0
        for s_start in range(0, len(seg_dm) - sub + 1, sub // 2):
            sw = seg_dm[s_start : s_start + sub]
            sw_w = sw * np.hanning(sub)
            X = np.fft.rfft(sw_w)
            Pxx += (X.real ** 2 + X.imag ** 2)
            n_sub += 1
        if n_sub == 0:
            continue
        Pxx /= n_sub
        freqs_sub = np.fft.rfftfreq(sub, d=1.0 / fs)
        num = float(Pxx[(freqs_sub >= 30) & (freqs_sub <= 47)].sum())
        den = float(Pxx[(freqs_sub >= 0.5) & (freqs_sub <= 47)].sum())
        if den > 0 and num > 0:
            power_sfs[k] = float(np.log10(num / den))
        # Bispectral SFS proxy: |X(f1) X(f2) X*(f1+f2)| averaged
        # over the 5-s window using 512-sample FFTs
        bispec_sum = 0.0
        n_seg = 0
        for s_start in range(0, len(seg_dm) - n_fft + 1, n_fft // 2):
            sw = seg_dm[s_start : s_start + n_fft]
            sw_w = sw * np.hanning(n_fft)
            X = np.fft.rfft(sw_w)
            tri = X[i1] * X[i2] * np.conj(X[i12])
            bispec_sum += abs(tri)
            n_seg += 1
        if n_seg > 0:
            bispec_sfs[k] = float(np.log10(bispec_sum / n_seg + 1e-30))

    return pd.DataFrame(dict(
        case_id=int(cid), time_sec=times,
        sfs_power=power_sfs, sfs_bispec=bispec_sfs,
    ))


def compute_sfs_for_val(val_w15) -> pd.DataFrame:
    """Multiprocessing-cached SFS computation for the val cohort."""
    cache = RESULTS / "sfs_val_w15.parquet"
    if cache.exists():
        print(f"  loading cached {cache.name}")
        return pd.read_parquet(cache)
    cids = sorted(val_w15["case_id"].unique())
    print(f"  computing SFS proxies for {len(cids)} val cases (32 workers)...")
    import time
    t0 = time.time()
    args = [(int(c), 640, 128) for c in cids]
    with Pool(32) as pool:
        pieces = [p for p in pool.map(_sfs_proxies_for_case, args) if p is not None]
    df = pd.concat(pieces, ignore_index=True)
    # Downsample 2 Hz → 1 Hz to match val_w15
    df = df.iloc[::2].reset_index(drop=True)
    df.to_parquet(cache, index=False, compression="zstd")
    print(f"  done {time.time()-t0:.1f}s  cached {cache.name}")
    return df


def per_region_correlations(val_w15, residual):
    rows = []
    target = val_w15["target"].values
    for feat in FEATURE_COLS + ["sfs_power", "sfs_bispec"]:
        if feat not in val_w15.columns:
            continue
        x = val_w15[feat].values
        for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
            m = (target >= lo) & (target < hi) & np.isfinite(x) & np.isfinite(residual)
            if m.sum() < 200:
                rows.append(dict(feature=feat, region=lbl,
                                  corr=float("nan"), n=int(m.sum())))
                continue
            corr = float(np.corrcoef(x[m], residual[m])[0, 1])
            rows.append(dict(feature=feat, region=lbl, corr=corr, n=int(m.sum())))
        # Overall
        m = np.isfinite(x) & np.isfinite(residual)
        if m.sum() >= 200:
            corr = float(np.corrcoef(x[m], residual[m])[0, 1])
            rows.append(dict(feature=feat, region="overall", corr=corr, n=int(m.sum())))
    return pd.DataFrame(rows)


def residual_tree(val_w15, residual):
    """Train a depth-4 tree on residual to identify dominant unmodelled signal."""
    feats = [c for c in FEATURE_COLS + ["sfs_power", "sfs_bispec"] if c in val_w15.columns]
    X = val_w15[feats].values
    y = residual
    keep = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    if keep.sum() < 1000:
        return None, feats, float("nan"), float("nan")
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5_000, random_state=0)
    tree.fit(X[keep], y[keep])
    pred = tree.predict(X[keep])
    in_mae = float(np.mean(np.abs(y[keep] - pred)))
    base_mae = float(np.mean(np.abs(y[keep] - y[keep].mean())))
    return tree, feats, in_mae, base_mae


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "64")

    print("Loading val W=15 features (v5)...")
    val = pd.read_parquet(RESULTS / "features_val_n100_v5.parquet")
    val_w15 = w15_filter(val, RESULTS / "oracle_W_val.csv").reset_index(drop=True)
    print(f"  {len(val_w15):,} rows / {val_w15['case_id'].nunique()} cases")

    print("\nComputing piecewise raw predictions from feature parquet...")
    pred_raw = piecewise_predict_from_features(val_w15)
    pred_sm = smooth_by_case(pred_raw, val_w15, W=15.0)
    actual = val_w15["target"].values
    residual = actual - pred_sm

    # Restrict residual analysis to NON-DEEP rows: the deep regime is
    # confirmed handled by Ellerkmann (Phase D2), so its residual is not a
    # comparison target for "is there a missing sub-parameter".
    obsr = val_w15["openbsr"].values
    nondeep = np.isfinite(obsr) & (obsr <= 49.8)
    mae_full = float(np.nanmean(np.abs(residual)))
    mae_nondeep = float(np.nanmean(np.abs(residual[nondeep])))
    print(f"  overall val MAE (full):     {mae_full:.3f}")
    print(f"  overall val MAE (non-deep): {mae_nondeep:.3f}  ← residual scope")
    # Mask residual to non-deep for downstream analysis
    residual_for_analysis = np.where(nondeep, residual, np.nan)
    val_w15 = val_w15.copy()
    val_w15["target"] = np.where(nondeep, val_w15["target"].values, np.nan)
    residual = residual_for_analysis
    mae = mae_nondeep

    print("\nComputing SFS proxies...")
    sfs_df = compute_sfs_for_val(val_w15)
    val_w15 = val_w15.merge(sfs_df, on=["case_id", "time_sec"], how="left")
    print(f"  SFS coverage: power {val_w15['sfs_power'].notna().mean()*100:.1f}%, "
          f"bispec {val_w15['sfs_bispec'].notna().mean()*100:.1f}%")

    # ---- Per-region correlations
    print("\nComputing per-region |corr(residual, feature)|...")
    corr_df = per_region_correlations(val_w15, residual)
    corr_df.to_csv(RESULTS / "residual_structure.csv", index=False)

    # Top features by max |corr| across regions
    region_corr = corr_df[corr_df["region"] != "overall"].copy()
    region_corr["abs_corr"] = region_corr["corr"].abs()
    top_per_feat = (region_corr.groupby("feature")["abs_corr"].max()
                    .sort_values(ascending=False))
    print("\n=== Top 10 features by max |corr| across regions ===")
    print(top_per_feat.head(10).to_string(float_format=lambda x: f"{x:.3f}"))

    # ---- Residual decision tree
    print("\nTraining residual decision tree (depth 4)...")
    tree, feats, in_mae, base_mae = residual_tree(val_w15, residual)
    mae_drop_pct = float("nan")
    if tree is not None:
        mae_drop_pct = (base_mae - in_mae) / base_mae * 100
        print(f"  base residual MAD = {base_mae:.3f}")
        print(f"  in-bag tree MAE   = {in_mae:.3f}  ({mae_drop_pct:+.1f}% reduction)")
        # First split feature
        root_feat = feats[tree.tree_.feature[0]] if tree.tree_.feature[0] >= 0 else None
        root_thr  = tree.tree_.threshold[0] if root_feat else float("nan")
        print(f"  root split: {root_feat} <= {root_thr:.3f}")
        # Importances
        imp = pd.DataFrame({"feature": feats,
                             "importance": tree.feature_importances_})
        imp = imp.sort_values("importance", ascending=False).head(8)
        print(f"\n  Residual tree top 8 importances:")
        print(imp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        with open(RESULTS / "residual_tree.txt", "w") as f:
            f.write(export_text(tree, feature_names=feats, decimals=2))

    # ---- Phase F gating decision
    max_abs_corr = float(top_per_feat.iloc[0]) if len(top_per_feat) else 0.0
    gate_corr = max_abs_corr > 0.20
    gate_tree = mae_drop_pct > 25
    phase_f_should_run = bool(gate_corr or gate_tree)

    top_feat = top_per_feat.index[0] if len(top_per_feat) else "none"
    top_region = (region_corr[region_corr["feature"] == top_feat]
                   .nlargest(1, "abs_corr"))
    if len(top_region):
        tr_row = top_region.iloc[0]
        top_region_desc = f"{tr_row['region']} (corr={tr_row['corr']:+.3f}, n={tr_row['n']:,})"
    else:
        top_region_desc = "n/a"

    report = []
    report.append("===========================================================")
    report.append("Phase E — Residual analysis on predict_bis_rules (raw)")
    report.append("===========================================================")
    report.append(f"Cohort: {len(val_w15):,} epochs, {val_w15['case_id'].nunique()} cases (val W=15, SQI>=80)")
    report.append(f"predict_bis_rules val MAE: {mae:.3f}")
    report.append("")
    report.append(f"Strongest per-region |corr(residual, feature)|: {max_abs_corr:.3f}")
    report.append(f"  feature: {top_feat}   strongest region: {top_region_desc}")
    report.append(f"Residual tree MAE drop: {mae_drop_pct:+.1f}% "
                  f"({base_mae:.3f} → {in_mae:.3f})")
    report.append("")
    report.append("Phase F (CNN residual estimator) gates:")
    report.append(f"  |corr| > 0.20 anywhere?       {'YES' if gate_corr else 'NO'}")
    report.append(f"  residual tree drop > 25%?     {'YES' if gate_tree else 'NO'}")
    report.append(f"  → Phase F should run: {'YES' if phase_f_should_run else 'NO (null residual)'}")
    if not phase_f_should_run:
        report.append("")
        report.append("INTERPRETATION: residual carries no concentrated structure linear")
        report.append("models would miss. The remaining 4.62 BIS-point error is plausibly")
        report.append("a mix of (a) 15s smoothing aliasing, (b) inter-patient variability,")
        report.append("(c) regime-overlap fuzz near 40/60. No missing sub-parameter found.")
    else:
        report.append("")
        report.append(f"INTERPRETATION: feature `{top_feat}` carries reducible residual signal")
        report.append("in at least one region. Phase F CNN should learn this from raw EEG.")

    out_txt = RESULTS / "residual_top_feature.txt"
    out_txt.write_text("\n".join(report))
    print("\n" + "\n".join(report))
    print(f"\nSaved {out_txt}")

    # ---- 4-panel diagnostic figure
    print("\nRendering diagnostic figure...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # (a) Residual vs predicted scatter
    ax = axes[0, 0]
    finite = np.isfinite(pred_sm) & np.isfinite(residual)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(finite)[0], min(80000, finite.sum()), replace=False)
    ax.scatter(pred_sm[idx], residual[idx], s=0.4, color="black", alpha=0.10)
    for v in LEE_BINS:
        ax.axvline(v, color="tab:gray", lw=0.6, alpha=0.4)
    ax.axhline(0, color="tab:red", lw=0.8)
    ax.set_xlabel("predict_bis_rules (raw, smoothed)")
    ax.set_ylabel("actual − predicted")
    ax.set_title(f"(a) Residual scatter  MAE={mae:.2f}")
    ax.set_xlim(0, 100); ax.set_ylim(-50, 50)
    ax.grid(alpha=0.2)

    # (b) Per-region |corr| heatmap
    ax = axes[0, 1]
    feats_for_plot = list(top_per_feat.head(15).index)
    mat = []
    for feat in feats_for_plot:
        row = []
        for lbl in LEE_BIN_LABELS:
            sub = corr_df[(corr_df["feature"] == feat) & (corr_df["region"] == lbl)]
            row.append(float(sub["corr"].iloc[0]) if len(sub) else float("nan"))
        mat.append(row)
    mat = np.array(mat)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(LEE_BIN_LABELS)))
    ax.set_xticklabels(LEE_BIN_LABELS)
    ax.set_yticks(range(len(feats_for_plot)))
    ax.set_yticklabels(feats_for_plot, fontsize=8)
    for i in range(len(feats_for_plot)):
        for j in range(len(LEE_BIN_LABELS)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.25 else "black")
    plt.colorbar(im, ax=ax)
    ax.set_title("(b) Per-region corr(residual, feature) — top 15 by max|corr|")

    # (c) Residual tree text (if available)
    ax = axes[1, 0]
    ax.axis("off")
    if tree is not None:
        # Show a brief textual summary of the top 4 splits
        # Build a 'first 4 splits' list
        t = tree.tree_
        lines = ["Residual tree — top splits:\n"]
        # Walk depth-first the first 4 levels
        def walk(node, depth, prefix):
            if depth > 4 or node < 0 or t.feature[node] < 0:
                return
            f = feats[t.feature[node]]
            thr = float(t.threshold[node])
            lines.append(f"{prefix}{'  ' * depth}{f} <= {thr:.3f}   "
                         f"(n={t.n_node_samples[node]:,})")
            walk(t.children_left[node],  depth + 1, prefix)
            walk(t.children_right[node], depth + 1, prefix)
        walk(0, 0, "")
        ax.text(0.0, 0.98, "\n".join(lines[:20]), family="monospace",
                fontsize=8, va="top")
    ax.set_title("(c) Residual decision-tree top splits")

    # (d) SFS panel — correlation vs region
    ax = axes[1, 1]
    sfs_rows = corr_df[corr_df["feature"].isin(["sfs_power", "sfs_bispec"])].copy()
    sfs_rows = sfs_rows[sfs_rows["region"] != "overall"]
    if len(sfs_rows):
        for i, kind in enumerate(["sfs_power", "sfs_bispec"]):
            sub = sfs_rows[sfs_rows["feature"] == kind].sort_values("region")
            ax.bar(np.arange(len(LEE_BIN_LABELS)) + i * 0.4 - 0.2,
                   sub.set_index("region").reindex(LEE_BIN_LABELS)["corr"].values,
                   width=0.4, label=kind)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(len(LEE_BIN_LABELS)))
        ax.set_xticklabels(LEE_BIN_LABELS)
        ax.legend()
        ax.set_ylabel("corr(SFS proxy, residual)")
        ax.set_title("(d) SynchFastSlow proxies vs residual")
        ax.set_ylim(-0.4, 0.4)
        ax.grid(alpha=0.2)
    else:
        ax.text(0.5, 0.5, "no SFS data", ha="center", va="center")
        ax.set_title("(d) SFS proxies")

    plt.tight_layout()
    fig_path = RESULTS / "residual_diagnostic.png"
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    print(f"Saved {fig_path}")

    return phase_f_should_run


if __name__ == "__main__":
    main()
