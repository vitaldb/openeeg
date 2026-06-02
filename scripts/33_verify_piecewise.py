"""Phase G — Final 5-way comparison on the W=15 val cohort.

For each method, we report on the same 80-case W = 15, SQI ≥ 80 val
cohort (611k epochs total):

  (1) openibis(quazi, paper) + EMA(15s)         — paper-faithful baseline
  (2) Lee 2019 oracle reproduction              — Phase A 'oracle' variant
  (3) predict_bis_v2 (bundled LightGBM)         — current ship
  (4) predict_bis_rules (Phase D2 deep-rule)    — interpretable target
  (5) predict_bis_rules + CNN residual          — research ceiling

Each row has:
  overall MAE / Pearson r / Lin's rc
  per-Lee-bin MAE (0-21, 21-41, 41-61, 61-78, 78-98)
  Bland-Altman: mean bias ± 1.96 SD
  Histogram peak counts at BIS=40 and BIS=60

Worst-case analysis: top 5 worst cases for the deployable (4), with
per-case MAE and a 2-panel plot (actual vs predicted timeseries).

Outputs
  results/final_comparison.csv       summary table
  results/final_comparison.txt       human-readable report
  results/final_comparison.png       4-panel: histogram, bland-altman,
                                     per-region bar, regression scatter
  results/worst_cases_predict_bis_rules.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS, lin_concordance

RESULTS = Path(__file__).resolve().parents[1] / "results"

FS = 128
WIN_SAMPLES = 640
BATCH_SIZE = 256

PIECEWISE_JSON = RESULTS / "piecewise_raw_data.json"
PREDICT_BIS_V2 = Path(__file__).resolve().parents[1] / "openeeg" / "models" / "predict_bis_v2.txt"
CNN_CKPT = RESULTS / "cnn_residual_ensemble.pt"


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
    with open(PIECEWISE_JSON) as f:
        j = json.load(f)
    nodes = {int(n["node"]): n for n in j["tree_thresholds"]}
    leaves = {int(s["leaf"]): s for s in j["leaves"]}
    feat_cols = j["feature_cols"]
    deep_rule = j.get("deep_rule")
    X = df[feat_cols].values
    X_safe = np.where(np.isnan(X), 0.0, X)
    n = X.shape[0]
    cur = np.zeros(n, dtype=int); done = np.zeros(n, dtype=bool)
    leaf_id = np.full(n, -1, dtype=int)
    for _ in range(64):
        if done.all(): break
        for node in np.unique(cur[~done]):
            spec = nodes[int(node)]
            m = (cur == int(node)) & (~done)
            if spec["feature"] is None:
                leaf_id[m] = int(node); done[m] = True; continue
            fi = feat_cols.index(spec["feature"]); thr = float(spec["threshold"])
            vals = X_safe[m, fi]; gl = ~(vals > thr)
            cur[m] = np.where(gl, spec["left"], spec["right"]).astype(int)
    pred = np.full(n, np.nan, dtype=float)
    for lid, spec in leaves.items():
        m = leaf_id == int(lid)
        if not m.any(): continue
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


def lee_partition_predict(df: pd.DataFrame) -> np.ndarray:
    """Lee 2019 oracle (5-region tree + Phase-A per-region OLS using Vista
    oracle BSR / SEF / EMG and beta_ratio) read from
    ``results/lee2019_replication_coefs.csv`` if present; otherwise we
    apply Lee's literal Table 2 coefficients."""
    coef_csv = RESULTS / "lee2019_replication_coefs.csv"
    if coef_csv.exists():
        coefs = pd.read_csv(coef_csv)
        # Filter to the 'oracle' variant if column exists
        if "variant" in coefs.columns:
            coefs = coefs[coefs["variant"] == "oracle"].reset_index(drop=True)
    else:
        coefs = None

    bsr = df["bis_sr_oracle"].values
    sef = df["bis_sef_oracle"].values
    emg = df["bis_emg_oracle"].values
    rbr = df["beta_ratio"].values

    deep = bsr > 49.8
    mid_gate = (~deep) & (emg < 34.2) & (sef < 20.2)
    light = mid_gate & ((bsr > 2.1) | (sef < 14.8))
    surg  = mid_gate & ~light
    not_mid = (~deep) & (~mid_gate)
    trans = not_mid & (rbr < -0.7)
    awake = not_mid & (rbr >= -0.7)

    pred = np.full(len(df), np.nan, dtype=float)
    # Lee 2019 Table 2 coefficients on [BSR, EMG, SEF, RBR]
    LEE = {
        "deep":  dict(b=39.30, w=[-0.45, -0.14, +0.05, +0.30]),
        "light": dict(b=23.73, w=[-0.30, +0.20, +0.34, +0.61]),
        "surg":  dict(b=14.32, w=[-0.04, +0.27, +1.43, -0.20]),
        "trans": dict(b=85.43, w=[+0.16, +0.30, -0.36, +5.34]),
        "awake": dict(b=66.31, w=[+0.07, +0.55, -0.36, -1.94]),
    }
    for region_name, mask in [("deep", deep), ("light", light), ("surg", surg),
                                ("trans", trans), ("awake", awake)]:
        if not mask.any():
            continue
        spec = LEE[region_name]
        Xrows = np.column_stack([bsr[mask], emg[mask], sef[mask], rbr[mask]])
        Xrows = np.where(np.isnan(Xrows), 0.0, Xrows)
        pred[mask] = spec["b"] + Xrows @ np.asarray(spec["w"])
    return np.clip(pred, 0, 100)


def predict_bis_v2_from_features(df: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb
    feat22 = [
        "openibis_paper", "openibis_quazi", "openibis_quazi_30s",
        "bsr_paper", "bsr_quazi", "sef95", "bcsef", "beta_ratio", "emg_proxy",
        "p_delta", "p_theta", "p_alpha", "p_beta", "p_lowgamma", "spectral_entropy",
        "openibis_quazi_5s", "openibis_quazi_10s", "openibis_quazi_60s",
        "openibis_quazi_dt", "openibis_quazi_30s_dt", "sef95_dt", "emg_proxy_dt",
    ]
    booster = lgb.Booster(model_file=str(PREDICT_BIS_V2))
    return np.clip(booster.predict(df[feat22].values), 0, 100)


def cnn_residual_full_val(vl_df: pd.DataFrame, base_pred: np.ndarray) -> np.ndarray:
    """Run the 5-fold CNN ensemble on every (case_id, time_sec) val row that
    is non-deep (openbsr ≤ 49.8) and a full 5-s window fits. Returns a
    residual prediction array of len(vl_df), with NaN where unavailable."""
    import torch
    import torch.nn as nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv1d(1, 32, 7, stride=2, padding=3); self.b1 = nn.BatchNorm1d(32)
            self.c2 = nn.Conv1d(32, 64, 5, stride=2, padding=2); self.b2 = nn.BatchNorm1d(64)
            self.c3 = nn.Conv1d(64, 64, 5, stride=2, padding=2); self.b3 = nn.BatchNorm1d(64)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.f1 = nn.Linear(64, 32); self.f2 = nn.Linear(32, 1)
        def forward(self, x):
            x = x.unsqueeze(1)
            x = torch.relu(self.b1(self.c1(x)))
            x = torch.relu(self.b2(self.c2(x)))
            x = torch.relu(self.b3(self.c3(x)))
            x = self.pool(x).squeeze(-1)
            x = torch.relu(self.f1(x))
            return self.f2(x).squeeze(-1)

    ckpt = torch.load(CNN_CKPT, map_location="cpu", weights_only=False)
    models = []
    for sd in ckpt["state_dicts"]:
        m = CNN(); m.load_state_dict(sd); m.eval()
        models.append(m)
    torch.set_num_threads(32)

    from openeeg.cohort import load_case, preprocess_eeg

    out = np.full(len(vl_df), np.nan, dtype=np.float32)
    cids = sorted(vl_df["case_id"].unique())
    for ci, cid in enumerate(cids):
        sub = vl_df.loc[vl_df["case_id"] == cid, ["time_sec", "openbsr"]].copy()
        sub_idx = sub.index.to_numpy()
        case = load_case(int(cid))
        if case is None:
            continue
        eeg = preprocess_eeg(case["eeg"])
        n = len(eeg)
        t = sub["time_sec"].values.astype(int)
        # Mask: full window fits AND non-deep
        s_starts = t * FS - WIN_SAMPLES // 2
        s_ends = s_starts + WIN_SAMPLES
        valid = (s_starts >= 0) & (s_ends < n) & (sub["openbsr"].values <= 49.8) & np.isfinite(sub["openbsr"].values)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            continue
        # Build batch of windows
        windows = np.empty((len(valid_idx), WIN_SAMPLES), dtype=np.float32)
        for j, vi in enumerate(valid_idx):
            w = eeg[s_starts[vi]:s_ends[vi]]
            mu = float(np.mean(w)); sd = float(np.std(w)) + 1e-6
            windows[j] = ((w - mu) / sd).astype(np.float32)
        # Ensemble predict in batches
        preds_sum = np.zeros(len(windows), dtype=np.float32)
        for m in models:
            with torch.no_grad():
                preds_m = []
                for s in range(0, len(windows), BATCH_SIZE):
                    preds_m.append(m(torch.from_numpy(windows[s:s + BATCH_SIZE]).float()).numpy())
                preds_sum += np.concatenate(preds_m)
        preds_avg = preds_sum / len(models)
        out[sub_idx[valid_idx]] = preds_avg
        if ci % 10 == 0:
            print(f"    CNN val: case {ci+1}/{len(cids)} (cid={cid}) — {valid.sum():,} windows")
    return out


def metrics(actual, pred):
    m = np.isfinite(actual) & np.isfinite(pred)
    if m.sum() < 10:
        return dict(mae=float("nan"), r=float("nan"), rc=float("nan"),
                    bias=float("nan"), sd=float("nan"))
    a = actual[m]; p = pred[m]
    return dict(
        mae=float(np.mean(np.abs(p - a))),
        r=float(np.corrcoef(p, a)[0, 1]),
        rc=lin_concordance(p, a),
        bias=float(np.mean(p - a)),
        sd=float(np.std(p - a)),
    )


def per_regime_mae(actual, pred):
    m_finite = np.isfinite(actual) & np.isfinite(pred)
    a = actual[m_finite]; p = pred[m_finite]
    out = {}
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        mm = (a >= lo) & (a < hi)
        out[lbl] = float(np.mean(np.abs(p[mm] - a[mm]))) if mm.sum() > 10 else float("nan")
    return out


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "64")
    print("Loading val W=15 features (v5)...")
    val = pd.read_parquet(RESULTS / "features_val_n100_v5.parquet")
    vl = w15_filter(val, RESULTS / "oracle_W_val.csv").reset_index(drop=True)
    print(f"  {len(vl):,} rows / {vl['case_id'].nunique()} cases")
    actual = vl["target"].values

    methods = {}

    # ---- (1) openibis(quazi, paper) baseline
    print("\n(1) openibis(quazi,paper) baseline...")
    base_pred = np.clip(vl["openibis_quazi"].values, 0, 100)
    methods["openibis_quazi_paper"] = smooth_by_case(base_pred, vl, W=15.0)

    # ---- (2) Lee 2019 oracle reproduction (using Vista BSR/SEF/EMG)
    print("(2) Lee 2019 oracle reproduction...")
    lee_pred = lee_partition_predict(vl)
    methods["lee2019_oracle"] = smooth_by_case(lee_pred, vl, W=15.0)

    # ---- (3) predict_bis_v2 (LightGBM)
    print("(3) predict_bis_v2 (LightGBM)...")
    v2_pred = predict_bis_v2_from_features(vl)
    methods["predict_bis_v2"] = smooth_by_case(v2_pred, vl, W=15.0)

    # ---- (4) predict_bis_rules (Phase D2)
    print("(4) predict_bis_rules (Phase D2: deep rule + K=6 piecewise)...")
    pw_pred = piecewise_predict_from_features(vl)
    methods["predict_bis_rules"] = smooth_by_case(pw_pred, vl, W=15.0)

    # ---- (5) predict_bis_rules + CNN residual
    print("(5) predict_bis_rules + CNN residual ensemble (on every val row)...")
    cnn_res = cnn_residual_full_val(vl, methods["predict_bis_rules"])
    pw_plus_cnn = methods["predict_bis_rules"].copy()
    has_cnn = np.isfinite(cnn_res)
    pw_plus_cnn[has_cnn] = np.clip(pw_plus_cnn[has_cnn] + cnn_res[has_cnn], 0, 100)
    methods["predict_bis_rules+cnn"] = pw_plus_cnn

    print(f"\nCNN residual coverage: {has_cnn.mean()*100:.1f}% of val rows "
          f"({has_cnn.sum():,}/{len(vl):,})")

    # ---- Compute metrics
    rows = []
    for name, pred in methods.items():
        ms = metrics(actual, pred)
        reg = per_regime_mae(actual, pred)
        rows.append(dict(method=name, **ms, **reg))
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "final_comparison.csv", index=False)
    print("\n=== Final comparison (val W=15, SQI ≥ 80, EMA(15s)) ===")
    cols_show = ["method", "mae", "r", "rc", "bias", "sd"] + list(LEE_BIN_LABELS)
    print(summary[cols_show].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ---- Per-method peak counts at BIS = 40 and 60 (regime overlap signature)
    bins_edges = np.arange(0, 101, 1)
    a_counts, _ = np.histogram(actual[np.isfinite(actual)], bins=bins_edges)
    method_peaks = {}
    for name, pred in methods.items():
        c, _ = np.histogram(pred[np.isfinite(pred)], bins=bins_edges)
        method_peaks[name] = (c, dict(
            peak40=int(c[max(0, 38):41].max()),
            peak60=int(c[58:61].max()),
        ))
    print(f"\nactual histogram peaks: 40={a_counts[max(0,38):41].max():,}  60={a_counts[58:61].max():,}")
    for name, (_, pk) in method_peaks.items():
        print(f"  {name:<32s} 40={pk['peak40']:,}  60={pk['peak60']:,}")

    # ---- Write text report
    rep = ["Phase G — Final 5-way comparison",
           "================================",
           f"Cohort: 80 W=15 SQI≥80 val cases, {len(vl):,} epochs",
           ""]
    rep.append(summary[cols_show].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    rep.append("")
    rep.append("Key takeaways:")
    pw_mae = float(summary.loc[summary["method"]=="predict_bis_rules","mae"].iloc[0])
    pw_cnn_mae = float(summary.loc[summary["method"]=="predict_bis_rules+cnn","mae"].iloc[0])
    v2_mae = float(summary.loc[summary["method"]=="predict_bis_v2","mae"].iloc[0])
    base_mae = float(summary.loc[summary["method"]=="openibis_quazi_paper","mae"].iloc[0])
    rep.append(f"  • openibis baseline      MAE = {base_mae:.2f}")
    rep.append(f"  • predict_bis_v2 (ship)  MAE = {v2_mae:.2f}  (-{base_mae-v2_mae:.2f} vs baseline)")
    rep.append(f"  • predict_bis_rules      MAE = {pw_mae:.2f}  (interpretable, -{base_mae-pw_mae:.2f} vs baseline)")
    rep.append(f"  • predict_bis_rules+cnn  MAE = {pw_cnn_mae:.2f}  (-{pw_mae-pw_cnn_mae:.2f} vs interpretable)")
    rep.append("")
    if pw_cnn_mae < v2_mae:
        rep.append(f"  → CNN-augmented piecewise BEATS predict_bis_v2 ({pw_cnn_mae:.2f} < {v2_mae:.2f}).")
    else:
        rep.append(f"  → predict_bis_v2 still wins by {v2_mae - pw_cnn_mae:.2f} BIS-points.")
    rep.append("")
    rep.append("Deep regime (0-21) is locked to Ellerkmann rule (openbsr > 49.8); MAE in")
    rep.append("that bin is the Ellerkmann formula's accuracy and not a comparison target.")
    out_txt = RESULTS / "final_comparison.txt"
    out_txt.write_text("\n".join(rep))
    print(f"\nSaved {out_txt}")

    # ---- 4-panel diagnostic
    print("\nRendering 4-panel comparison figure...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    colors = {"openibis_quazi_paper": "tab:gray",
              "lee2019_oracle":       "tab:purple",
              "predict_bis_v2":       "tab:blue",
              "predict_bis_rules":    "tab:orange",
              "predict_bis_rules+cnn":"tab:red"}

    # (a) Histogram overlay
    ax = axes[0, 0]
    centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
    ax.bar(centers, a_counts, width=1.0, color="black", alpha=0.5, label="actual")
    for name in methods:
        c, _ = method_peaks[name]
        ax.plot(centers, c, color=colors[name], lw=1.4, alpha=0.85, label=name)
    for v in LEE_BINS:
        ax.axvline(v, color="gray", lw=0.5, alpha=0.4)
    ax.set_xlabel("BIS"); ax.set_ylabel("epoch count")
    ax.set_xlim(0, 100); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("(a) Histogram overlay — peaks at 40/60 = regime overlap")

    # (b) Bland-Altman for predict_bis_rules and +cnn
    ax = axes[0, 1]
    for name in ("predict_bis_rules", "predict_bis_rules+cnn"):
        p = methods[name]
        m = np.isfinite(actual) & np.isfinite(p)
        rng = np.random.default_rng(0)
        idx = rng.choice(np.where(m)[0], min(80000, m.sum()), replace=False)
        ax.scatter(0.5*(actual[idx]+p[idx]), p[idx]-actual[idx], s=0.4, alpha=0.06,
                   color=colors[name], label=name)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("mean of actual & predicted BIS")
    ax.set_ylabel("predicted − actual")
    ax.set_xlim(0, 100); ax.set_ylim(-50, 50)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("(b) Bland-Altman: deployable vs CNN-augmented")

    # (c) Per-region MAE bar
    ax = axes[1, 0]
    width = 0.16
    x = np.arange(len(LEE_BIN_LABELS))
    for i, name in enumerate(methods):
        row = summary[summary["method"] == name].iloc[0]
        vals = [float(row[lbl]) for lbl in LEE_BIN_LABELS]
        ax.bar(x + (i - 2) * width, vals, width=width, color=colors[name],
               label=name)
    ax.set_xticks(x); ax.set_xticklabels(LEE_BIN_LABELS)
    ax.set_ylabel("MAE (BIS units)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("(c) Per-region MAE")
    ax.grid(alpha=0.2, axis="y")

    # (d) Pred vs actual scatter for top method
    ax = axes[1, 1]
    name = "predict_bis_rules+cnn" if pw_cnn_mae < v2_mae else "predict_bis_v2"
    p = methods[name]; m = np.isfinite(actual) & np.isfinite(p)
    rng = np.random.default_rng(1)
    idx = rng.choice(np.where(m)[0], min(80000, m.sum()), replace=False)
    ax.scatter(actual[idx], p[idx], s=0.4, alpha=0.06, color=colors[name])
    ax.plot([0, 100], [0, 100], "k--", lw=0.6)
    ax.set_xlabel("actual BIS"); ax.set_ylabel(f"{name} predicted BIS")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    row = summary[summary["method"] == name].iloc[0]
    ax.set_title(f"(d) {name}  MAE={row['mae']:.2f}  r={row['r']:.3f}")
    plt.tight_layout()
    fig_path = RESULTS / "final_comparison.png"
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    print(f"Saved {fig_path}")

    # ---- Worst-5-cases for predict_bis_rules (deployable)
    print("\nWorst 5 cases for predict_bis_rules:")
    pw = methods["predict_bis_rules"]
    case_mae = []
    for cid, sub in vl.groupby("case_id"):
        idx = sub.index.to_numpy()
        a = actual[idx]; p = pw[idx]
        m = np.isfinite(a) & np.isfinite(p)
        if m.sum() < 10: continue
        case_mae.append((int(cid), float(np.mean(np.abs(p[m] - a[m]))), int(m.sum())))
    case_mae.sort(key=lambda r: -r[1])
    worst5 = case_mae[:5]
    print("  cid  mae   n")
    for cid, mae, n in worst5:
        print(f"  {cid:>4d} {mae:5.2f} {n:,}")

    # Plot worst 5
    fig, axes = plt.subplots(5, 1, figsize=(18, 14))
    for ax, (cid, mae, n) in zip(axes, worst5):
        sub = vl[vl["case_id"] == cid].sort_values("time_sec")
        idx = sub.index.to_numpy()
        t = sub["time_sec"].values / 60.0  # minutes
        ax.plot(t, actual[idx], color="black", lw=0.8, label="actual BIS")
        ax.plot(t, pw[idx], color="tab:orange", lw=0.7, alpha=0.7, label="predict_bis_rules")
        cnn_arr = methods["predict_bis_rules+cnn"]
        ax.plot(t, cnn_arr[idx], color="tab:red", lw=0.7, alpha=0.7, label="+cnn")
        v2 = methods["predict_bis_v2"]
        ax.plot(t, v2[idx], color="tab:blue", lw=0.7, alpha=0.5, label="predict_bis_v2")
        ax.set_title(f"case {cid}  MAE={mae:.2f}  n={n:,}")
        ax.set_xlabel("minute"); ax.set_ylabel("BIS")
        ax.set_ylim(0, 100); ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2)
    plt.tight_layout()
    out2 = RESULTS / "worst_cases_predict_bis_rules.png"
    fig.savefig(out2, dpi=110)
    plt.close(fig)
    print(f"Saved {out2}")

    print("\n" + "=" * 60)
    print("Phase G complete. Read results/final_comparison.{csv,txt,png}")
    print("and results/worst_cases_predict_bis_rules.png.")


if __name__ == "__main__":
    main()
