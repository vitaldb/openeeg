"""Phase F — Small 1-D CNN on raw EEG to learn the piecewise-linear residual.

Goal: test whether raw EEG carries information *beyond* the 23 hand-crafted
features the piecewise model uses. The 0-21 region had |corr(residual,
openibis_quazi_60s)| = 0.94 in Phase E, driven by the 660 "missed-deep"
val rows where actual BIS<21 but openbsr stays ≈ 0 (suppression paradox).
A CNN may pick this up directly from the EEG waveform.

Architecture (CPU-friendly, ~50 k parameters):
  Conv1d(1 → 32, k=7, s=2)  + BN + ReLU
  Conv1d(32 → 64, k=5, s=2) + BN + ReLU
  Conv1d(64 → 64, k=5, s=2) + BN + ReLU
  AdaptiveAvgPool1d(1)
  Linear(64 → 32) + ReLU
  Linear(32 → 1)

Training data:
  * 397 W=15 train cases
  * Per case, sample ``WINDOWS_PER_CASE`` 5-s EEG windows uniformly
  * For each window, target = residual at the centre 1-Hz epoch
    where residual = actual_BIS − predict_bis_rules(eeg)
  * 5-fold case-grouped CV

Evaluation:
  * Per val case, score the CNN on ``EVAL_WINDOWS_PER_CASE`` random
    timestamps drawn from the 80 W=15 val cases
  * Compare overall and per-region MAE for
      baseline: piecewise + EMA(15 s)
      treated : piecewise + CNN residual + EMA(15 s)
  * Headline pass criterion: treated MAE ≤ baseline − 0.30 BIS-points.

If the CNN does NOT clear that bar, Phase F is reported as null and
nothing is shipped from this script.

Outputs
  results/cnn_residual_summary.csv     baseline vs treated MAE, per region
  results/cnn_residual_diagnostic.png  residual scatter before/after
  results/cnn_residual_curve.csv       train/val loss per epoch
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS

RESULTS = Path(__file__).resolve().parents[1] / "results"

WINDOW_S = 5.0
FS = 128
WIN_SAMPLES = int(WINDOW_S * FS)  # 640
WINDOWS_PER_CASE = 80   # train: 397 × 80 = ~32k windows
EVAL_WINDOWS_PER_CASE = 200  # val: 80 × 200 = 16k
BATCH_SIZE = 256
EPOCHS = 12
LR = 1e-3
N_FOLDS = 5

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


def _sample_windows_for_case(args):
    """Worker: extract WIN_SAMPLES-long EEG windows centred at random
    valid 1-Hz timestamps for one case, returning (X, y_resid, t_sec)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    cid, t_pool, residuals_pool, k_pick, seed = args
    from openeeg.cohort import load_case, preprocess_eeg
    case = load_case(int(cid))
    if case is None:
        return None
    eeg = preprocess_eeg(case["eeg"])
    n = len(eeg)
    rng = np.random.default_rng(seed)
    valid = (t_pool * FS - WIN_SAMPLES // 2 >= 0) & \
            (t_pool * FS + WIN_SAMPLES // 2 < n) & \
            np.isfinite(residuals_pool)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return None
    take = min(k_pick, len(valid_idx))
    picked = rng.choice(valid_idx, take, replace=False)
    X = np.empty((take, WIN_SAMPLES), dtype=np.float32)
    y = np.empty(take, dtype=np.float32)
    ts = np.empty(take, dtype=np.int32)
    for i, idx in enumerate(picked):
        t = int(t_pool[idx])
        s = t * FS - WIN_SAMPLES // 2
        w = eeg[s:s + WIN_SAMPLES]
        # z-score per window
        mu = float(np.mean(w)); sd = float(np.std(w)) + 1e-6
        X[i] = ((w - mu) / sd).astype(np.float32)
        y[i] = float(residuals_pool[idx])
        ts[i] = t
    return int(cid), X, y, ts


def build_window_dataset(df: pd.DataFrame, residual: np.ndarray,
                          windows_per_case: int, seed: int):
    """Sample windows per case in parallel; return concatenated tensors and
    a per-window case_id array (for case-grouped CV splitting)."""
    cids = sorted(df["case_id"].unique())
    print(f"  building windows: {len(cids)} cases × ~{windows_per_case} = "
          f"~{len(cids)*windows_per_case:,} samples")
    # Build (cid, t_pool, residuals_pool) per case
    work = []
    for i, cid in enumerate(cids):
        sub = df.loc[df["case_id"] == cid, ["time_sec"]].copy()
        sub_idx = sub.index.to_numpy()
        t_pool = sub["time_sec"].values.astype(np.int32)
        residuals_pool = residual[sub_idx]
        work.append((int(cid), t_pool, residuals_pool, windows_per_case, seed + i))
    with Pool(32) as pool:
        outputs = [r for r in pool.map(_sample_windows_for_case, work) if r is not None]
    Xs = []; ys = []; cs = []; ts = []
    for cid, X, y, t in outputs:
        Xs.append(X); ys.append(y)
        cs.append(np.full(len(y), cid, dtype=np.int32))
        ts.append(t)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    c = np.concatenate(cs); ts_all = np.concatenate(ts)
    print(f"  built {len(X):,} windows across {len(np.unique(c))} cases")
    return X, y, c, ts_all


def _torch_train_one_fold(X_tr, y_tr, X_v, y_v, fold_idx):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv1d(1, 32, 7, stride=2, padding=3)
            self.b1 = nn.BatchNorm1d(32)
            self.c2 = nn.Conv1d(32, 64, 5, stride=2, padding=2)
            self.b2 = nn.BatchNorm1d(64)
            self.c3 = nn.Conv1d(64, 64, 5, stride=2, padding=2)
            self.b3 = nn.BatchNorm1d(64)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.f1 = nn.Linear(64, 32)
            self.f2 = nn.Linear(32, 1)

        def forward(self, x):
            x = x.unsqueeze(1)
            x = torch.relu(self.b1(self.c1(x)))
            x = torch.relu(self.b2(self.c2(x)))
            x = torch.relu(self.b3(self.c3(x)))
            x = self.pool(x).squeeze(-1)
            x = torch.relu(self.f1(x))
            return self.f2(x).squeeze(-1)

    torch.manual_seed(fold_idx)
    torch.set_num_threads(32)
    device = "cpu"
    model = CNN().to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()
    Xt_tr = torch.from_numpy(X_tr).float()
    yt_tr = torch.from_numpy(y_tr).float()
    Xt_v  = torch.from_numpy(X_v).float()
    yt_v  = torch.from_numpy(y_v).float()
    history = []
    best_v = float("inf"); best_state = None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt_tr))
        ep_loss = 0.0; n_seen = 0
        for s in range(0, len(perm), BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            xb = Xt_tr[idx].to(device); yb = yt_tr[idx].to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward(); opt.step()
            ep_loss += float(loss) * len(idx); n_seen += len(idx)
        model.eval()
        with torch.no_grad():
            v_preds = []
            for s in range(0, len(Xt_v), BATCH_SIZE):
                v_preds.append(model(Xt_v[s:s + BATCH_SIZE].to(device)))
            v_pred = torch.cat(v_preds)
            v_mae = float((v_pred - yt_v).abs().mean())
        tr_mae = ep_loss / max(n_seen, 1)
        history.append(dict(fold=fold_idx, epoch=ep, train_mae=tr_mae, val_mae=v_mae))
        if v_mae < best_v:
            best_v = v_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"    fold {fold_idx}  epoch {ep}: train_mae={tr_mae:.3f}  val_mae={v_mae:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, best_v


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "64")
    print("Loading W=15 train & val (v5)...")
    train = pd.read_parquet(RESULTS / "features_train_n500_v5.parquet")
    val   = pd.read_parquet(RESULTS / "features_val_n100_v5.parquet")
    tr_w = w15_filter(train, RESULTS / "oracle_W_train.csv").reset_index(drop=True)
    vl_w = w15_filter(val,   RESULTS / "oracle_W_val.csv").reset_index(drop=True)
    print(f"  train: {len(tr_w):,} rows / {tr_w['case_id'].nunique()} cases")
    print(f"  val:   {len(vl_w):,} rows / {vl_w['case_id'].nunique()} cases")

    print("\nComputing piecewise predictions (deep-rule + K=6 + EMA(15s))...")
    tr_pred = piecewise_predict_from_features(tr_w)
    tr_pred_sm = smooth_by_case(tr_pred, tr_w, W=15.0)
    tr_resid = tr_w["target"].values - tr_pred_sm
    vl_pred = piecewise_predict_from_features(vl_w)
    vl_pred_sm = smooth_by_case(vl_pred, vl_w, W=15.0)
    vl_resid = vl_w["target"].values - vl_pred_sm
    base_mae = float(np.nanmean(np.abs(vl_resid)))
    print(f"  baseline val MAE = {base_mae:.3f}")

    # Restrict CNN learning to NON-DEEP rows (rule already handles openbsr>49.8)
    tr_nd = tr_w["openbsr"].values <= 49.8
    vl_nd = vl_w["openbsr"].values <= 49.8

    # Build window datasets — non-deep only
    print("\nSampling train windows...")
    tr_idx_nd = tr_w[tr_nd].index.to_numpy()
    tr_resid_nd = np.full(len(tr_w), np.nan); tr_resid_nd[tr_idx_nd] = tr_resid[tr_idx_nd]
    Xt, yt, ct, _ = build_window_dataset(tr_w, tr_resid_nd, WINDOWS_PER_CASE, seed=0)
    print(f"  train residual stats: mean={yt.mean():+.2f}  std={yt.std():.2f}  median|y|={np.median(np.abs(yt)):.2f}")

    print("\nSampling val windows...")
    vl_idx_nd = vl_w[vl_nd].index.to_numpy()
    vl_resid_nd = np.full(len(vl_w), np.nan); vl_resid_nd[vl_idx_nd] = vl_resid[vl_idx_nd]
    Xv, yv, cv, tv = build_window_dataset(vl_w, vl_resid_nd, EVAL_WINDOWS_PER_CASE, seed=1)
    print(f"  val residual stats:   mean={yv.mean():+.2f}  std={yv.std():.2f}  median|y|={np.median(np.abs(yv)):.2f}")

    # ---- 5-fold case-grouped CV
    print("\nTraining 5-fold CNN (case-grouped)...")
    rng = np.random.default_rng(42)
    cases = np.unique(ct)
    rng.shuffle(cases)
    folds = np.array_split(cases, N_FOLDS)
    fold_cv_history = []
    all_val_preds = []
    fold_state_dicts = []
    for fi, hold in enumerate(folds):
        tr_mask = ~np.isin(ct, hold)
        v_mask = np.isin(ct, hold)
        print(f"\n  Fold {fi}: train {tr_mask.sum():,}  cv {v_mask.sum():,}")
        model, hist, best_v = _torch_train_one_fold(
            Xt[tr_mask], yt[tr_mask], Xt[v_mask], yt[v_mask], fi)
        fold_cv_history.extend(hist)
        # Score the held-out val set
        import torch
        with torch.no_grad():
            preds = []
            for s in range(0, len(Xv), BATCH_SIZE):
                preds.append(model(torch.from_numpy(Xv[s:s + BATCH_SIZE]).float()))
            preds = torch.cat(preds).numpy()
        all_val_preds.append(preds)
        fold_state_dicts.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
        del model

    cnn_val_pred = np.mean(all_val_preds, axis=0)  # ensemble across folds
    print(f"\nEnsemble CNN val MAE on residual targets: {float(np.mean(np.abs(cnn_val_pred - yv))):.3f}")

    # ---- Persist the 5-fold ensemble for downstream Phase G use
    import torch
    ckpt = RESULTS / "cnn_residual_ensemble.pt"
    torch.save(dict(state_dicts=fold_state_dicts,
                    arch="cnn_v1",
                    win_samples=WIN_SAMPLES,
                    fs=FS), ckpt)
    print(f"Saved CNN ensemble checkpoint → {ckpt.name}")

    # ---- Build the combined predictor on val
    # For each val (case_id, time_sec) we have a window y_resid + cnn_pred
    val_assess = pd.DataFrame(dict(
        case_id=cv.astype(int), time_sec=tv.astype(int),
        target_resid=yv, cnn_resid=cnn_val_pred,
    ))
    val_assess = val_assess.merge(
        vl_w[["case_id", "time_sec", "target"]],
        on=["case_id", "time_sec"], how="left",
    )
    val_assess = val_assess.merge(
        pd.DataFrame(dict(case_id=vl_w["case_id"].values,
                          time_sec=vl_w["time_sec"].values,
                          base_pred=vl_pred_sm)),
        on=["case_id", "time_sec"], how="left",
    )
    val_assess = val_assess.dropna(subset=["target", "base_pred"]).reset_index(drop=True)
    treated_pred = np.clip(val_assess["base_pred"].values + val_assess["cnn_resid"].values, 0, 100)
    base_mae_subset = float(np.mean(np.abs(val_assess["base_pred"].values - val_assess["target"].values)))
    treated_mae    = float(np.mean(np.abs(treated_pred                   - val_assess["target"].values)))

    print(f"\n=== Phase F gating comparison on sampled val windows ===")
    print(f"  baseline (piecewise + EMA15)   MAE = {base_mae_subset:.3f}")
    print(f"  treated  (baseline + CNN res)  MAE = {treated_mae:.3f}")
    print(f"  ΔMAE = {treated_mae - base_mae_subset:+.3f}")

    # ---- Per-region comparison
    rows = []
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        m = (val_assess["target"].values >= lo) & (val_assess["target"].values < hi)
        if m.sum() < 30:
            rows.append(dict(region=lbl, n=int(m.sum()),
                             base_mae=float("nan"), treated_mae=float("nan"),
                             delta=float("nan")))
            continue
        b = float(np.mean(np.abs(val_assess["base_pred"].values[m] - val_assess["target"].values[m])))
        t = float(np.mean(np.abs(treated_pred[m] - val_assess["target"].values[m])))
        rows.append(dict(region=lbl, n=int(m.sum()), base_mae=b, treated_mae=t, delta=t - b))
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(RESULTS / "cnn_residual_summary.csv", index=False)
    print("\n  per-region comparison:")
    print(cmp_df.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # ---- Save loss curve
    hist_df = pd.DataFrame(fold_cv_history)
    hist_df.to_csv(RESULTS / "cnn_residual_curve.csv", index=False)

    # ---- Diagnostic figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    ax = axes[0]
    actual = val_assess["target"].values
    base_p = val_assess["base_pred"].values
    ax.scatter(base_p, actual, s=1, alpha=0.1, color="tab:blue", label="baseline")
    ax.scatter(treated_pred, actual, s=1, alpha=0.1, color="tab:red", label="treated")
    ax.plot([0, 100], [0, 100], "k--", lw=0.6)
    ax.set_xlabel("predicted BIS"); ax.set_ylabel("actual BIS")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title(f"Predicted vs actual  base MAE={base_mae_subset:.2f}  treated MAE={treated_mae:.2f}")

    ax = axes[1]
    ax.scatter(base_p, val_assess["target_resid"].values, s=1, alpha=0.1, color="tab:blue", label="target resid")
    ax.scatter(base_p, val_assess["cnn_resid"].values, s=1, alpha=0.1, color="tab:red", label="cnn resid")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("baseline predicted BIS")
    ax.set_ylabel("residual")
    ax.legend()
    ax.set_xlim(0, 100); ax.set_ylim(-50, 50)
    ax.set_title("Target residual vs CNN-predicted residual")

    ax = axes[2]
    for fi in range(N_FOLDS):
        sub = hist_df[hist_df["fold"] == fi]
        ax.plot(sub["epoch"], sub["train_mae"], color="tab:blue", alpha=0.4)
        ax.plot(sub["epoch"], sub["val_mae"], color="tab:red", alpha=0.4)
    ax.plot([], [], color="tab:blue", label="train MAE")
    ax.plot([], [], color="tab:red", label="cv MAE")
    ax.set_xlabel("epoch"); ax.set_ylabel("MAE")
    ax.legend()
    ax.set_title("CNN training curves (5 folds)")

    plt.tight_layout()
    fig_path = RESULTS / "cnn_residual_diagnostic.png"
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    print(f"\nSaved {fig_path}")

    # ---- Final verdict
    pass_gate = (base_mae_subset - treated_mae) >= 0.30
    print("\n" + "=" * 60)
    print("Phase F verdict")
    print("=" * 60)
    if pass_gate:
        print(f"PASS — CNN reduces val MAE by {base_mae_subset - treated_mae:.3f} ≥ 0.30 BIS-points.")
        print("  Phase G should include the CNN residual variant in the comparison.")
    else:
        print(f"NULL — CNN ΔMAE = {treated_mae - base_mae_subset:+.3f}, below the 0.30 cliff.")
        print("  Conclusion: no new sub-parameter signal is recoverable from raw EEG")
        print("  beyond what the 23 hand-crafted features already encode. Phase G should")
        print("  ship the piecewise model as the deployable target.")


if __name__ == "__main__":
    main()
