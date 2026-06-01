"""Phase 3 — extract per-second feature rows from VitalDB cases.

Output: results/features_<fold>_n<N>.parquet with one row per 1-second
epoch:
  * 16 engineered features (openibis predictions, BSR variants, spectral
    features, sub-band powers, EMG proxy + oracle, 30 s context)
  * target = actual BIS Vista value at that second
  * sqi, case_id, time_sec

Filtered to SQI ≥ 80 and non-NaN target.

Usage::

    python scripts/06_extract_features.py --fold val --n 100
    python scripts/06_extract_features.py --fold train --n 500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg import openibis, sef, bcsef, beta_ratio, emg_estimate
from openeeg.openibis import bsr as openibis_bsr, _psd, _band, _n_epochs, STRIDE, PSD_WINDOW, FS, N_BINS
from openeeg.cohort import load_case, preprocess_eeg, caseids_bis, split

SQI_THRESH = 80


def per_epoch_psd(eeg: np.ndarray, hp_hz: float = 0.65) -> np.ndarray:
    """Identical PSD path to openeeg.features._per_epoch_psd, kept local
    so this script does not depend on private attributes."""
    b, a = signal.butter(2, hp_hz / (FS / 2.0), "high")
    eeg_hp = signal.lfilter(b, a, eeg)
    N = _n_epochs(eeg)
    psd_arr = np.full((N, N_BINS), np.nan)
    for n in range(N):
        s = (n + 4) * STRIDE
        e = s + PSD_WINDOW
        if e > len(eeg_hp):
            continue
        psd_arr[n] = _psd(eeg_hp[s:e])
    return psd_arr


def band_db(psd_arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Mean dB power in a band, per epoch."""
    v = psd_arr[:, _band(lo, hi)]
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.maximum(np.nanmean(v, axis=1), 1e-30))


def spectral_entropy(psd_arr: np.ndarray, band: tuple[float, float] = (0.5, 30.0)) -> np.ndarray:
    """Shannon entropy of the normalised PSD over band."""
    sub = psd_arr[:, _band(*band)]
    norm = sub / np.maximum(np.nansum(sub, axis=1, keepdims=True), 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(norm * np.log(np.maximum(norm, 1e-30)), axis=1)


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average; NaN-safe at edges via mean-of-mode='same'."""
    kernel = np.ones(w) / w
    return np.convolve(np.where(np.isnan(x), 0.0, x), kernel, mode="same")


def features_for_case(case: dict) -> pd.DataFrame | None:
    """Build a per-second feature DataFrame for one case."""
    eeg = preprocess_eeg(case["eeg"])
    fs = case["fs"]

    # Predictions (2 Hz)
    pred_paper = openibis(eeg, bsr="paper", deep="paper")
    pred_quazi = openibis(eeg, bsr="quazi", deep="paper")
    bsr_paper_v = openibis_bsr(eeg, kind="paper")
    bsr_quazi_v = openibis_bsr(eeg, kind="quazi")

    # Spectral features (2 Hz)
    sef_v = sef(eeg)
    bcsef_v = bcsef(eeg)
    beta_v = beta_ratio(eeg)
    emg_proxy = emg_estimate(eeg)

    # Sub-band powers (2 Hz) — one PSD compute
    psd_arr = per_epoch_psd(eeg)
    p_delta = band_db(psd_arr, 0.5, 4)
    p_theta = band_db(psd_arr, 4, 8)
    p_alpha = band_db(psd_arr, 8, 13)
    p_beta  = band_db(psd_arr, 13, 30)
    p_lowgamma = band_db(psd_arr, 30, 47)
    se = spectral_entropy(psd_arr)

    # Context features (BIS Vista smooths over 15-30 s)
    pred_quazi_30s = rolling_mean(pred_quazi, 60)  # 60 strides @ 2Hz = 30 s

    # Downsample 2 Hz → 1 Hz to align with BIS
    feat = pd.DataFrame({
        "openibis_paper":    pred_paper[::2],
        "openibis_quazi":    pred_quazi[::2],
        "openibis_quazi_30s": pred_quazi_30s[::2],
        "bsr_paper":         bsr_paper_v[::2],
        "bsr_quazi":         bsr_quazi_v[::2],
        "sef95":             sef_v[::2],
        "bcsef":             bcsef_v[::2],
        "beta_ratio":        beta_v[::2],
        "emg_proxy":         emg_proxy[::2],
        "p_delta":           p_delta[::2],
        "p_theta":           p_theta[::2],
        "p_alpha":           p_alpha[::2],
        "p_beta":            p_beta[::2],
        "p_lowgamma":        p_lowgamma[::2],
        "spectral_entropy":  se[::2],
    })

    # Align targets / oracle features (already at 1 Hz)
    n = min(len(feat), len(case["bis"]), len(case["sqi"]),
            len(case["emg"]), len(case["sef"]),
            len(case["sr"]), len(case["totpow"]))
    feat = feat.iloc[:n].copy()
    feat["bis_emg_oracle"] = case["emg"][:n]
    feat["bis_sef_oracle"] = case["sef"][:n]
    feat["bis_sr_oracle"]  = case["sr"][:n]   # Lee 2019's "BSR"
    feat["bis_totpow_oracle"] = case["totpow"][:n]
    feat["target"] = case["bis"][:n]
    feat["sqi"] = case["sqi"][:n]
    feat["case_id"] = case["caseid"]
    feat["time_sec"] = np.arange(n, dtype=np.int32)

    # Filter: SQI ≥ 80 and non-NaN target
    mask = (~feat["target"].isna()) & (~feat["sqi"].isna()) & (feat["sqi"] >= SQI_THRESH)
    feat = feat.loc[mask].reset_index(drop=True)
    if len(feat) < 60:
        return None
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=100, help="Cases to extract from fold.")
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_parquet = repo / "results" / f"features_{args.fold}_n{args.n}.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    fold_ids = split(caseids_bis(), fold=args.fold)[args.offset : args.offset + args.n]
    print(f"Extracting fold={args.fold}, N={len(fold_ids)} cases "
          f"({fold_ids[0]} … {fold_ids[-1]}); output → {out_parquet.name}")

    parts = []
    n_used = 0
    n_skip = 0
    t0 = time.time()
    for cid in fold_ids:
        case = load_case(cid)
        if case is None:
            n_skip += 1
            continue
        try:
            df = features_for_case(case)
        except Exception as exc:
            print(f"  case {cid}: SKIP ({exc!r})")
            n_skip += 1
            continue
        if df is None or len(df) == 0:
            n_skip += 1
            continue
        parts.append(df)
        n_used += 1
        if n_used % 25 == 0:
            elapsed = time.time() - t0
            rate = n_used / elapsed
            eta = (len(fold_ids) - n_used) / rate
            print(f"  {n_used}/{len(fold_ids)} cases  rows={sum(len(x) for x in parts):,}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    all_df = pd.concat(parts, ignore_index=True)
    feat_cols = [c for c in all_df.columns if c not in ("target", "sqi", "case_id", "time_sec")]
    all_df[feat_cols] = all_df[feat_cols].astype(np.float32)
    all_df.to_parquet(out_parquet, index=False, compression="zstd")
    print(f"\nWrote {out_parquet.name}: {len(all_df):,} rows × {len(all_df.columns)} cols, "
          f"{out_parquet.stat().st_size/1e6:.1f} MB, {n_used} cases, {n_skip} skipped, "
          f"total {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
