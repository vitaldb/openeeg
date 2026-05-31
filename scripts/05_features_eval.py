"""Phase 2C + 2D — cohort evaluation of new features.

Reports across the 100 cached val-fold cases:

  C1. **SEF95 / BetaRatio** sanity vs actual BIS (per-case Pearson r).
  C2. **Morimoto BcSEF model** ``BIS ≈ 2.3·BcSEF + 12`` — direct
      comparison vs actual BIS for BIS < 80 epochs (the validated
      range), and vs openibis on the full range.
  D1. **emg_estimate vs BIS/EMG** — correlation between our 47–63 Hz
      proxy and the BIS Vista's published EMG track.
  D2. **emg_correct using emg_estimate** — does the correction work
      without the external EMG track? Compare:
          openibis              (baseline)
        + emg_correct(BIS/EMG)  (oracle EMG)
        + emg_correct(emg_estimate)  (proxy EMG)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg import openibis, emg_correct, sef, bcsef, beta_ratio, emg_estimate
from openeeg.cohort import load_case, preprocess_eeg
from openeeg.metrics import LEE_BIN_LABELS, LEE_BINS

CACHE = Path("C:/temp/openeeg_cache")
SQI = 80


def per_regime(actual: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict:
    a, p = actual[valid], pred[valid]
    out = {}
    for lbl, lo, hi in zip(LEE_BIN_LABELS, LEE_BINS[:-1], LEE_BINS[1:]):
        m = (a >= lo) & (a < hi)
        out[lbl] = float(np.mean(np.abs(p[m] - a[m]))) if m.sum() > 10 else float("nan")
    return out


def main():
    caseids = sorted(int(p.stem) for p in CACHE.glob("*.vital"))
    print(f"Loading {len(caseids)} cached cases...")

    # Aggregations
    per_case_r_sef = []         # SEF95 correlation with actual BIS
    per_case_r_betaratio = []   # BetaRatio correlation
    per_case_r_emgproxy = []    # emg_estimate correlation with BIS/EMG
    morimoto_resid = []         # residual of Morimoto's BcSEF model

    # MAE arrays for openibis ± emg correction variants
    mae_baseline = []
    mae_oracle_emg = []
    mae_proxy_emg = []

    # Per-regime collection
    regime_baseline = {lbl: [] for lbl in LEE_BIN_LABELS}
    regime_oracle  = {lbl: [] for lbl in LEE_BIN_LABELS}
    regime_proxy   = {lbl: [] for lbl in LEE_BIN_LABELS}

    n_used = 0
    for cid in caseids:
        case = load_case(cid, cache_dir=CACHE)
        if case is None:
            continue
        eeg = preprocess_eeg(case["eeg"])
        bis_actual = case["bis"]
        sqi = case["sqi"]
        emg_actual = case["emg"]

        # Compute everything once
        pred_baseline = openibis(eeg, bsr="quazi", deep="paper")[::2]
        sef_v = sef(eeg)[::2]
        bcsef_v = bcsef(eeg)[::2]
        br_v = beta_ratio(eeg)[::2]
        emg_v = emg_estimate(eeg)[::2]

        n = min(len(bis_actual), len(pred_baseline), len(sqi), len(emg_actual),
                len(sef_v), len(bcsef_v), len(br_v), len(emg_v))
        if n < 60:
            continue
        a = bis_actual[:n]
        s = sqi[:n]
        e_actual = emg_actual[:n]
        pred = pred_baseline[:n]
        sef_v = sef_v[:n]; bcsef_v = bcsef_v[:n]; br_v = br_v[:n]; emg_v = emg_v[:n]

        v = (~np.isnan(a) & ~np.isnan(s) & (s >= SQI)
             & ~np.isnan(pred) & ~np.isnan(e_actual) & ~np.isnan(emg_v))
        if v.sum() < 100:
            continue
        n_used += 1

        # --- C1: standalone feature correlations with actual BIS ---
        for arr, store in [(sef_v, per_case_r_sef), (br_v, per_case_r_betaratio)]:
            sub = arr[v]
            if sub.std() > 0:
                r = float(np.corrcoef(sub, a[v])[0, 1])
                if np.isfinite(r):
                    store.append(r)

        # --- C2: Morimoto BcSEF model — applies for BIS < 80 ---
        morimoto = 2.3 * bcsef_v + 12.0
        m80 = v & (a < 80)
        if m80.sum() > 30:
            morimoto_resid.append(float(np.mean(np.abs(morimoto[m80] - a[m80]))))

        # --- D1: emg_estimate vs BIS/EMG ---
        if emg_v[v].std() > 0 and e_actual[v].std() > 0:
            r = float(np.corrcoef(emg_v[v], e_actual[v])[0, 1])
            if np.isfinite(r):
                per_case_r_emgproxy.append(r)

        # --- D2: emg_correct using each EMG source ---
        # Need to fit the slope from the proxy because its scale differs
        # from BIS/EMG (dB). Use default slope/threshold for both.
        oracle = emg_correct(pred, e_actual)
        proxy = emg_correct(pred, emg_v)
        for tag, p_arr, mae_list, reg in (
            ("baseline", pred, mae_baseline, regime_baseline),
            ("oracle",   oracle, mae_oracle_emg, regime_oracle),
            ("proxy",    proxy, mae_proxy_emg, regime_proxy),
        ):
            mae_list.append(float(np.mean(np.abs(p_arr[v] - a[v]))))
            reg_d = per_regime(a, p_arr, v)
            for k in LEE_BIN_LABELS:
                if not np.isnan(reg_d[k]):
                    reg[k].append(reg_d[k])

    print(f"  {n_used} cases used\n")

    def stat(arr, name):
        if len(arr) == 0:
            print(f"  {name:<28s}  no data")
            return
        arr = np.array(arr)
        print(f"  {name:<28s}  mean={arr.mean():+.3f}  median={np.median(arr):+.3f}  N={len(arr)}")

    print("=== C1. Standalone feature correlation with actual BIS (per case) ===")
    stat(per_case_r_sef, "SEF95 vs BIS")
    stat(per_case_r_betaratio, "BetaRatio vs BIS")

    print("\n=== C2. Morimoto BcSEF model (BIS ≈ 2.3·BcSEF + 12, BIS<80) ===")
    if morimoto_resid:
        arr = np.array(morimoto_resid)
        print(f"  per-case MAE  mean={arr.mean():.2f}  median={np.median(arr):.2f}  N={len(arr)}")

    print("\n=== D1. emg_estimate(47-63Hz dB) vs BIS/EMG ===")
    stat(per_case_r_emgproxy, "Pearson r per case")

    print("\n=== D2. EMG correction with proxy vs oracle EMG ===")
    base = np.array(mae_baseline)
    orc = np.array(mae_oracle_emg)
    prx = np.array(mae_proxy_emg)
    print(f"  baseline                      mean MAE = {base.mean():.2f}")
    print(f"  + emg_correct(BIS/EMG)        mean MAE = {orc.mean():.2f}  (Δ={orc.mean()-base.mean():+.2f})")
    print(f"  + emg_correct(emg_estimate)   mean MAE = {prx.mean():.2f}  (Δ={prx.mean()-base.mean():+.2f})")

    print("\n  Per-regime MAE (mean across cases):")
    print(f"  {'regime':>7s}  {'baseline':>9s}  {'oracle':>8s}  {'proxy':>8s}")
    for lbl in LEE_BIN_LABELS:
        b = np.mean(regime_baseline[lbl]) if regime_baseline[lbl] else float("nan")
        o = np.mean(regime_oracle[lbl])   if regime_oracle[lbl]   else float("nan")
        p = np.mean(regime_proxy[lbl])    if regime_proxy[lbl]    else float("nan")
        print(f"  {lbl:>7s}  {b:9.2f}  {o:8.2f}  {p:8.2f}")


if __name__ == "__main__":
    main()
