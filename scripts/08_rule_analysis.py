"""Phase 3d exploration — Lee 2019 single-variable gating rules.

Lee 2019 (VitalDB, N=5,427) describes a decision tree:

   BSR > 49.8%                                    -> 0-21
   else if EMG < 34.2 dB AND SEF95 < 20.2 Hz      -> 21-61
        BSR > 2.1% OR SEF95 < 14.8 Hz             -> 21-41
        else                                       -> 41-61
   else                                            -> 61-98
        RBR < -0.7                                 -> 61-78
        else                                       -> 78-98

For each gating rule we report on the val parquet:
  * sensitivity  = P(rule fires | actual bin)
  * precision    = P(actual bin | rule fires)
  * the empirically-best single-variable threshold by F1 against
    the bin, alongside Lee's published threshold

Then we look at the cases where LightGBM's prediction lands far
from the actual bin and ask which single rule, if applied as a
*hard* override on top of LightGBM, would fix them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VAL_PARQUET = Path(__file__).resolve().parents[1] / "results" / "features_val_n100_v2.parquet"
TRAIN_PARQUET = Path(__file__).resolve().parents[1] / "results" / "features_train_n500_v2.parquet"


def report_rule(name: str, rule_mask: np.ndarray, actual_bin_mask: np.ndarray):
    tp = int((rule_mask & actual_bin_mask).sum())
    fp = int((rule_mask & ~actual_bin_mask).sum())
    fn = int((~rule_mask & actual_bin_mask).sum())
    tn = int((~rule_mask & ~actual_bin_mask).sum())
    sens = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * sens * prec / max(sens + prec, 1e-9)
    print(f"  {name}")
    print(f"    rule fires: {int(rule_mask.sum()):,}   actual in bin: {int(actual_bin_mask.sum()):,}")
    print(f"    sens={sens*100:5.1f}%   prec={prec*100:5.1f}%   F1={f1:.3f}")
    print(f"    TP={tp:6,d}  FP={fp:6,d}  FN={fn:6,d}")


def sweep_threshold(feature: np.ndarray, in_bin: np.ndarray, *,
                    direction: str = ">", n_quantiles: int = 200) -> dict:
    """Find the single-variable threshold maximizing F1 vs in_bin."""
    valid = ~np.isnan(feature)
    f = feature[valid]
    in_b = in_bin[valid]
    candidates = np.unique(np.quantile(f, np.linspace(0, 1, n_quantiles)))
    best = {"thr": float("nan"), "f1": 0.0, "sens": 0.0, "prec": 0.0}
    for thr in candidates:
        if direction == ">":
            rule = f > thr
        else:
            rule = f < thr
        tp = (rule & in_b).sum()
        if tp == 0:
            continue
        sens = tp / max(in_b.sum(), 1)
        prec = tp / max(rule.sum(), 1)
        f1 = 2 * sens * prec / max(sens + prec, 1e-9)
        if f1 > best["f1"]:
            best = {"thr": float(thr), "f1": float(f1),
                    "sens": float(sens), "prec": float(prec)}
    return best


def main():
    print(f"Loading {VAL_PARQUET.name}...")
    df = pd.read_parquet(VAL_PARQUET)
    print(f"  {len(df):,} rows × {len(df.columns)} cols")

    a = df["target"].values
    # Lee 2019 bins
    deep   = (a >= 0)  & (a < 21)
    light  = (a >= 21) & (a < 41)
    surg   = (a >= 41) & (a < 61)
    trans  = (a >= 61) & (a < 78)
    awake  = (a >= 78) & (a < 98)
    print(f"  Lee bins:  0-21:{int(deep.sum()):,}   21-41:{int(light.sum()):,}   "
          f"41-61:{int(surg.sum()):,}   61-78:{int(trans.sum()):,}   78-98:{int(awake.sum()):,}")

    bsr_p = df["bsr_paper"].values
    bsr_q = df["bsr_quazi"].values
    emg   = df["bis_emg_oracle"].values
    sef95_ours = df["sef95"].values
    br    = df["beta_ratio"].values
    p_beta  = df["p_beta"].values

    # Vista oracle equivalents (Lee 2019 actually uses these)
    bsr_vista = df["bis_sr_oracle"].values        # Lee's "BSR"
    sef_vista = df["bis_sef_oracle"].values       # Lee's "SEF95"

    print("\n=== (1) Lee 2019 published rules — three input variants ===")

    print("\n [A] 0-21 gate: BSR > 49.8%")
    for name, bsr_var in [
        ("BSR_quazi (raw EEG)", bsr_q),
        ("BSR_paper (raw EEG)", bsr_p),
        ("BIS/SR  (Vista oracle, Lee's actual input)", bsr_vista),
    ]:
        report_rule(name, bsr_var > 49.8, deep)

    print("\n [B] 21-61 gate: EMG < 34.2 AND SEF < 20.2")
    in_bin_B = (a >= 21) & (a < 61)
    for name, sef_var in [
        ("our SEF95 (raw EEG)", sef95_ours),
        ("BIS/SEF   (Vista oracle, Lee's actual input)", sef_vista),
    ]:
        rule_B = (emg < 34.2) & (sef_var < 20.2)
        report_rule(name, rule_B, in_bin_B)

    print("\n [C] 78-98 gate: NOT B  AND  RBR (beta_ratio) >= -0.7")
    rule_B_vista = (emg < 34.2) & (sef_vista < 20.2)
    rule_C = (~rule_B_vista) & (br >= -0.7)
    report_rule("with Vista SEF in B", rule_C, awake)

    print("\n=== (2) Best single-variable threshold per bin on our data ===")
    for name, in_bin, feats in [
        ("0-21 deep",    deep,  [("bsr_quazi", bsr_q, ">"), ("bsr_paper", bsr_p, ">"),
                                  ("sef95", sef95, "<"), ("p_beta", p_beta, "<")]),
        ("21-41 light",  light, [("bsr_quazi", bsr_q, ">"), ("sef95", sef95, "<"),
                                  ("emg", emg, "<"), ("beta_ratio", br, "<")]),
        ("78-98 awake",  awake, [("emg", emg, ">"), ("sef95", sef95, ">"),
                                  ("beta_ratio", br, ">"), ("bsr_quazi", bsr_q, "<")]),
    ]:
        print(f" {name}:")
        for fname, fval, direction in feats:
            best = sweep_threshold(fval, in_bin, direction=direction)
            sym = ">" if direction == ">" else "<"
            print(f"   {fname:<12s} {sym} {best['thr']:7.2f}  "
                  f"F1={best['f1']:.3f}  sens={best['sens']*100:5.1f}%  prec={best['prec']*100:5.1f}%")

    print("\n=== (3) Where LightGBM mispredicts deep (actual 0-21) ===")
    # Load same LightGBM the train pipeline uses
    try:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(Path(__file__).resolve().parents[1] / "results" / "lgbm.txt"))
        feature_cols = [c for c in df.columns if c not in ("target", "sqi", "case_id", "time_sec")]
        pred = np.clip(booster.predict(df[feature_cols].values), 0.0, 100.0)
    except Exception as exc:
        print(f"  (skipping — model load failed: {exc!r})")
        return

    miss = deep & (pred > 30)
    hit  = deep & (pred <= 30)
    print(f"  deep epochs total: {int(deep.sum()):,}")
    print(f"  LightGBM correctly says deep (pred<=30):       {int(hit.sum()):6,d}")
    print(f"  LightGBM misses (pred>30 while actual<21):     {int(miss.sum()):6,d}")
    if miss.sum() > 0:
        print(f"\n  Feature distribution of the MISSED deep epochs vs HIT deep epochs:")
        print(f"  {'feature':<14s}  {'missed_median':>14s}  {'missed_p75':>11s}  {'hit_median':>11s}  {'hit_p25':>8s}")
        for fname, fval in [("bsr_quazi", bsr_q), ("bsr_paper", bsr_p), ("emg", emg),
                             ("sef95", sef95), ("beta_ratio", br), ("p_beta", p_beta)]:
            mm = np.nanmedian(fval[miss])
            mp = np.nanpercentile(fval[miss], 75)
            hm = np.nanmedian(fval[hit])
            hp = np.nanpercentile(fval[hit], 25)
            print(f"  {fname:<14s}  {mm:>14.2f}  {mp:>11.2f}  {hm:>11.2f}  {hp:>8.2f}")

        miss_caseids = df["case_id"][miss].unique()
        print(f"\n  Cases contributing the missed deep epochs: {len(miss_caseids)}")
        for cid in miss_caseids[:10]:
            n_miss = int((miss & (df["case_id"].values == cid)).sum())
            n_deep = int((deep & (df["case_id"].values == cid)).sum())
            print(f"    case {cid}: {n_miss}/{n_deep} deep epochs missed")

    # Counterfactual: what if we hard-override LightGBM with Ellerkmann when bsr_quazi > 49.8?
    print(f"\n=== (4) Hard rule override: pred = ellerkmann when bsr_quazi > 49.8 ===")
    ellerkmann = 44.1 - bsr_q / 2.25
    gate = bsr_q > 49.8
    pred_rule = np.where(gate, np.clip(ellerkmann, 0, 100), pred)
    print(f"  gate fires {int(gate.sum()):,} epochs ({100*gate.mean():.1f}%)")
    for name, pp in [("LightGBM baseline", pred), ("LightGBM + Lee A gate", pred_rule)]:
        overall_mae = float(np.mean(np.abs(pp - a)))
        deep_mae = float(np.mean(np.abs(pp[deep] - a[deep])))
        awake_mae = float(np.mean(np.abs(pp[awake] - a[awake])))
        surg_mae = float(np.mean(np.abs(pp[surg] - a[surg])))
        print(f"  {name:<28s}  overall={overall_mae:.2f}  0-21={deep_mae:.2f}  "
              f"41-61={surg_mae:.2f}  78-98={awake_mae:.2f}")


if __name__ == "__main__":
    main()
