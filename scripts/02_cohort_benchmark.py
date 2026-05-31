"""Cohort benchmark — run openibis/openbsr variants across a VitalDB BIS cohort.

Default: 10-case smoke test from the val split (caseid % 10 == 8).
Pass --n to scale, --fold {train,val,test} to choose split, --cache to
cache .vital files locally.

Outputs:
  * results/02_cohort_<fold>_n<N>.json — per-case metrics for every
    openibis (bsr × deep) and openbsr variant.
  * results/02_cohort_<fold>_n<N>_summary.txt — aggregated table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg import openibis, openbsr
from openeeg.openibis import bsr as openibis_bsr
from openeeg.cohort import caseids_bis, split, load_case, preprocess_eeg
from openeeg.metrics import evaluate, LEE_BIN_LABELS

SQI_THRESH = 80


def run_one(case: dict) -> dict:
    """Run all variants on a single case, returning a metrics dict."""
    eeg = preprocess_eeg(case["eeg"])
    bis_actual = case["bis"]
    sqi = case["sqi"]
    sr_actual = case["sr"]

    variants = {}
    for bsr_kind in ("paper", "quazi"):
        for deep in ("paper", "ellerkmann"):
            variants[f"openibis_{bsr_kind}_{deep}"] = openibis(eeg, bsr=bsr_kind, deep=deep)
    bsr_variants = {
        "bsr_paper": openibis_bsr(eeg, kind="paper"),
        "bsr_quazi": openibis_bsr(eeg, kind="quazi"),
        "openbsr_2025": openbsr(eeg),
    }

    # Align to 1 Hz
    pred_bis = {k: v[::2] for k, v in variants.items()}
    pred_bsr = {k: v[::2] for k, v in bsr_variants.items()}

    n = min(len(bis_actual), *[len(v) for v in pred_bis.values()],
            len(sqi), len(sr_actual))
    bis_actual = bis_actual[:n]
    sqi = sqi[:n]
    sr_actual = sr_actual[:n]
    pred_bis = {k: v[:n] for k, v in pred_bis.items()}
    pred_bsr = {k: v[:n] for k, v in pred_bsr.items()}

    valid_bis = (
        ~np.isnan(bis_actual)
        & ~np.isnan(sqi)
        & (sqi >= SQI_THRESH)
        & np.all([~np.isnan(v) for v in pred_bis.values()], axis=0)
    )
    valid_bsr = (
        ~np.isnan(sr_actual)
        & ~np.isnan(sqi)
        & (sqi >= SQI_THRESH)
        & np.all([~np.isnan(v) for v in pred_bsr.values()], axis=0)
    )

    return {
        "caseid": case["caseid"],
        "n_seconds": int(n),
        "n_valid_bis": int(valid_bis.sum()),
        "n_valid_bsr": int(valid_bsr.sum()),
        "bis_variants": {k: evaluate(bis_actual, v, valid_bis) for k, v in pred_bis.items()},
        "bsr_variants": {k: evaluate(sr_actual, v, valid_bsr) for k, v in pred_bsr.items()},
    }


def aggregate(results: list[dict]) -> dict:
    """Mean ± std (and median) across cases for each variant."""
    if not results:
        return {}
    bis_names = list(results[0]["bis_variants"].keys())
    bsr_names = list(results[0]["bsr_variants"].keys())

    def collect_global(variant_group: str, name: str, key: str) -> np.ndarray:
        vals = []
        for r in results:
            v = r[variant_group][name]["global"].get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(v)
        return np.array(vals)

    def collect_regime_mae(variant_group: str, name: str, label: str) -> np.ndarray:
        vals = []
        for r in results:
            v = r[variant_group][name]["per_regime"][label]["mae"]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(v)
        return np.array(vals)

    out: dict = {"bis": {}, "bsr": {}}
    for name in bis_names:
        mae = collect_global("bis_variants", name, "mae")
        r_ = collect_global("bis_variants", name, "r")
        rc = collect_global("bis_variants", name, "lin_rc")
        out["bis"][name] = {
            "n_cases": len(mae),
            "mae_mean": float(mae.mean()) if len(mae) else float("nan"),
            "mae_median": float(np.median(mae)) if len(mae) else float("nan"),
            "r_mean": float(r_.mean()) if len(r_) else float("nan"),
            "r_median": float(np.median(r_)) if len(r_) else float("nan"),
            "lin_rc_mean": float(rc.mean()) if len(rc) else float("nan"),
            "per_regime_mae_mean": {
                label: float(v.mean()) if len(v := collect_regime_mae("bis_variants", name, label)) else float("nan")
                for label in LEE_BIN_LABELS
            },
        }
    for name in bsr_names:
        mae = collect_global("bsr_variants", name, "mae")
        r_ = collect_global("bsr_variants", name, "r")
        rc = collect_global("bsr_variants", name, "lin_rc")
        out["bsr"][name] = {
            "n_cases": len(mae),
            "mae_mean": float(mae.mean()) if len(mae) else float("nan"),
            "mae_median": float(np.median(mae)) if len(mae) else float("nan"),
            "r_mean": float(r_.mean()) if len(r_) else float("nan"),
            "r_median": float(np.median(r_)) if len(r_) else float("nan"),
            "lin_rc_mean": float(rc.mean()) if len(rc) else float("nan"),
        }
    return out


def render_summary(agg: dict) -> str:
    lines = []
    lines.append(f"\n=== BIS variants (per-case mean ± median, N_cases) ===")
    lines.append(f"{'variant':<32s}  {'MAE_mean':>9s}  {'MAE_med':>8s}  {'r_mean':>7s}  {'rc_mean':>8s}  {'N':>4s}")
    for name, s in agg["bis"].items():
        lines.append(
            f"{name:<32s}  {s['mae_mean']:9.2f}  {s['mae_median']:8.2f}  "
            f"{s['r_mean']:7.3f}  {s['lin_rc_mean']:8.3f}  {s['n_cases']:>4d}"
        )
    lines.append(f"\n=== Per-regime MAE (mean across cases) ===")
    header = f"{'variant':<32s}  " + "  ".join(f"{lbl:>7s}" for lbl in LEE_BIN_LABELS)
    lines.append(header)
    for name, s in agg["bis"].items():
        row = f"{name:<32s}  " + "  ".join(
            f"{s['per_regime_mae_mean'][lbl]:7.2f}" for lbl in LEE_BIN_LABELS
        )
        lines.append(row)
    lines.append(f"\n=== BSR variants ===")
    lines.append(f"{'variant':<32s}  {'MAE_mean':>9s}  {'MAE_med':>8s}  {'r_mean':>7s}  {'rc_mean':>8s}  {'N':>4s}")
    for name, s in agg["bsr"].items():
        lines.append(
            f"{name:<32s}  {s['mae_mean']:9.2f}  {s['mae_median']:8.2f}  "
            f"{s['r_mean']:7.3f}  {s['lin_rc_mean']:8.3f}  {s['n_cases']:>4d}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=10, help="Number of cases to process.")
    ap.add_argument("--cache", default=None, help="Local cache directory for .vital files.")
    ap.add_argument("--offset", type=int, default=0, help="Skip the first N cases in the fold.")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    results_dir = repo / "results"
    results_dir.mkdir(exist_ok=True)
    out_json = results_dir / f"02_cohort_{args.fold}_n{args.n}.json"
    out_txt = results_dir / f"02_cohort_{args.fold}_n{args.n}_summary.txt"

    all_ids = caseids_bis()
    fold_ids = split(all_ids, fold=args.fold)
    fold_ids = fold_ids[args.offset : args.offset + args.n]

    cache_dir = Path(args.cache) if args.cache else None
    print(f"Cohort: {len(all_ids)} total BIS cases  →  {len(fold_ids)} in fold={args.fold}[{args.offset}:{args.offset+args.n}]")
    if cache_dir:
        print(f"Cache: {cache_dir}")

    results = []
    t0 = time.time()
    for i, cid in enumerate(fold_ids, 1):
        ts = time.time()
        case = load_case(cid, cache_dir=cache_dir)
        if case is None:
            print(f"  [{i:3d}/{len(fold_ids)}] case {cid}: SKIP (load failed)")
            continue
        try:
            r = run_one(case)
        except Exception as exc:
            print(f"  [{i:3d}/{len(fold_ids)}] case {cid}: SKIP ({exc!r})")
            continue
        results.append(r)
        elapsed = time.time() - ts
        # Pick the most-promising variant to report inline.
        s = r["bis_variants"]["openibis_quazi_ellerkmann"]["global"]
        print(f"  [{i:3d}/{len(fold_ids)}] case {cid}: {elapsed:5.1f}s  "
              f"n={r['n_valid_bis']:5d}  MAE={s['mae']:5.2f}  r={s['r']:.3f}")

    print(f"\nTotal: {len(results)} cases in {(time.time()-t0)/60:.1f} min")

    agg = aggregate(results)
    out_json.write_text(json.dumps(
        {"cohort": args.fold, "n_requested": args.n, "n_successful": len(results),
         "per_case": results, "aggregate": agg}, indent=2))
    txt = render_summary(agg)
    out_txt.write_text(txt)
    print(f"\nSaved: {out_json.name}, {out_txt.name}")
    print(txt)


if __name__ == "__main__":
    main()
