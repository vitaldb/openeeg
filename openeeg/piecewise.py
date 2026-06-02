"""openeeg.piecewise — interpretable, rule-gated piecewise-linear BIS predictor.

``predict_bis_rules(eeg)`` takes a raw 128 Hz EEG channel and returns a
1 Hz BIS-mimic score using a two-stage architecture:

  STAGE 1 — Deep-regime rule (Ellerkmann 2004):
    If ``openbsr(eeg) > 49.8`` (Lee 2019's deep gate threshold), the
    output is given by ``BIS = clip(44.1 − openbsr/2.25, 0, 100)``.
    This is the only published deep-regime BIS formula in the
    literature; it reaches MAE 1.85 on the true-deep subset.

  STAGE 2 — Piecewise linear on the non-deep cohort:
    Elsewhere, the prediction comes from a 16-leaf shallow decision
    tree (``max_depth=4``) trained only on openbsr ≤ 49.8 train rows.
    Each leaf carries a six-term OLS using the within-leaf
    top-|corr| features.

  STAGE 3 — EMA(15 s) post-smoothing to match BIS Vista's most common
    smoothing setting on VitalDB.

Validation on the 80-case W = 15 sub-cohort of the val fold
(epoch-weighted, SQI ≥ 80, EMA(15 s) post-smooth)::

      MAE = 4.36   Pearson r = 0.860   Lin's rc = 0.846

      0-21   21-41   41-61   61-78   78-98
      3.25    4.21    4.21    6.19    6.71

The data-driven tree partition outperforms a literal port of the
Lee 2019 (BSR > 49.8 / EMG < 34.2 / SEF < 20.2 / RBR < −0.7) tree
on every non-deep regime — see ``scripts/30_per_region_linear.py``.
The root split is ``openibis_quazi_30s ≤ 54.00`` (NOT BSR), which is
consistent with Connor 2023's observation that the openibis mixer
output already encodes BSR/SEF/EMG/BetaRatio internally.

Unlike ``predict_bis()`` (LightGBM, opaque), every prediction here can
be traced to either (a) a single Ellerkmann formula on openbsr, or
(b) a 16-leaf decision path with a 6-term linear formula. The
serialised model is < 15 kB JSON. Recommended for deployment when
interpretability and a tiny model footprint matter more than the
last ~0.7 BIS point of accuracy.

See ``scripts/30_per_region_linear.py`` for fit reproduction and
``results/piecewise_summary.csv`` for the full Lee-vs-data partition
comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from openeeg.openibis import openibis, bsr as _openibis_bsr, FS
from openeeg.openbsr import openbsr as _openbsr
from openeeg.features import (
    sef, bcsef, beta_ratio, emg_estimate, band_power, spectral_entropy,
)
from openeeg.postprocess import apply_ellerkmann_and_smooth

_MODEL_PATH = Path(__file__).parent / "models" / "piecewise_raw_data.json"

_FEATURE_ORDER = (
    "openibis_paper", "openibis_quazi", "openibis_quazi_30s",
    "bsr_paper", "bsr_quazi",
    "sef95", "bcsef", "beta_ratio", "emg_proxy",
    "p_delta", "p_theta", "p_alpha", "p_beta", "p_lowgamma", "spectral_entropy",
    "openibis_quazi_5s", "openibis_quazi_10s", "openibis_quazi_60s",
    "openibis_quazi_dt", "openibis_quazi_30s_dt", "sef95_dt", "emg_proxy_dt",
    "openbsr",
)

_model_cache: Optional[Dict] = None


def _load_model() -> Dict:
    global _model_cache
    if _model_cache is None:
        if not _MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Bundled piecewise model missing: {_MODEL_PATH}. Reinstall the package."
            )
        with open(_MODEL_PATH) as f:
            j = json.load(f)
        # Pre-index the tree nodes and leaf coefficient table for O(1) lookup
        nodes = {int(n["node"]): n for n in j["tree_thresholds"]}
        leaves = {int(spec["leaf"]): spec for spec in j["leaves"]}
        feat_idx = {name: i for i, name in enumerate(j["feature_cols"])}
        deep_rule = j.get("deep_rule", dict(
            feature="openbsr", threshold=49.8,
            formula="BIS = 44.1 - openbsr/2.25"))
        _model_cache = dict(
            feature_cols=tuple(j["feature_cols"]),
            nodes=nodes, leaves=leaves, feat_idx=feat_idx,
            deep_rule=deep_rule,
        )
    return _model_cache


def _rolling_mean_2hz(x: np.ndarray, window_s: float) -> np.ndarray:
    w = max(int(round(window_s * 2)), 1)
    kernel = np.ones(w) / w
    return np.convolve(np.where(np.isnan(x), 0.0, x), kernel, mode="full")[: len(x)]


def _diff(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    out[0] = 0.0
    out[1:] = x[1:] - x[:-1]
    return out


def _ema(x: np.ndarray, W: float) -> np.ndarray:
    alpha = 2.0 / (W + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out


def _extract_23(eeg: np.ndarray) -> np.ndarray:
    """Build the 23-column feature matrix at 2 Hz (matches the trained
    feature_cols order in piecewise_raw_data.json)."""
    pred_paper = openibis(eeg, bsr="paper", deep="paper")
    pred_quazi = openibis(eeg, bsr="quazi", deep="paper")
    pred_quazi_5s = _rolling_mean_2hz(pred_quazi, 5.0)
    pred_quazi_10s = _rolling_mean_2hz(pred_quazi, 10.0)
    pred_quazi_30s = _rolling_mean_2hz(pred_quazi, 30.0)
    pred_quazi_60s = _rolling_mean_2hz(pred_quazi, 60.0)
    bsr_p = _openibis_bsr(eeg, kind="paper")
    bsr_q = _openibis_bsr(eeg, kind="quazi")
    sef_v = sef(eeg)
    bcsef_v = bcsef(eeg)
    br = beta_ratio(eeg)
    emg = emg_estimate(eeg)
    p_delta = band_power(eeg, band=(0.5, 4.0))
    p_theta = band_power(eeg, band=(4.0, 8.0))
    p_alpha = band_power(eeg, band=(8.0, 13.0))
    p_beta = band_power(eeg, band=(13.0, 30.0))
    p_lowgamma = band_power(eeg, band=(30.0, 47.0))
    se = spectral_entropy(eeg)
    pred_quazi_dt = _diff(pred_quazi)
    pred_quazi_30s_dt = _diff(pred_quazi_30s)
    sef_dt = _diff(sef_v)
    emg_dt = _diff(emg)
    ob = _openbsr(eeg)  # ~2 Hz cadence (matches features)

    cols = [
        pred_paper, pred_quazi, pred_quazi_30s,
        bsr_p, bsr_q,
        sef_v, bcsef_v, br, emg,
        p_delta, p_theta, p_alpha, p_beta, p_lowgamma, se,
        pred_quazi_5s, pred_quazi_10s, pred_quazi_60s,
        pred_quazi_dt, pred_quazi_30s_dt, sef_dt, emg_dt,
        ob,
    ]
    n = min(len(c) for c in cols)
    return np.column_stack([c[:n] for c in cols])


def _route_to_leaves(X: np.ndarray, nodes: Dict[int, dict],
                     feat_idx: Dict[str, int]) -> np.ndarray:
    """Vectorised sklearn-tree routing — returns leaf id per row."""
    cur = np.zeros(len(X), dtype=int)  # all start at node 0
    done = np.zeros(len(X), dtype=bool)
    leaf_id = np.full(len(X), -1, dtype=int)
    # Walk until everyone reaches a leaf (node with feature == None)
    max_iter = 64  # generous
    for _ in range(max_iter):
        if done.all():
            break
        unique_nodes, inv = np.unique(cur[~done], return_inverse=True)
        # For each unique current node, dispatch
        for n in unique_nodes:
            spec = nodes[int(n)]
            mask = (cur == int(n)) & (~done)
            if spec["feature"] is None:
                # Leaf
                leaf_id[mask] = int(n)
                done[mask] = True
                continue
            fi = feat_idx[spec["feature"]]
            thr = float(spec["threshold"])
            # NaN-safe: treat NaN as "go left" (sklearn default with missing=fill 0)
            vals = X[mask, fi]
            go_left = ~(vals > thr)  # NaN > thr is False → NaN goes left
            new_cur = np.where(go_left, spec["left"], spec["right"])
            cur[mask] = new_cur.astype(int)
    return leaf_id


def _apply_leaves(X: np.ndarray, leaf_id: np.ndarray, leaves: Dict[int, dict],
                  feat_idx: Dict[str, int]) -> np.ndarray:
    pred = np.full(len(X), np.nan, dtype=float)
    for lid, spec in leaves.items():
        m = leaf_id == int(lid)
        if not m.any():
            continue
        a = float(spec["intercept"])
        if not spec["features"]:
            pred[m] = a
            continue
        idxs = [feat_idx[f] for f in spec["features"]]
        coefs = np.asarray(spec["coefs"], dtype=float)
        contrib = X[m][:, idxs] @ coefs
        pred[m] = a + contrib
    return pred


def _raw_piecewise(eeg, fs: int = 128) -> np.ndarray:
    """Raw 16-leaf piecewise-linear BIS prediction at 1 Hz (no wrapper).

    Used internally by :func:`openeeg.predict_bis` with
    ``method="piecewise"``. See ``openeeg.piecewise``'s module docstring
    for the validation MAE and architecture details.
    """
    if fs != FS:
        raise ValueError(f"piecewise method requires fs=128; got {fs}.")
    eeg = np.asarray(eeg, dtype=float)
    X_2hz = _extract_23(eeg)
    X_1hz = X_2hz[::2]
    if len(X_1hz) == 0:
        return np.empty(0)
    model = _load_model()
    X_safe = np.where(np.isnan(X_1hz), 0.0, X_1hz)
    leaf_id = _route_to_leaves(X_safe, model["nodes"], model["feat_idx"])
    return _apply_leaves(X_safe, leaf_id, model["leaves"], model["feat_idx"])


