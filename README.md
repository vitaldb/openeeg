# openeeg

Open-source EEG processing for depth-of-anesthesia — paper-faithful reimplementations of published BIS-mimic algorithms, validated against VitalDB.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Status

**Phase 0 — early alpha.** Functional API for `openibis` and `openbsr`.

## Install

```bash
# from PyPI (distribution name is vitaldb-openeeg; module name is openeeg)
pip install vitaldb-openeeg

# from source
pip install -e .[vitaldb,plot,dev]
```

The distribution name `vitaldb-openeeg` is a short-term placeholder
while [PEP 541 takeover](docs/pep541-openeeg-takeover.md) of the
abandoned `openeeg` name proceeds. The Python import name is always
`openeeg`, so user code does not need to change.

## Quick start

```python
import numpy as np
from openeeg import predict_bis, openibis, openbsr, emg_correct

# eeg: 1-D numpy array, raw EEG in microvolts, sampled at 128 Hz

# === Trained model (recommended) — needs vitaldb-openeeg[predict] ===
bis = predict_bis(eeg)                    # 1 Hz output, LightGBM trained
                                          # on 498 VitalDB BIS cases

# === Paper-faithful algorithms (no training, fast) ===
bis = openibis(eeg)                       # Connor 2023 paper-faithful (2 Hz)
bis = openibis(eeg, deep="ellerkmann")    # paper + Ellerkmann 2004 deep-regime fit
bis = openibis(eeg, bsr="quazi")          # pre-2023 QUAZI BSR detector
bsr_pct = openbsr(eeg)                    # Connor 2024 OpenBSR (frequency-domain)

# Optional EMG-aware post-correction (needs the BIS/EMG track in dB)
bis = emg_correct(bis, emg_track)         # subtracts 0.54·max(EMG−34,0)
```

`predict_bis` returns at **1 Hz** (matches BIS Vista output rate).
The paper-faithful `openibis` / `openbsr` return at **2 Hz** (per
0.5 s epoch); downsample by 2 to align with `predict_bis`.

## What's implemented

| Function | Reference | Status |
|---|---|---|
| `predict_bis()` | LightGBM regressor over 15 spectral features (this repo) | Trained on 498 VitalDB BIS cases; bundled model ships in the wheel |
| `openibis(deep="paper")` | Connor 2023 (A&A) | Paper-faithful (Table 1 verified) |
| `openibis(deep="ellerkmann")` | Connor 2023 + Ellerkmann 2004 deep-regime BSR fit | Implemented |
| `openibis(bsr="quazi")` | Pre-2023 BIS-convention burst-suppression detector | Implemented |
| `openbsr()` | Connor 2024 OpenBSR | Best-effort (prose-based; Table 1 was a raster image) |
| `emg_correct()` | Lee 2019 EMG threshold + this repo's 100-case fit | Post-correction; reduces awake (BIS 78–98) MAE by ~28% |
| `sef()` | Spectral Edge Frequency at p% in a band | Standalone feature |
| `bcsef()` | Burst-compensated SEF95 (Morimoto 2004) | Standalone feature |
| `beta_ratio()` | log10(P_30-47 / P_11-20) (Noh 2017; Lee 2019) | Standalone feature |
| `band_power()` | Mean dB power in any frequency band | Standalone feature |
| `spectral_entropy()` | Shannon entropy of the normalised PSD | Standalone feature |
| `emg_estimate()` | 47–63 Hz dB band power | Feature only — **not** a `BIS/EMG` replacement (r ≈ 0.32 vs real EMG on 100 cases) |

## Cohort baseline (val fold, N=100 VitalDB cases, SQI ≥ 80)

Per-case mean (Phase 0–2):

| Variant | MAE mean | r mean | 78-98 MAE | 61-78 MAE |
|---|---|---|---|---|
| `openibis(bsr="paper")` | 6.82 | 0.786 | 10.63 | 7.42 |
| `openibis(bsr="quazi")` | 6.31 | 0.795 | 11.51 | 8.40 |
| `openibis(bsr="quazi")` + `emg_correct()` | 6.11 | 0.785 | 8.26 | 6.55 |

Epoch-weighted, full val parquet (784,550 epochs):

| Variant | MAE | r | Lin's rc | 0-21 | 21-41 | 41-61 | 61-78 | 78-98 |
|---|---|---|---|---|---|---|---|---|
| `openibis(quazi, paper)` baseline | 5.90 | 0.764 | 0.756 | 3.77 | 5.91 | 5.68 | 6.31 | 10.73 |
| **`predict_bis()` (15 features, 498 train cases)** | **4.25** | **0.850** | **0.844** | 15.16 | 4.14 | 4.18 | 4.56 | 6.06 |
| `predict_bis` + Vista oracle inputs (research only) | 3.73 | 0.891 | 0.888 | 1.75 | 3.58 | 3.81 | 3.99 | 5.79 |

The bundled `predict_bis()` model uses only raw-EEG-derived features
(no `BIS/EMG`, `BIS/SR`, `BIS/SEF`, `BIS/TOTPOW`). It can therefore
run on any 128 Hz EEG channel, not just BIS-sensor data. The
oracle-augmented variant (Phase 3d/3e) is documented in
`scripts/07_train_lightgbm.py` and `scripts/10_hybrid_rule_lgbm.py`
for research reproducibility but is not deployed.

The deep-regime (BIS 0–21) is still a known weak spot — about
3,000 of 784,550 val epochs, dominated by a single hard case.
See `scripts/08_rule_analysis.py` for a Lee-2019-style rule
analysis of where the model misses.

## Validation

`scripts/01_validate_1vital.py` fetches VitalDB case 1 on demand and
runs all variants, reporting MAE, Pearson r, and Lin's concordance
vs the BIS Vista's `BIS/BIS` and `BIS/SR` tracks (filtered to SQI ≥ 80).

`scripts/02_cohort_benchmark.py --fold val --n 100` runs the same
grid across a 100-case validation cohort.

## Data

The VitalDB Open Dataset is licensed under **CC-BY-NC 4.0**. The
scripts in this repository download cases on demand via the `vitaldb`
Python package; no .vital files are bundled in the repository. By
using these scripts you agree to the
[VitalDB dataset terms](https://vitaldb.net/dataset/).

## License

Apache-2.0 for the source code in `openeeg/`, `scripts/`, and `tests/`.
Algorithm implementations follow the *described methodology* in the
cited papers; no proprietary code is included.
