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
from openeeg import openibis, openbsr, emg_correct

# eeg: 1-D numpy array, raw EEG in microvolts, sampled at 128 Hz
bis = openibis(eeg)                       # Connor 2023 paper-faithful
bis = openibis(eeg, deep="ellerkmann")    # paper + Ellerkmann 2004 deep-regime fit
bis = openibis(eeg, bsr="quazi")          # pre-2023 QUAZI BSR detector
bsr_pct = openbsr(eeg)                    # Connor 2025 OpenBSR (frequency-domain)

# Optional EMG-aware post-correction (needs the BIS/EMG track in dB)
bis = emg_correct(bis, emg_track)         # subtracts 0.54·max(EMG−34,0)
```

All outputs are at **2 Hz** (one value per 0.5 s epoch). Downsample by 2 to align with BIS Vista's 1 Hz output.

## What's implemented

| Function | Reference | Status |
|---|---|---|
| `openibis(deep="paper")` | Connor 2023 (A&A) | Paper-faithful (Table 1 verified) |
| `openibis(deep="ellerkmann")` | Connor 2023 + Ellerkmann 2004 deep-regime BSR fit | Implemented |
| `openibis(bsr="quazi")` | Pre-2023 BIS-convention burst-suppression detector | Implemented |
| `openbsr()` | Connor 2025 OpenBSR | Best-effort (prose-based; Table 1 was a raster image) |
| `emg_correct()` | Lee 2019 EMG threshold + this repo's 100-case fit | Post-correction; reduces awake (BIS 78–98) MAE by ~28% |

## Cohort baseline (val fold, N=100 VitalDB cases, SQI ≥ 80)

| Variant | MAE mean | r mean | 78-98 MAE | 61-78 MAE |
|---|---|---|---|---|
| `openibis(bsr="paper")` | 6.82 | 0.786 | 10.63 | 7.42 |
| `openibis(bsr="quazi")` | 6.31 | 0.795 | 11.51 | 8.40 |
| `openibis(bsr="quazi")` + `emg_correct()` | **6.11** | 0.785 | **8.26** | **6.55** |

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
