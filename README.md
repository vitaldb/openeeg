# openeeg

Open-source EEG processing for depth-of-anesthesia — paper-faithful reimplementations of published BIS-mimic algorithms, validated against VitalDB.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Status

**Phase 0 — early alpha.** Functional API for `openibis` and `openbsr`.

## Install

```bash
pip install -e .[vitaldb,plot,dev]
```

## Quick start

```python
import numpy as np
from openeeg import openibis, openbsr

# eeg: 1-D numpy array, raw EEG in microvolts, sampled at 128 Hz
bis = openibis(eeg)                       # Connor 2023 paper-faithful
bis = openibis(eeg, deep="ellerkmann")   # paper + Ellerkmann 2004 deep-regime fit
bsr_pct = openbsr(eeg)                    # Connor 2025 OpenBSR (frequency-domain)
```

All outputs are at **2 Hz** (one value per 0.5 s epoch). Downsample by 2 to align with BIS Vista's 1 Hz output.

## What's implemented

| Function | Reference | Status |
|---|---|---|
| `openibis(deep="paper")` | Connor 2023 (A&A) | Paper-faithful (Table 1 verified) |
| `openibis(deep="ellerkmann")` | Connor 2023 + Ellerkmann 2004 deep-regime BSR fit | Implemented |
| `openbsr()` | Connor 2025 OpenBSR | Best-effort (prose-based; Table 1 was a raster image) |

## Validation

`scripts/01_validate_1vital.py` runs all variants against the bundled `1.vital` case and reports MAE, Pearson r, and Lin's concordance vs the BIS Vista's `BIS/BIS` and `BIS/SR` tracks (filtered to SQI ≥ 80).

## License

Apache-2.0. Algorithm implementations follow the *described methodology* in the cited papers; no proprietary code is included.
