"""openeeg — open-source EEG processing for depth-of-anesthesia.

Top-level entry points::

    >>> from openeeg import openibis, openbsr
    >>> bis = openibis(eeg, fs=128)            # BIS-mimic, paper-faithful
    >>> bis = openibis(eeg, fs=128, deep="ellerkmann")  # Ellerkmann deep-regime
    >>> bsr_pct = openbsr(eeg, fs=128)          # frequency-domain BSR

Implementations follow the methods described in the cited papers; no
proprietary algorithm code is included. See each function's docstring
for the canonical reference.
"""
from openeeg.openibis import openibis
from openeeg.openbsr import openbsr

__version__ = "0.0.1"

__all__ = ["openibis", "openbsr", "__version__"]
