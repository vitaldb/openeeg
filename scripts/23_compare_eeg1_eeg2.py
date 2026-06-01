"""Compare BIS sensor channels EEG1 and EEG2 on a few cases.

Visualises raw EEG, computes openibis/BSR on each, and checks
whether averaging or differencing the two channels improves
agreement with the Vista BSR oracle.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vitaldb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openeeg import openibis, openbsr
from openeeg.openibis import bsr as openibis_bsr
from openeeg.cohort import preprocess_eeg, _resolve_local_vital

RESULTS = Path(__file__).resolve().parents[1] / "results"

FS = 128


def load_dual_eeg(caseid: int):
    """Load both EEG1 and EEG2 along with the auxiliary tracks."""
    p = _resolve_local_vital(caseid)
    if p is None:
        return None
    vf = vitaldb.VitalFile(str(p), ["BIS/EEG1_WAV", "BIS/EEG2_WAV", "BIS/BIS",
                                     "BIS/SQI", "BIS/SR", "BIS/EMG"])
    try:
        eeg1 = vf.to_numpy(["BIS/EEG1_WAV"], 1.0 / FS).flatten()
        eeg2 = vf.to_numpy(["BIS/EEG2_WAV"], 1.0 / FS).flatten()
    except Exception as exc:
        print(f"  case {caseid}: {exc}")
        return None
    rest = vf.to_numpy(["BIS/BIS", "BIS/SQI", "BIS/SR", "BIS/EMG"], 1.0)
    n = min(len(eeg1), len(eeg2))
    return {
        "caseid": caseid, "fs": FS,
        "eeg1": eeg1[:n], "eeg2": eeg2[:n],
        "bis":  rest[:, 0], "sqi": rest[:, 1],
        "sr":   rest[:, 2], "emg": rest[:, 3],
    }


def main():
    # Mix of cases: heavy-deep, transient burst, surgical-only
    case_ids = [1098, 988, 548, 838, 358, 818, 628]
    print(f"Loading {len(case_ids)} cases...")

    rows = []
    for cid in case_ids:
        case = load_dual_eeg(cid)
        if case is None:
            print(f"  case {cid}: skip")
            continue
        eeg1 = preprocess_eeg(case["eeg1"])
        eeg2 = preprocess_eeg(case["eeg2"])
        # Channel similarity
        n = min(len(eeg1), len(eeg2))
        valid = ~np.isnan(eeg1[:n]) & ~np.isnan(eeg2[:n])
        if valid.sum() < 1000:
            r = float("nan")
        else:
            r = float(np.corrcoef(eeg1[:n][valid], eeg2[:n][valid])[0, 1])
        # BSR from each + average
        bsr_q1 = openibis_bsr(eeg1, kind="quazi")[::2]
        bsr_q2 = openibis_bsr(eeg2, kind="quazi")[::2]
        bsr_p1 = openibis_bsr(eeg1, kind="paper")[::2]
        bsr_p2 = openibis_bsr(eeg2, kind="paper")[::2]
        ob1 = openbsr(eeg1)[::2]
        ob2 = openbsr(eeg2)[::2]
        n_align = min(len(bsr_q1), len(bsr_q2), len(case["sr"]))
        sr = case["sr"][:n_align]
        # mean detector outputs
        bsr_q_mean = (bsr_q1[:n_align] + bsr_q2[:n_align]) / 2
        bsr_p_mean = (bsr_p1[:n_align] + bsr_p2[:n_align]) / 2
        ob_mean = (ob1[:n_align] + ob2[:n_align]) / 2
        # vs Vista SR correlation
        mask = ~np.isnan(sr) & ~np.isnan(case["sqi"][:n_align]) & (case["sqi"][:n_align] >= 80)

        def cor(x):
            m = mask & ~np.isnan(x[:n_align])
            if m.sum() < 100 or x[:n_align][m].std() < 0.1 or sr[m].std() < 0.1:
                return float("nan")
            return float(np.corrcoef(x[:n_align][m], sr[m])[0, 1])

        rows.append({
            "caseid": cid,
            "n": n_align,
            "r_eeg1_eeg2_raw": r,
            "r_bsr_quazi1_sr": cor(bsr_q1),
            "r_bsr_quazi2_sr": cor(bsr_q2),
            "r_bsr_quazi_mean_sr": cor(bsr_q_mean),
            "r_bsr_paper1_sr": cor(bsr_p1),
            "r_bsr_paper2_sr": cor(bsr_p2),
            "r_bsr_paper_mean_sr": cor(bsr_p_mean),
            "r_openbsr1_sr": cor(ob1),
            "r_openbsr2_sr": cor(ob2),
            "r_openbsr_mean_sr": cor(ob_mean),
        })

    R = pd.DataFrame(rows)
    print("\n=== Per-case correlations ===")
    print(R.to_string(index=False, float_format=lambda x: f"{x:+.3f}" if not np.isnan(x) else "    nan"))

    print("\n=== Mean across cases ===")
    print(R.drop(columns="caseid").mean(numeric_only=True).round(3))

    # Plot a few cases — EEG1 vs EEG2 raw and BSR detectors on both
    fig, axes = plt.subplots(len(case_ids), 2, figsize=(20, 3.5 * len(case_ids)))
    if len(case_ids) == 1:
        axes = axes.reshape(1, -1)
    for ax_row, cid in zip(axes, case_ids):
        case = load_dual_eeg(cid)
        if case is None:
            continue
        eeg1 = preprocess_eeg(case["eeg1"])
        eeg2 = preprocess_eeg(case["eeg2"])
        # EEG plot (1 Hz subsampled for display)
        t_eeg = np.arange(len(eeg1)) / FS
        ax_row[0].plot(t_eeg[::FS], eeg1[::FS], color="tab:blue", lw=0.4, alpha=0.7, label="EEG1")
        ax_row[0].plot(t_eeg[::FS], eeg2[::FS], color="tab:orange", lw=0.4, alpha=0.7, label="EEG2")
        ax_row[0].set_ylim(-60, 60)
        ax_row[0].set_title(f"case {cid} — raw EEG (1-Hz subsample)")
        ax_row[0].set_ylabel("µV")
        ax_row[0].legend(loc="upper right", fontsize=8)

        # BSR plot
        bsr_q1 = openibis_bsr(eeg1, kind="quazi")[::2]
        bsr_q2 = openibis_bsr(eeg2, kind="quazi")[::2]
        ob1 = openbsr(eeg1)[::2]
        ob2 = openbsr(eeg2)[::2]
        n = min(len(bsr_q1), len(bsr_q2), len(case["sr"]))
        t = np.arange(n)
        ax_row[1].plot(t, case["sr"][:n], color="black", lw=1.2, alpha=0.9, label="Vista SR")
        ax_row[1].plot(t, bsr_q1[:n], color="tab:green", lw=0.7, alpha=0.7, label="bsr_quazi EEG1")
        ax_row[1].plot(t, bsr_q2[:n], color="tab:olive", lw=0.7, alpha=0.7, label="bsr_quazi EEG2")
        ax_row[1].plot(t, ob1[:n], color="tab:blue", lw=0.7, alpha=0.7, label="openbsr EEG1")
        ax_row[1].plot(t, ob2[:n], color="tab:cyan", lw=0.7, alpha=0.7, label="openbsr EEG2")
        ax_row[1].set_ylim(-2, 100)
        ax_row[1].set_title(f"case {cid} — BSR detectors on EEG1 vs EEG2")
        ax_row[1].set_ylabel("BSR %")
        ax_row[1].legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    out = RESULTS / "eeg1_vs_eeg2_comparison.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nSaved: {out.name}")


if __name__ == "__main__":
    main()
