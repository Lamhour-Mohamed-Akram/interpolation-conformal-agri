"""Sensitivity of the clean protocol to the interpolation cap (Morocco).

Scientific purpose
------------------
The protocol interpolates only missing runs no longer than a cap, and the
30-minute cap used in the paper is a design choice rather than a derived
quantity. This experiment repeats the clean-protocol calibration with caps of
15, 30, 60 and 120 minutes and reports how the resulting calibration moves, so
a reader can see over what range the conclusions hold and where a permissive
cap starts to reintroduce the effect the protocol is meant to remove.

The preprocessing is the canonical whole-gap rule (preprocess.interpolate_capped)
parameterised by the cap, and the chronological split is target-isolated
(splits.isolated_split), identical to clean_protocol.py: calibration sequences
whose target crosses the test-origin boundary are excluded, and the removed
counts are recorded.

Per cap and horizon the script reports the number of admissible sequences, the
isolated split sizes, split-conformal coverage and width, and
availability-aware ACI coverage and width (gamma = 0.02, window 500, feedback
released only once each outcome's target bin has been reached, identical to
the delayed_aci experiment).

The persistence base is used so that no fitted model varies across caps.

Input       data/data.csv.gz or data/data.csv (either form is accepted)
Output      results/cap_sensitivity/cap_sensitivity.csv
Manuscript  the cap-sensitivity table; the 30-min rows are also the source of
            the Morocco clean-protocol table, so both display identical values
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import scp_quantile, aci, ALPHA                 # noqa: E402
from preprocess import (build_horizon, load_clean_capped, STEPS,   # noqa: E402
                        HORIZONS_H, TRAIN_RATIO, VAL_RATIO)
from splits import isolated_split                              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cap_sensitivity"
CAPS_MIN = (15, 30, 60, 120)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print("Interpolation-cap sensitivity (Morocco, persistence base, target-isolated split)")
    print(f"{'cap':>5} {'H':>4} {'n_seq':>6} {'n_cal':>6} {'SCP PICP':>9} {'SCP MPIW':>9} "
          f"{'ACI PICP':>9} {'ACI MPIW':>9}")
    for cap in CAPS_MIN:
        res, _, valid_win, valid_tgt = load_clean_capped(cap // 5)
        for h in HORIZONS_H:
            X, y, pers, _, ob, tb = build_horizon(res, valid_win, valid_tgt, STEPS[h])
            n = len(y)
            m_tr, m_cal, m_te, st = isolated_split(ob, tb, TRAIN_RATIO, VAL_RATIO)
            # calibration outcomes are all observable before the first test origin
            assert tb[m_cal].max() < ob[m_te][0]
            res_cal = y[m_cal] - pers[m_cal]
            q = scp_quantile(res_cal, ALPHA)
            picp = float(np.mean(np.abs(y[m_te] - pers[m_te]) <= q))
            aci_p, aci_w = aci(res_cal, pers[m_te], y[m_te],
                               origin_bin=ob[m_te], target_bin=tb[m_te],
                               feedback_mode="observable")
            rows.append(dict(cap_min=cap, horizon_h=h, n_seq=n,
                             n_train=st["n_train"], n_cal=st["n_cal"],
                             n_test=st["n_test"],
                             n_train_dropped=st["n_train_dropped"],
                             n_cal_dropped=st["n_cal_dropped"],
                             scp_picp=picp, scp_mpiw=2 * q,
                             aci_picp=aci_p, aci_mpiw=aci_w))
            print(f"{cap:>4}m {h:>3}h {n:>6} {st['n_cal']:>6} {100*picp:>8.1f}% {2*q:>9.2f} "
                  f"{100*aci_p:>8.1f}% {aci_w:>9.2f}")
    pd.DataFrame(rows).to_csv(OUT / "cap_sensitivity.csv", index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}/cap_sensitivity.csv")


if __name__ == "__main__":
    main()
