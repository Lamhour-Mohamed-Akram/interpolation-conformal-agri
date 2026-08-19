"""Immediate versus availability-aware adaptive conformal inference.

Scientific purpose
------------------
An h-step-ahead forecast's outcome becomes observable only h hours after the
forecast origin, so the online coverage feedback that drives ACI must wait for
that physical availability. An implementation that applies the feedback
immediately reacts to information that is not yet available at prediction
time; this experiment measures how much that changes reported coverage.

The causally correct arm uses the shared availability-aware schedule
(conformal.aci, feedback_mode="observable"): the (score, error) of forecast j
enters the online state before issuing forecast k iff
target_bin[j] <= origin_bin[k], evaluated on the physical grid bins of each
surviving sequence. On gap-filtered streams this is NOT equivalent to waiting
a fixed number of evaluation rows — surviving-row counts say nothing about
elapsed physical time — and no row-count approximation is used anywhere.

Both variants run on the same gap-cleaned, target-isolated Morocco split, with
two bases (persistence, and ridge fitted on the isolated training block). The
calibration scores seeding the pool are legal by construction: the shared
target-isolation rule guarantees every calibration target bin lies strictly
before the first test origin bin, which is asserted here.

Alongside coverage the script stores causal feedback diagnostics per base and
horizon: how many test outcomes become observable before the last test
forecast is issued, how many feedback events the online state actually
consumed, when the first test-period feedback arrived, the median number of
outstanding (pending) forecasts, and the fraction of forecasts issued before
any test-period feedback existed.

Input       data/data.csv.gz (or data/data.csv) via preprocess.load_clean()
Computation conformal.aci with feedback_mode="immediate" and "observable"
Output      results/delayed_aci/aci_feedback.csv
Manuscript  the premature-feedback values and the immediate-vs-observable
            coverage figure
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import aci                                    # noqa: E402
from preprocess import (load_clean, build_horizon, STEPS, HORIZONS_H,  # noqa: E402
                        TRAIN_RATIO, VAL_RATIO)
from splits import isolated_split                            # noqa: E402


def main():
    OUT = Path(__file__).resolve().parents[1] / "results" / "delayed_aci"
    OUT.mkdir(parents=True, exist_ok=True)
    res, _, valid_win, valid_tgt = load_clean()
    rows = []
    for h in HORIZONS_H:
        X, y, pers, _, ob, tb = build_horizon(res, valid_win, valid_tgt, STEPS[h])
        n = len(y)
        m_tr, m_cal, m_te, st = isolated_split(ob, tb, TRAIN_RATIO, VAL_RATIO)
        # calibration-pool legality: every calibration outcome is observable
        # strictly before the first test forecast origin
        assert tb[m_cal].max() < ob[m_te][0], \
            "calibration pool contains outcomes not observable before the test"
        rg = Ridge(alpha=1.0).fit(X[m_tr].reshape(m_tr.sum(), -1), y[m_tr])
        ridge_pred = rg.predict(X.reshape(n, -1)).astype(np.float32)
        for base, pred in [("persistence", pers), ("ridge", ridge_pred)]:
            res_cal = y[m_cal] - pred[m_cal]
            cov_imm, wid_imm = aci(res_cal, pred[m_te], y[m_te],
                                   feedback_mode="immediate")
            cov_obs, wid_obs, diag = aci(res_cal, pred[m_te], y[m_te],
                                         origin_bin=ob[m_te], target_bin=tb[m_te],
                                         feedback_mode="observable",
                                         return_diagnostics=True)
            rows.append(dict(base=base, horizon_h=h, horizon_steps=STEPS[h],
                             n_cal=st["n_cal"], n_test=st["n_test"],
                             cov_immediate=cov_imm, cov_delayed=cov_obs,
                             mpiw_immediate=wid_imm, mpiw_delayed=wid_obs,
                             bias_pp=100 * (cov_imm - cov_obs),
                             n_feedback_consumed=diag["n_feedback_consumed"],
                             n_observable_before_last=diag["n_observable_before_last_origin"],
                             first_feedback_index=diag["first_feedback_index"],
                             median_pending=diag["median_pending_at_issue"],
                             frac_before_any_feedback=diag["frac_issued_before_any_feedback"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "aci_feedback.csv", index=False)

    print("ACI coverage at nominal 90% (gap-cleaned, target-isolated Morocco)")
    print(f"{'base':12s}{'H':>4} {'immediate':>11} {'observable':>11} {'diff(pp)':>9} "
          f"{'fb used':>8} {'first fb':>9}")
    for _, r in df.iterrows():
        print(f"{r.base:12s}{int(r.horizon_h):>3}h {100*r.cov_immediate:>10.1f}% "
              f"{100*r.cov_delayed:>10.1f}% {r.bias_pp:>+8.1f} "
              f"{int(r.n_feedback_consumed):>8} {int(r.first_feedback_index):>9}")
    print(f"\nwrote results/delayed_aci/aci_feedback.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
