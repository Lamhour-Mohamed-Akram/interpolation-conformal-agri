"""Sensitivity of the cost-loss irrigation simulation to its two free settings.

Scientific purpose
------------------
The decision-level analysis irrigates whenever an interval's lower bound
crosses a stress threshold tau, and scores each strategy by
(irrigations + r x missed stress events) per test hour. The headline table
fixes tau at the 25th percentile of the training target and r = 50. Because
the ranking of strategies can depend on both choices, this module re-runs the
identical decision rule over a grid of threshold percentiles and penalty
ratios, on the same nominal-90% intervals, and records the stress-event count
that each threshold produces (a threshold that yields zero or all-hour events
makes the simulation degenerate).

Nothing is recomputed on the interval side: the calling script passes the
intervals it already built for the headline table, so the sensitivity rows
are exactly comparable with it.

Used by    ensembles_and_decision.py (Morocco) and
           ensembles_and_decision_multisite.py (Iraq, Johannesburg)
Output     <site results dir>/uq_extensions/decision_sensitivity.csv
Manuscript the threshold x penalty sensitivity table of the decision section
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SENS_PERCENTILES = [10, 15, 20, 25, 30, 35, 40, 50]
SENS_RATIOS = [1, 2, 5, 10, 20, 50, 100]
KEYS = ["irr_rate", "miss_rate", "fa_rate"] + [f"cost_r{r}" for r in SENS_RATIOS]


def _decide(lo, event, n_ev, tau, n):
    irr = lo < tau
    miss = event & ~irr
    fa = irr & ~event
    out = {"irr_rate": float(irr.mean()), "miss_rate": float(miss.sum() / max(n_ev, 1)),
           "fa_rate": float(fa.mean())}
    for r in SENS_RATIOS:
        out[f"cost_r{r}"] = float((r * miss.sum() + irr.sum()) / n)
    return out


def sensitivity_rows(y_train, y_test, intervals90, sim_methods, horizons):
    """Decision metrics for every (percentile, horizon, strategy) cell.

    ``intervals90[(method, horizon)]`` is either a (lo, hi) pair or a list of
    such pairs (one per seed), exactly as the calling script stores them; a
    list is averaged over seeds as in the headline simulation.
    """
    rows = []
    for pct in SENS_PERCENTILES:
        tau = float(np.percentile(y_train[:, 0], pct))
        for h_i, h in enumerate(horizons):
            yt = y_test[:, h_i]
            n = len(yt)
            event = yt < tau
            n_ev = int(event.sum())
            oracle = {"irr_rate": float(event.mean()), "miss_rate": 0.0, "fa_rate": 0.0}
            always = {"irr_rate": 1.0, "miss_rate": 0.0, "fa_rate": float((~event).mean())}
            never = {"irr_rate": 0.0, "miss_rate": 1.0 if n_ev else 0.0, "fa_rate": 0.0}
            for r in SENS_RATIOS:
                oracle[f"cost_r{r}"] = float(event.mean())
                always[f"cost_r{r}"] = 1.0
                never[f"cost_r{r}"] = float(r * n_ev / n)
            for name, d in [("oracle", oracle), ("always_irrigate", always),
                            ("never_irrigate", never)]:
                rows.append((pct, tau, name, h, n_ev, d))
            for m in sim_methods:
                iv = intervals90[(m, h)]
                if isinstance(iv, list):
                    ds = [_decide(np.asarray(lo, float), event, n_ev, tau, n) for lo, _ in iv]
                    d = {k: float(np.mean([x[k] for x in ds])) for k in ds[0]}
                else:
                    d = _decide(np.asarray(iv[0], float), event, n_ev, tau, n)
                rows.append((pct, tau, m, h, n_ev, d))
    return rows


def write_sensitivity(out_dir: Path, rows) -> None:
    with open(Path(out_dir) / "decision_sensitivity.csv", "w") as f:
        f.write("threshold_pct,tau,method,horizon,n_events," + ",".join(KEYS) + "\n")
        for pct, tau, m, h, n_ev, d in rows:
            f.write(f"{pct},{tau:.4f},{m},{h},{n_ev}," + ",".join(f"{d[k]:.4f}" for k in KEYS) + "\n")
