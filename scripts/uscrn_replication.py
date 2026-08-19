"""Replication of the controlled injection experiment on an independent record.

Scientific purpose
------------------
The controlled experiment is repeated on the USCRN Fairhope 3 NE station
(hourly, 2021, soil moisture at 5 cm), a research-grade record maintained
independently of the greenhouse deployments studied elsewhere in the paper and
essentially complete. Reproducing the effect there shows it is a property of the
calibration procedure rather than of one particular sensor installation.

As in the main controlled experiment, the test block is held real, interpolation
is injected only into the calibration block, and coverage is measured with a
persistence base under two arms:

linear      the injected gaps are linearly filled and all calibration origins
            are used
gap-aware   the injected bins are excluded from the calibration stream instead
            of being filled

The injected mask and the fill share the main experiment's primitives
(injection.py): the mask removes exactly the requested number of bins, the
fill operates on the pre-test prefix only, and requested/realized counts are
recorded per row.

Design      fractions 0-50%, horizons 6/12/24/48 h, distributed blocks of
            12-96 h, 20 random placements (seeds 0..19)

Input       data/uscrn_fairhope_AL_2021.txt (fixed-width hourly product;
            column 28 is the 5 cm soil-moisture channel, -99/-9999 are the
            product's missing-value codes)
Output      results/uscrn/uscrn_injection.csv   one row per (fraction, horizon,
            arm, placement), so means and spreads are computed downstream rather
            than being baked in here
Manuscript  the USCRN replication figure and the quoted gap-aware coverage range
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import scp_quantile, ALPHA                     # noqa: E402
from injection import injected_mask, fill_prefix              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "uscrn_fairhope_AL_2021.txt"
OUT = ROOT / "results" / "uscrn"

SM_COLUMN = 28
MISSING = [-99.0, -9999.0]
HORIZONS = (6, 12, 24, 48)
FRACS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
REPS = 20
TR, VA = 0.6, 0.8


def load_series():
    """Hourly 5 cm soil moisture on a complete grid.

    The product's missing-value codes are mapped to NaN and the series is then
    asserted to be complete, for the same reason as in the main controlled
    experiment: the replication is only interpretable if every interpolated
    value present is one this script injected.
    """
    d = pd.read_csv(RAW, sep=r"\s+", header=None)
    raw = d[SM_COLUMN].replace(MISSING, np.nan)
    n_missing = int(raw.isna().sum())
    assert n_missing == 0, (
        f"substrate is not gap-free: {n_missing} missing hourly bins. The "
        "controlled injection design assumes a complete record.")
    return raw.to_numpy(float), float(1 - raw.isna().mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sm, completeness = load_series()
    L = len(sm)
    tr, va = int(L * TR), int(L * VA)
    print(f"USCRN Fairhope 3 NE 2021: {L} hourly observations, "
          f"soil-moisture completeness {100*completeness:.1f}%")

    rows = []
    for frac in FRACS:
        for seed in range(REPS):
            rng = np.random.RandomState(seed)
            inj = np.zeros(L, bool)
            if frac > 0:
                inj[tr:va] = injected_mask(va - tr, frac, "distributed", rng)
            requested_n = int(frac * (va - tr))
            realized_n = int(inj.sum())
            # fill sees the pre-test prefix only (see injection.fill_prefix)
            filled = fill_prefix(sm[:va], inj[:va], "linear")
            for h in HORIZONS:
                oc = np.arange(tr, va - h)                    # calibration origins
                q_lin = scp_quantile(filled[oc + h] - filled[oc], ALPHA)
                ok = ~inj[oc] & ~inj[oc + h]                  # gap-aware exclusion
                q_gap = scp_quantile(sm[oc[ok] + h] - sm[oc[ok]], ALPHA)
                ot = np.arange(va, L - h)                     # test origins, always real
                err = np.abs(sm[ot + h] - sm[ot])
                common = dict(frac=frac, horizon_h=h, seed=seed,
                              requested_frac=frac, realized_frac=realized_n / (va - tr),
                              requested_n=requested_n, realized_n=realized_n)
                rows.append(dict(**common, arm="linear",
                                 picp=float(np.mean(err <= q_lin)), mpiw=2 * q_lin))
                rows.append(dict(**common, arm="gap_aware",
                                 picp=float(np.mean(err <= q_gap)), mpiw=2 * q_gap))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "uscrn_injection.csv", index=False)

    print(f"\n{'frac':>6} " + " ".join(f"{f'{a} {h}h':>16}" for a in ("linear", "gap_aware")
                                       for h in (6, 48)))
    for frac in FRACS:
        cells = []
        for arm in ("linear", "gap_aware"):
            for h in (6, 48):
                g = df[(df.frac == frac) & (df.arm == arm) & (df.horizon_h == h)]
                cells.append(f"{100*g.picp.mean():.1f}%")
        print(f"{frac:6.2f} " + " ".join(f"{c:>16}" for c in cells))
    print(f"\nwrote {OUT.relative_to(ROOT)}/uscrn_injection.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
