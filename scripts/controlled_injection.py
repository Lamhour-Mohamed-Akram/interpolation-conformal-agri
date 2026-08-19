"""Controlled calibration-only interpolation injection.

Scientific purpose
------------------
This is the paper's central causal experiment. The substrate is the
Johannesburg soil-moisture record, which is gap-free and hourly, so any
interpolation present is the interpolation we put there ourselves.

The TEST block is held real and untouched throughout. Synthetic gaps are
injected ONLY into the calibration (validation) block, filled by one of the
candidate methods, and split-conformal is then calibrated on the filled block
and evaluated on the real test block. Any change in coverage is therefore
attributable to the calibration-side interpolation alone, not to a change in
the evaluation data or in the forecaster.

Test isolation
--------------
Every fill method operates on the training + calibration prefix ONLY
(injection.fill_prefix): the cubic spline is fitted on pre-test observations,
and a gap that touches the calibration/test boundary is extended from the last
pre-test observation instead of borrowing a test value. Training-side history
is deliberately visible to the fill — it is observable at calibration time —
but no future/test value ever is, so replacing every test observation with an
arbitrary value leaves every calibration residual and conformal quantile
unchanged (verified in verify_outputs.py).

Exact integer injected count
----------------------------
The injected mask removes exactly ``int(frac * block length)`` bins for every
geometry (injection.injected_mask); for distributed blocks the final 12-96 h
block is trimmed to the requested total rather than overshooting it. Each row
records requested_n/realized_n and requested_frac/realized_frac, and
verify_outputs.py checks that requested and realized counts agree.

Design
------
fraction   0, 10, 20, 30, 40, 50 % of the calibration block
geometry   single long block | distributed blocks (12-96 h) | isolated points
fill       linear | forward-fill | cubic spline | none (gap-aware exclusion)
horizons   6, 12, 24, 48 h
placements 20 independent random placements (seeds 0..19)

The base forecaster is persistence, which requires no training, so no fitted
model can absorb or mask the effect.

Per cell the script records split-conformal coverage (PICP), mean interval
width (MPIW), the calibration error |PICP - 0.90|, and the 90th percentile of
the calibration residuals, which is the quantity the fill method distorts.

Input       data/mendeley_soil_data_incl_rain_v3.csv
Output      results/controlled_injection/injection_matrix.csv (one row per cell)
Manuscript  the injection tables, the geometry-resolved table, the per-horizon
            table, and the injection and mechanism figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import scp_quantile, ALPHA                    # noqa: E402
from injection import injected_mask, fill_prefix             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "mendeley_soil_data_incl_rain_v3.csv"
OUT = ROOT / "results" / "controlled_injection"

TARGET = "Soil_Moisture"
HORIZONS = (6, 12, 24, 48)
FRACS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
GEOMS = ("single", "distributed", "isolated")
METHODS = ("linear", "ffill", "spline", "none")
REPS = 20
TR, VA = 0.6, 0.8


def load_series() -> np.ndarray:
    """Hourly soil-moisture series on a complete grid.

    The controlled design requires a substrate with no pre-existing gaps, so
    that every interpolated value in the experiment is one this script injected.
    That property is asserted here rather than repaired by interpolation: if the
    record ever ceased to be complete, silently filling it would confound the
    very effect being measured.
    """
    df = pd.read_csv(CSV)
    df.columns = df.columns.str.strip()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = (df.dropna(subset=["Time"]).sort_values("Time")
            .drop_duplicates("Time", keep="last").set_index("Time"))
    series = df[TARGET].resample("1h").median()
    n_missing = int(series.isna().sum())
    assert n_missing == 0, (
        f"substrate is not gap-free: {n_missing} missing hourly bins. The "
        "controlled injection design assumes a complete record.")
    return series.to_numpy(np.float64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = load_series()
    L = len(s)
    tr, va = int(L * TR), int(L * VA)
    rows = []
    for geom in GEOMS:
        for method in METHODS:
            for frac in FRACS:
                for seed in range(REPS):
                    rng = np.random.RandomState(seed)
                    inj = np.zeros(L, bool)
                    inj[tr:va] = injected_mask(va - tr, frac, geom, rng)
                    requested_n = int(frac * (va - tr))
                    realized_n = int(inj.sum())
                    # fill sees the pre-test prefix only: no test value can
                    # participate in filling a calibration gap
                    sf = fill_prefix(s[:va], inj[:va], method)
                    for h in HORIZONS:
                        oc = np.arange(tr, va - h)            # calibration origins
                        if method == "none":
                            ok = ~inj[oc] & ~inj[oc + h]      # exclude injected bins
                            rc = np.abs(s[oc[ok] + h] - s[oc[ok]])
                        else:
                            rc = np.abs(sf[oc + h] - sf[oc])
                        q = scp_quantile(rc, ALPHA)
                        ot = np.arange(va, L - h)             # test origins, always real
                        rt = np.abs(s[ot + h] - s[ot])
                        picp = float(np.mean(rt <= q))
                        rows.append(dict(geom=geom, method=method, frac=frac, seed=seed,
                                         requested_frac=frac,
                                         realized_frac=realized_n / (va - tr),
                                         requested_n=requested_n, realized_n=realized_n,
                                         h=h, picp=picp, mpiw=2 * q,
                                         calib_err=abs(picp - (1 - ALPHA)),
                                         val_res_p90=float(np.nanpercentile(rc, 90))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "injection_matrix.csv", index=False)

    print("fill method x fraction (distributed geometry, mean |PICP-0.90| over horizons and placements)")
    print(f"{'frac':>6} " + " ".join(f"{m:>12}" for m in METHODS))
    for frac in FRACS:
        cells = []
        for m in METHODS:
            g = df[(df.geom == "distributed") & (df.method == m) & (df.frac == frac)]
            cells.append(f"{100*g.calib_err.mean():.1f} pp")
        print(f"{frac:6.2f} " + " ".join(f"{c:>12}" for c in cells))

    print("\ngeometry x fraction (linear fill, mean PICP; nominal 90%)")
    print(f"{'frac':>6} " + " ".join(f"{g:>14}" for g in GEOMS))
    for frac in FRACS:
        cells = [f"{100*df[(df.geom == g) & (df.method == 'linear') & (df.frac == frac)].picp.mean():.1f}%"
                 for g in GEOMS]
        print(f"{frac:6.2f} " + " ".join(f"{c:>14}" for c in cells))
    print(f"\nwrote {OUT.relative_to(ROOT)}/injection_matrix.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
