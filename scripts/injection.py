"""Shared primitives of the controlled calibration-only injection experiments.

Scientific purpose
------------------
The controlled experiments remove a known set of bins from the calibration
block of a complete record, fill them with a candidate method, and measure the
effect on split-conformal calibration while the test block stays real. Two
properties are load-bearing and are enforced here, in one place:

Exact integer injected count (floor rounding policy)
    ``injected_mask`` removes EXACTLY ``requested_n = floor(frac * block
    length)`` bins for every geometry. For distributed blocks the drawn
    12-96 h blocks are kept, and the final block is trimmed so the total hits
    the requested integer count instead of overshooting it. Note the exactness
    is at the level of the integer count under the floor rule, not the nominal
    fraction itself: for block lengths where frac * length is not an integer,
    realized_n / length differs from frac by less than one bin. Both the
    requested and realized quantities are recorded per run so downstream
    tables report what actually happened.

Strict test isolation
    ``fill_prefix`` receives ONLY the bins strictly before the test boundary
    (training + calibration). No fill method can read a test observation:
    the cubic spline is fitted on pre-test observations only, and a missing
    run that touches the calibration/test boundary is extended from the last
    pre-test observation instead of borrowing the first test value.
    Training-side observations ARE visible to the fill, deliberately: real
    pipelines interpolate with the past available, and the training block is
    observable history at calibration time. Future (test) values never are.

Boundary handling (explicit)
    linear   interpolation between the observations flanking each gap; a gap
             with no observation on its right within the prefix is extended
             with the last observed pre-test value (constant, past-only)
    ffill    last observation carried forward (past-only by construction)
    spline   a cubic spline fitted on all observed prefix bins, evaluated on
             the gaps; extrapolation beyond the last observed prefix bin uses
             the spline's own polynomial, still fitted on pre-test data only
    none     gaps stay NaN (the gap-aware arm: excluded, never fabricated)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BLOCK_MIN_H, BLOCK_MAX_H = 12, 96      # distributed-block geometry, in hours/bins


def injected_mask(length: int, frac: float, geom: str, rng) -> np.ndarray:
    """Boolean mask over a calibration block marking exactly the requested bins.

    Parameters
    ----------
    length : number of bins in the calibration block
    frac   : requested missing fraction; the requested count is
             ``int(frac * length)``
    geom   : 'single' (one contiguous block placed uniformly over ALL legal
             start positions 0..length-target inclusive), 'distributed'
             (12-96 h blocks, final block trimmed to the exact count), or
             'isolated' (uniformly random single bins)
    rng    : np.random.RandomState

    The returned mask always satisfies ``mask.sum() == int(frac * length)``
    (floor rounding of the requested fraction).
    """
    m = np.zeros(length, dtype=bool)
    target = int(frac * length)
    if target == 0:
        return m
    if geom == "single":
        # randint's upper bound is exclusive: +1 so the final legal start
        # position (length - target) can be sampled
        start = rng.randint(0, length - target + 1)
        m[start:start + target] = True
    elif geom == "isolated":
        idx = rng.choice(length, size=target, replace=False)
        m[idx] = True
    elif geom == "distributed":
        removed = 0
        guard = 0
        while removed < target and guard < 100000:
            guard += 1
            blen = rng.randint(BLOCK_MIN_H, BLOCK_MAX_H + 1)
            st = rng.randint(0, length)
            en = min(st + blen, length)
            new = np.flatnonzero(~m[st:en])
            take = min(len(new), target - removed)   # trim the final block: no overshoot
            if take:
                m[st + new[:take]] = True
                removed += take
        assert removed == target, "distributed mask failed to reach the requested count"
    else:
        raise ValueError(f"unknown geometry {geom!r}")
    assert int(m.sum()) == target
    return m


def fill_prefix(prefix: np.ndarray, injected: np.ndarray, method: str) -> np.ndarray:
    """Fill injected gaps using ONLY the pre-test prefix passed in.

    ``prefix`` must contain exactly the training + calibration bins; the test
    block is excluded by the caller, so no method here can observe it. See the
    module docstring for each method's boundary behaviour.
    """
    if len(prefix) != len(injected):
        raise ValueError("prefix and injected mask must have the same length")
    s = np.asarray(prefix, dtype=float).copy()
    s[np.asarray(injected, bool)] = np.nan
    if method == "ffill":
        return pd.Series(s).ffill().bfill().to_numpy()
    if method == "linear":
        return pd.Series(s).interpolate("linear", limit_direction="both").to_numpy()
    if method == "spline":
        from scipy.interpolate import CubicSpline
        idx = np.arange(len(s))
        m = ~np.isnan(s)
        try:
            return CubicSpline(idx[m], s[m], extrapolate=True)(idx)
        except Exception:
            return np.interp(idx, idx[m], s[m])
    if method == "none":
        return s                              # gap-aware: excluded bins stay NaN
    raise ValueError(f"unknown fill method {method!r}")
