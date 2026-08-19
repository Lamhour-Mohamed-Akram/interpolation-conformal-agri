"""Chronological target-isolated split masks shared by the clean-protocol scripts.

Scientific purpose
------------------
Splitting a sequence stream by index alone looks chronological but is not:
an h-step-ahead training sequence near the boundary carries a TARGET that
falls inside the validation period, and a calibration sequence near the next
boundary carries a target inside the test period. The clean protocol therefore
removes, from each block, every sequence whose target bin crosses that block's
boundary. This module is the single implementation of that rule; every script
that claims the leakage-free protocol takes its masks from here.

Rule
----
Given each sequence's ``origin_bin`` (the last input bin) and ``target_bin``
(the bin of its h-step-ahead outcome), both on the resampled grid:

* the index split assigns the first ``train_frac`` of sequences to training,
  the next ``val_frac`` to calibration/validation, and the rest to test;
* the validation boundary is the origin bin of the first validation sequence,
  and the test boundary is the origin bin of the first test sequence;
* a training sequence is kept only if its target bin lies strictly before the
  validation boundary;
* a calibration sequence is kept only if its target bin lies strictly before
  the test boundary;
* the test block is the final evaluation block and is left as is.

The returned masks satisfy, by construction and by assertion,
``max(train target bin) < validation origin boundary`` and
``max(calibration target bin) < test origin boundary``.
"""
from __future__ import annotations

import numpy as np


def isolated_split(origin_bin, target_bin, train_frac: float = 0.6,
                   val_frac: float = 0.2):
    """Target-isolated chronological train/calibration/test masks.

    Parameters
    ----------
    origin_bin, target_bin : per-sequence origin and target bins on the grid,
                             in chronological (index) order
    train_frac, val_frac   : chronological split fractions

    Returns
    -------
    train, cal, test : boolean masks over the sequence stream
    stats            : dict with candidate/kept/dropped counts and boundaries
    """
    origin_bin = np.asarray(origin_bin)
    target_bin = np.asarray(target_bin)
    n = len(origin_bin)
    if np.any(np.diff(origin_bin) < 0):
        raise ValueError("sequences must be in chronological origin order")
    tr = int(n * train_frac)
    va = tr + int(n * val_frac)
    train = np.zeros(n, bool); train[:tr] = True
    cal = np.zeros(n, bool); cal[tr:va] = True
    test = np.zeros(n, bool); test[va:] = True

    val_boundary = int(origin_bin[tr]) if tr < n else np.iinfo(np.int64).max
    test_boundary = int(origin_bin[va]) if va < n else np.iinfo(np.int64).max
    train_iso = train & (target_bin < val_boundary)
    cal_iso = cal & (target_bin < test_boundary)

    if train_iso.any():
        assert int(target_bin[train_iso].max()) < val_boundary
    if cal_iso.any():
        assert int(target_bin[cal_iso].max()) < test_boundary

    stats = dict(
        n_candidates=int(n),
        n_train_index=int(tr), n_cal_index=int(va - tr), n_test=int(n - va),
        n_train=int(train_iso.sum()), n_cal=int(cal_iso.sum()),
        n_train_dropped=int(tr - train_iso.sum()),
        n_cal_dropped=int((va - tr) - cal_iso.sum()),
        val_origin_boundary=val_boundary, test_origin_boundary=test_boundary,
    )
    return train_iso, cal_iso, test, stats
