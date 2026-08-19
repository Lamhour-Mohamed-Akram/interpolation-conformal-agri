"""Synthetic tests of the repository's methodological invariants.

Each test constructs a small synthetic example with a known correct answer and
checks the shared implementation against it. verify_outputs.py runs the same
suite (plus data-level checks); this file can also be run on its own:

    python scripts/test_methodology.py

Exits non-zero if any invariant fails.

Covered invariants
------------------
* whole-gap capped interpolation: runs of length 1, exactly cap, cap+1, very
  long, leading, trailing, two separate runs, and independent runs in several
  features — short bounded runs are fully filled, longer runs stay entirely
  NaN, edges are never extrapolated
* chronological target isolation: no retained training target reaches the
  validation-origin boundary and no retained calibration target reaches the
  test-origin boundary
* injection masks: realized count equals the requested count for every
  geometry, and distributed blocks respect the 12-96 h geometry
* controlled-injection test isolation: the fill never sees test values, so
  changing every test value leaves calibration residuals and the conformal
  quantile unchanged
* persistence scaler: mapping the feature-scaled soil-moisture value back
  through the FEATURE scaler recovers the raw value even when feature and
  target scaling ranges differ
* conformal quantile: the canonical function returns the hand-computed
  finite-sample order statistic on a toy residual array
* horizon semantics: on synthetic 5-minute and hourly grids, every nominal
  horizon satisfies target_bin - origin_bin == steps exactly (6 h means
  72 five-minute bins and 6 hourly bins, never 73 or 7), and target
  timestamps sit exactly h hours after the origin timestamps
* availability-aware ACI: on a gap-filtered synthetic stream, feedback is
  released by target-bin availability, not by surviving-row counts; on a
  dense stream the availability rule reproduces a fixed-row-delay reference
  exactly; look-ahead is structurally impossible
* single-block placement: the final legal start position (length - target)
  is inside the sampling domain, and full-block/empty edge cases behave
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import conformal_quantile, scp_quantile, aci     # noqa: E402
from injection import injected_mask, fill_prefix                # noqa: E402
from preprocess import interpolate_capped, nan_runs             # noqa: E402
from preprocess import build_horizon, FEATURE_COLS, SEQ_LEN     # noqa: E402
import sites                                                    # noqa: E402
from splits import isolated_split                               # noqa: E402

CAP = 6


def _frame(values_by_col):
    n = max(len(v) for v in values_by_col.values())
    idx = pd.date_range("2022-01-01", periods=n, freq="5min")
    return pd.DataFrame({c: v for c, v in values_by_col.items()}, index=idx)


def check_whole_gap():
    results = []

    def series_with_gap(gap_len, lead=10, tail=10):
        v = np.arange(lead + gap_len + tail, dtype=float)
        v[lead:lead + gap_len] = np.nan
        return v

    # gap of length 1, exactly cap, cap+1, and very long
    for gap, should_fill in [(1, True), (CAP, True), (CAP + 1, False), (50, False)]:
        v = series_with_gap(gap)
        out = interpolate_capped(_frame({"a": v}), CAP)["a"].to_numpy()
        filled = ~np.isnan(out[10:10 + gap])
        ok = filled.all() if should_fill else (~filled).all()
        results.append((f"whole-gap: run of {gap} (cap {CAP}) "
                        f"{'fully interpolated' if should_fill else 'entirely NaN'}",
                        bool(ok), f"filled bins: {int(filled.sum())}/{gap}"))
        # never a partial fill, whatever the length
        results.append((f"whole-gap: run of {gap} not partially filled",
                        bool(filled.all() or (~filled).any() and not filled.any()),
                        f"filled bins: {int(filled.sum())}/{gap}"))

    # short interpolated values are the linear interpolation between endpoints
    v = series_with_gap(3)
    out = interpolate_capped(_frame({"a": v}), CAP)["a"].to_numpy()
    results.append(("whole-gap: short run linearly interpolated between endpoints",
                    bool(np.allclose(out, np.arange(len(v), dtype=float))), ""))

    # leading and trailing runs are never extrapolated, even short ones
    v = np.arange(20.0); v[:3] = np.nan
    out = interpolate_capped(_frame({"a": v}), CAP)["a"].to_numpy()
    results.append(("whole-gap: leading run stays NaN (no extrapolation)",
                    bool(np.isnan(out[:3]).all() and ~np.isnan(out[3:]).any()), ""))
    v = np.arange(20.0); v[-3:] = np.nan
    out = interpolate_capped(_frame({"a": v}), CAP)["a"].to_numpy()
    results.append(("whole-gap: trailing run stays NaN (no extrapolation)",
                    bool(np.isnan(out[-3:]).all() and ~np.isnan(out[:-3]).any()), ""))

    # two separate runs treated independently
    v = np.arange(40.0)
    v[5:5 + 4] = np.nan            # short: filled
    v[20:20 + CAP + 2] = np.nan    # long: stays NaN in full
    out = interpolate_capped(_frame({"a": v}), CAP)["a"].to_numpy()
    ok = (~np.isnan(out[5:9]).any() and np.isnan(out[20:20 + CAP + 2]).all()
          and ~np.isnan(out[9:20]).any())
    results.append(("whole-gap: two separate runs handled independently", bool(ok), ""))

    # independent gaps in multiple features
    a = np.arange(40.0); a[8:8 + 2] = np.nan                # short in a
    b = np.arange(40.0) * 2; b[15:15 + CAP + 3] = np.nan    # long in b
    out = interpolate_capped(_frame({"a": a, "b": b}), CAP)
    ok = (~out["a"].isna().any()
          and out["b"].to_numpy()[15:15 + CAP + 3].tolist().count(0) == 0
          and np.isnan(out["b"].to_numpy()[15:15 + CAP + 3]).all()
          and ~np.isnan(out["b"].to_numpy()[:15]).any())
    results.append(("whole-gap: features interpolated independently", bool(ok), ""))

    # nan_runs identifies exact run extents
    m = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    results.append(("whole-gap: nan_runs finds exact run extents",
                    nan_runs(m) == [(1, 3), (5, 6), (7, 10)], str(nan_runs(m))))
    return results


def check_isolation():
    results = []
    # 10 sequences, origins 0..9, targets origin + 3 (horizon 3)
    origin = np.arange(24, 34)
    target = origin + 3
    train, cal, test, st = isolated_split(origin, target, 0.6, 0.2)
    # index split: train 0..5, cal 6..7, test 8..9
    val_b, test_b = st["val_origin_boundary"], st["test_origin_boundary"]
    ok_train = (not train.any()) or (target[train].max() < val_b)
    ok_cal = (not cal.any()) or (target[cal].max() < test_b)
    results.append(("isolation: max train target < validation origin boundary",
                    bool(ok_train), ""))
    results.append(("isolation: max calibration target < test origin boundary",
                    bool(ok_cal), ""))
    # hand check: train indices 0..5 have targets 27..32; val boundary = origin[6]=30
    # -> indices with target >= 30 (targets 30,31,32 => idx 3,4,5) must be dropped
    results.append(("isolation: hand-computed dropped training sequences",
                    st["n_train_dropped"] == 3 and st["n_train"] == 3,
                    f"dropped {st['n_train_dropped']}, kept {st['n_train']}"))
    # cal indices 6..7 -> targets 33,34; test boundary = origin[8]=32 -> both dropped
    results.append(("isolation: hand-computed dropped calibration sequences",
                    st["n_cal_dropped"] == 2 and st["n_cal"] == 0,
                    f"dropped {st['n_cal_dropped']}, kept {st['n_cal']}"))
    # horizon 1: only the sequence whose target lands exactly on the boundary
    # bin is dropped from each block (strict inequality)
    t1 = origin + 1
    _, _, _, st1 = isolated_split(origin, t1, 0.6, 0.2)
    results.append(("isolation: 1-step horizon drops exactly the boundary sequence",
                    st1["n_train_dropped"] == 1 and st1["n_cal_dropped"] == 1,
                    f"train {st1['n_train_dropped']}, cal {st1['n_cal_dropped']}"))
    return results


def check_injection_masks():
    results = []
    mismatches = []
    for L in (500, 1753):
        for frac in (0.1, 0.2, 0.3, 0.5):
            for geom in ("single", "distributed", "isolated"):
                for seed in (0, 1, 7):
                    m = injected_mask(L, frac, geom, np.random.RandomState(seed))
                    want = int(frac * L)
                    if int(m.sum()) != want:
                        mismatches.append(f"{geom} L={L} f={frac} seed={seed}: "
                                          f"{int(m.sum())} != {want}")
    results.append(("injection: realized count equals requested count "
                    "for every geometry/fraction/seed tested",
                    not mismatches, "; ".join(mismatches)))
    # distributed geometry: every drawn block is 12-96 h; a run shorter than
    # 12 h can only come from the trimmed final block or a block truncated at
    # the array edge, so at most two such runs exist per mask (overlapping
    # blocks can merge into longer contiguous runs, so no upper bound applies)
    for seed in (0, 3, 9):
        m = injected_mask(1753, 0.3, "distributed", np.random.RandomState(seed))
        runs = nan_runs(m)
        short = sum(1 for a, b in runs if b - a < 12)
        results.append((f"injection: distributed geometry blocky (seed {seed})",
                        short <= 2, f"{short} runs shorter than 12 bins"))
    return results


def check_fill_isolation():
    results = []
    rng = np.random.RandomState(0)
    L, tr, va = 1000, 600, 800
    s = np.cumsum(rng.randn(L)) + 50.0
    inj = np.zeros(L, bool)
    inj[tr:va] = injected_mask(va - tr, 0.3, "distributed", np.random.RandomState(5))
    inj[va - 5: va] = True                              # gap touching the test boundary
    for method in ("linear", "ffill", "spline", "none"):
        base = fill_prefix(s[:va], inj[:va], method)
        s2 = s.copy()
        s2[va:] = rng.randn(L - va) * 1e3               # arbitrary test values
        alt = fill_prefix(s2[:va], inj[:va], method)
        same_fill = np.allclose(np.nan_to_num(base, nan=-9e9),
                                np.nan_to_num(alt, nan=-9e9))
        # calibration residuals and the conformal quantile must also be unchanged
        h = 6
        oc = np.arange(tr, va - h)
        if method == "none":
            ok_m = ~inj[oc] & ~inj[oc + h]
            r1 = np.abs(s[oc[ok_m] + h] - s[oc[ok_m]])
            r2 = np.abs(s2[oc[ok_m] + h] - s2[oc[ok_m]])
        else:
            r1 = np.abs(base[oc + h] - base[oc])
            r2 = np.abs(alt[oc + h] - alt[oc])
        same_q = np.isclose(scp_quantile(r1, 0.1), scp_quantile(r2, 0.1))
        results.append((f"fill isolation: {method} fill invariant to test values",
                        bool(same_fill and np.allclose(r1, r2, equal_nan=True) and same_q), ""))
    # the boundary-touching gap must not equal what a test-borrowing fill gives
    lin = fill_prefix(s[:va], inj[:va], "linear")
    full = pd.Series(np.where(inj, np.nan, s)).interpolate(
        "linear", limit_direction="both").to_numpy()
    results.append(("fill isolation: boundary gap does not borrow the test value",
                    not np.allclose(lin[va - 5:va], full[va - 5:va]),
                    "prefix fill equals full-series fill at the boundary"))
    return results


def check_persistence_scaler():
    from sklearn.preprocessing import MinMaxScaler
    results = []
    rng = np.random.RandomState(1)
    raw_feat = rng.uniform(20, 90, size=(300, 2))       # col 0 is the target feature
    raw_targ = rng.uniform(35, 70, size=(300, 4))       # deliberately different range
    fsc, tsc = MinMaxScaler().fit(raw_feat), MinMaxScaler().fit(raw_targ)
    X_scaled = fsc.transform(raw_feat)
    ti = 0
    recovered = X_scaled[:, ti] * (fsc.data_max_[ti] - fsc.data_min_[ti]) + fsc.data_min_[ti]
    results.append(("persistence: feature-scaler inversion recovers raw values",
                    bool(np.allclose(recovered, raw_feat[:, ti])), ""))
    # the previously used path (inverting through the TARGET scaler) is wrong
    # whenever the ranges differ — demonstrate the discrepancy exists
    wrong = tsc.inverse_transform(np.repeat(X_scaled[:, ti:ti + 1], 4, axis=1))[:, 0]
    results.append(("persistence: target-scaler inversion is biased when ranges differ",
                    bool(not np.allclose(wrong, raw_feat[:, ti])), ""))
    return results


def check_conformal_quantile():
    results = []
    scores = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 9.0, 7.0, 6.0, 8.0, 10.0])
    srt = np.sort(scores)
    # n=10, alpha=0.1: k = ceil(11*0.9) = 10 -> 10th smallest = 10.0
    results.append(("conformal: toy case alpha=0.1 -> 10th order statistic",
                    conformal_quantile(scores, 0.1) == srt[9],
                    f"{conformal_quantile(scores, 0.1)}"))
    # alpha=0.5: k = ceil(11*0.5) = 6 -> 6th smallest = 6.0
    results.append(("conformal: toy case alpha=0.5 -> 6th order statistic",
                    conformal_quantile(scores, 0.5) == srt[5],
                    f"{conformal_quantile(scores, 0.5)}"))
    # k is clipped to n when (n+1)(1-alpha) exceeds n
    results.append(("conformal: rank clipped to n",
                    conformal_quantile(scores, 0.01) == srt[9], ""))
    # NaNs are dropped, not propagated
    results.append(("conformal: NaN scores dropped",
                    conformal_quantile(np.concatenate([scores, [np.nan]]), 0.5) == srt[5], ""))
    # scp_quantile takes absolute values first
    results.append(("conformal: scp_quantile uses |residual|",
                    scp_quantile(-scores, 0.5) == srt[5], ""))
    return results


def check_horizon_semantics():
    results = []
    rng = np.random.RandomState(0)

    # synthetic Morocco-like 5-minute grid, fully observed
    n5 = 24 + 48 * 12 + 40
    idx5 = pd.date_range("2022-01-01", periods=n5, freq="5min")
    res5 = pd.DataFrame({c: rng.rand(n5) * 50 + 20 for c in FEATURE_COLS}, index=idx5)
    valid = np.ones(n5, bool)
    for h in (6, 12, 24, 48):
        steps = h * 12
        X, y, pers, ds, ob, tb = build_horizon(res5, valid, valid, steps)
        ok = len(ob) > 0 and np.all(tb - ob == steps)
        results.append((f"horizon: 5-min grid {h}h is exactly {steps} bins "
                        f"(not {steps + 1})", bool(ok),
                        f"diffs {np.unique(tb - ob) if len(ob) else 'empty'}"))
        dt = (res5.index.to_numpy()[tb] - res5.index.to_numpy()[ob])
        ok_t = np.all(dt == np.timedelta64(h, "h"))
        results.append((f"horizon: 5-min grid {h}h target timestamp is exactly "
                        f"{h}h after origin", bool(ok_t), ""))
        # window/target relationship: target sits steps bins after the LAST
        # observed input, and persistence equals the origin-bin value
        ok_p = np.allclose(pers, res5["humiditysol"].to_numpy(np.float32)[ob])
        results.append((f"horizon: 5-min grid {h}h persistence reads the origin bin",
                        bool(ok_p), ""))

    # synthetic hourly grid through sites.build_sequences
    nH = 24 + 48 + 30
    idxH = pd.date_range("2022-01-01", periods=nH, freq="1h")
    feats = ["a", "b", "tgt"]
    resH = pd.DataFrame({c: rng.rand(nH) * 10 for c in feats}, index=idxH)
    validH = np.ones(nH, bool)
    for h in (6, 12, 24, 48):
        X, y, pers, ob, tb = sites.build_sequences(resH, validH, validH,
                                                   feats, "tgt", h)
        ok = len(ob) > 0 and np.all(tb - ob == h)
        results.append((f"horizon: hourly grid {h}h is exactly {h} bins (not {h + 1})",
                        bool(ok), f"diffs {np.unique(tb - ob) if len(ob) else 'empty'}"))
        dt = (resH.index.to_numpy()[tb] - resH.index.to_numpy()[ob])
        results.append((f"horizon: hourly grid {h}h target timestamp is exactly "
                        f"{h}h after origin",
                        bool(np.all(dt == np.timedelta64(h, "h"))), ""))
    return results


def check_aci_availability():
    results = []
    rng = np.random.RandomState(3)

    # 8C: gap-filtered stream. Origins [0,1,2,20,21,22], 5-bin horizon ->
    # targets [5,6,7,25,26,27]. At origin 20 the first three outcomes are
    # already observable although only 3 evaluation rows have elapsed; a
    # row-count-delayed algorithm (delay=5 rows) would release nothing yet.
    ob = np.array([0, 1, 2, 20, 21, 22])
    tb = ob + 5
    y = rng.rand(6) * 10
    yhat = y + rng.randn(6)
    res_cal = np.abs(rng.randn(50)) + 0.5
    cov, wid, diag = aci(res_cal, yhat, y, origin_bin=ob, target_bin=tb,
                         feedback_mode="observable", return_diagnostics=True)
    results.append(("ACI availability: gap-filtered stream releases feedback "
                    "by target bin, first at the post-gap forecast",
                    diag["first_feedback_index"] == 3,
                    f"first_feedback_index={diag['first_feedback_index']}"))
    results.append(("ACI availability: exactly the three observable outcomes "
                    "are consumed within the stream",
                    diag["n_feedback_consumed"] == 3
                    and diag["n_observable_before_last_origin"] == 3,
                    f"consumed={diag['n_feedback_consumed']}"))

    # 8D: dense-grid equivalence against a fixed-row-delay reference
    def reference_row_delay(res_cal, yhat, y, delay, gamma=0.02, window=500,
                            alpha=0.10):
        pool = list(np.abs(np.asarray(res_cal, dtype=float))[-window:])
        a_t = alpha
        n = len(y)
        lo = np.zeros(n); hi = np.zeros(n)
        pending = []
        for t in range(n):
            for _, r, e in [p for p in pending if p[0] <= t]:
                a_t += gamma * (alpha - e)
                pool.append(r)
                if len(pool) > window:
                    pool.pop(0)
            pending = [p for p in pending if p[0] > t]
            eff = min(max(1 - a_t, 1e-3), 1 - 1e-3)
            q = float(np.quantile(pool, eff))
            lo[t], hi[t] = yhat[t] - q, yhat[t] + q
            c = lo[t] <= y[t] <= hi[t]
            pending.append((t + delay, abs(y[t] - yhat[t]), float(not c)))
        return lo, hi

    nD, D = 400, 12
    yD = np.cumsum(rng.randn(nD)) + 50
    yhatD = yD + rng.randn(nD) * 2
    calD = np.abs(rng.randn(120)) * 2
    obD = np.arange(nD)
    loA, hiA, _ = aci(calD, yhatD, yD, origin_bin=obD, target_bin=obD + D,
                      feedback_mode="observable", return_traces=True)
    loR, hiR = reference_row_delay(calD, yhatD, yD, D)
    results.append(("ACI availability: dense stream equals fixed-row-delay "
                    "reference exactly",
                    bool(np.allclose(loA, loR) and np.allclose(hiA, hiR)),
                    f"max diff {np.max(np.abs(loA - loR)):.3g}"))

    # immediate mode equals row-delay reference with delay 0 (the defect arm)
    loI, hiI, _ = aci(calD, yhatD, yD, feedback_mode="immediate",
                      return_traces=True)
    loR0, hiR0 = reference_row_delay(calD, yhatD, yD, 0)
    results.append(("ACI availability: immediate mode reproduces the "
                    "next-forecast-feedback defect exactly",
                    bool(np.allclose(loI, loR0) and np.allclose(hiI, hiR0)), ""))

    # 8E: structural no-look-ahead — a target bin at/before its origin is refused
    try:
        aci(calD[:10], yhatD[:5], yD[:5], origin_bin=np.arange(5),
            target_bin=np.arange(5), feedback_mode="observable")
        ok = False
    except ValueError:
        ok = True
    results.append(("ACI availability: target bins not after origins are refused",
                    ok, ""))
    return results


def check_single_block_range():
    results = []
    # 8G: with length 10 and target 7, legal starts are 0..3 inclusive; the
    # final legal placement must actually occur under repeated sampling
    starts = set()
    for seed in range(400):
        m = injected_mask(10, 0.7, "single", np.random.RandomState(seed))
        run = nan_runs(m)
        assert len(run) == 1 and run[0][1] - run[0][0] == 7
        starts.add(run[0][0])
    results.append(("single-block: all legal starts 0..length-target sampled, "
                    "including the final one",
                    starts == {0, 1, 2, 3}, f"observed starts {sorted(starts)}"))
    # target == length: the whole block is masked (single legal placement)
    m = injected_mask(8, 1.0, "single", np.random.RandomState(0))
    results.append(("single-block: target == length masks the whole block",
                    bool(m.all()), ""))
    # target == 0: empty mask
    m = injected_mask(8, 0.0, "single", np.random.RandomState(0))
    results.append(("single-block: zero fraction masks nothing", bool(~m.any()), ""))
    # tiny block edge case
    m = injected_mask(1, 1.0, "single", np.random.RandomState(0))
    results.append(("single-block: length-1 block handled", bool(m.all()), ""))
    return results


def run_all():
    checks = []
    for fn in (check_whole_gap, check_isolation, check_injection_masks,
               check_fill_isolation, check_persistence_scaler,
               check_conformal_quantile, check_horizon_semantics,
               check_aci_availability, check_single_block_range):
        checks.extend(fn())
    return checks


if __name__ == "__main__":
    checks = run_all()
    n_fail = sum(1 for _, ok, _ in checks if not ok)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL':4s} {name}" + (f"  [{detail}]" if detail and not ok else ""))
    print()
    if n_fail:
        print(f"FAILED: {n_fail} of {len(checks)} methodological checks")
        sys.exit(1)
    print(f"ALL METHODOLOGICAL CHECKS PASSED: {len(checks)}/{len(checks)}")
