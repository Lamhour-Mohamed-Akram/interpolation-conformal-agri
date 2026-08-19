"""Structural and methodological checks on the generated outputs.

This is deliberately NOT a check of the paper's numbers: it asserts no expected
value. It verifies that the pipeline produced what it claims to produce, that
the outputs are internally consistent, and that the protocol's methodological
invariants actually hold on the data the experiments ran on.

Structural checks
-----------------
* every experiment wrote its result file, and every generator wrote its tables
* required columns are present
* no NaN in columns that must be complete
* coverages are probabilities and widths are non-negative
* the experiment grids are complete (all fraction x geometry x method x horizon
  x placement cells present, all horizons present, 20 unique seeds)
* the seed list is exactly the 20 seeds used for the deep case study
* quantities computed by two different scripts agree (the Morocco clean-protocol
  rows from cap_sensitivity at the 30-min cap versus clean_protocol)
* raw data files match their recorded SHA-256, when present

Methodological invariants (see also test_methodology.py, which is run here)
---------------------------------------------------------------------------
* horizon semantics on every real clean-protocol grid: target_bin - origin_bin
  equals exactly the horizon in grid bins (6 h on the hourly grid is 6 bins,
  not 7; 48 h on the 5-min grid is 576 bins, not 577), and target timestamps
  sit exactly h hours after origin timestamps
* ACI calibration-pool legality: every calibration outcome used to seed the
  online state is observable strictly before the first test forecast origin
* whole-gap interpolation on the real Morocco grid: no missing run longer than
  the cap is partially filled, every bounded run within the cap is fully
  interpolated, long runs stay entirely excluded
* target isolation, recomputed from the raw data for every clean-protocol site
  and horizon: max(train target bin) < validation origin boundary and
  max(calibration target bin) < test origin boundary, and the stored counts
  match the recomputation
* controlled-injection isolation on the real substrate: replacing every test
  value with noise leaves the calibration residuals and conformal quantile of
  each fill method unchanged
* requested vs realized injection counts agree in every stored row
* the stored raw persistence forecast equals the feature-scaler inversion of
  the last input-window soil-moisture value (when the arrays are present)
* the canonical conformal quantile matches a hand-sorted toy case

Exits non-zero if any check fails.
Run: python scripts/verify_outputs.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = RES / "tables"
DATA = ROOT / "data"

EXPECTED_SEEDS = [42, 123, 2024, 7, 99, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok, detail: str = ""):
    checks.append((name, bool(ok), str(detail)))


def need_cols(df: pd.DataFrame, cols, label: str):
    missing = [c for c in cols if c not in df.columns]
    check(f"{label}: required columns present", not missing, f"missing {missing}")


# ----------------------------------------------------------- result files --
REQUIRED = [
    "interpolation_audit/daily_row_counts.csv",
    "interpolation_audit/acquisition_summary.csv",
    "interpolation_audit/grid_summary.csv",
    "interpolation_audit/shift_diagnostic.csv",
    "controlled_injection/injection_matrix.csv",
    "delayed_aci/aci_feedback.csv",
    "target_isolation/target_isolation.csv",
    "boundary_robustness/boundary_robustness.csv",
    "uscrn/uscrn_injection.csv",
    "same_model_shift/same_model_shift.csv",
    "cap_sensitivity/cap_sensitivity.csv",
    "clean_protocol/clean_cross_site.csv",
    "deep_20seed/seed_list.json",
    "deep_20seed/morocco_seed_results.csv",
]
for rel in REQUIRED:
    check(f"result present: {rel}", (RES / rel).exists())

# An empty directory carries no information, is not tracked by Git, and would
# therefore differ between a working copy and a fresh clone.
empty_dirs = [d.relative_to(ROOT) for d in RES.rglob("*")
              if d.is_dir() and not any(d.iterdir())]
check("no empty result directories", not empty_dirs, f"{empty_dirs}")

EXPECTED_FIGURES = ["fig1_dataset_split", "fig2_rmse_by_horizon", "fig3_residual_shift",
                    "fig5_rolling_coverage", "fig8_multisite_coverage", "fig_matrix",
                    "fig_mechanism", "fig_uscrn", "fig_leak", "fig_aci"]
for f in EXPECTED_FIGURES:
    check(f"figure present: {f}.pdf", (RES / "figures" / f"{f}.pdf").exists())

EXPECTED_TABLES = ["controlled_injection_distributed.csv", "geometry_resolved.csv",
                   "per_horizon_linear.csv", "same_model_shift.csv", "cap_sensitivity.csv",
                   "clean_protocol_morocco.csv", "delayed_aci.csv", "target_isolation.csv",
                   "uscrn_replication.csv"]
for t in EXPECTED_TABLES:
    check(f"table present: {t}", (TAB / t).exists())

# --------------------------------------------------- controlled injection --
p = RES / "controlled_injection" / "injection_matrix.csv"
if p.exists():
    m = pd.read_csv(p)
    need_cols(m, ["geom", "method", "frac", "seed", "h", "picp", "mpiw",
                  "calib_err", "val_res_p90", "requested_frac", "realized_frac",
                  "requested_n", "realized_n"], "injection_matrix")
    n_expected = (m.geom.nunique() * m.method.nunique() * m.frac.nunique()
                  * m.h.nunique() * m.seed.nunique())
    check("injection grid complete", len(m) == n_expected, f"{len(m)} vs {n_expected}")
    check("injection: 20 placements", m.seed.nunique() == 20, m.seed.nunique())
    check("injection: no NaN", not m[["picp", "mpiw", "calib_err"]].isna().any().any())
    check("injection: PICP in [0,1]", bool(((m.picp >= 0) & (m.picp <= 1)).all()))
    check("injection: widths non-negative", bool((m.mpiw >= 0).all()))
    check("injection: calib_err equals |picp-0.90|",
          bool(np.allclose(m.calib_err, (m.picp - 0.90).abs())))
    check("injection: realized count equals requested count in every row",
          bool((m.realized_n == m.requested_n).all()),
          f"{int((m.realized_n != m.requested_n).sum())} rows differ")
    # integer-count semantics: requested_n follows the documented floor rule,
    # and realized_frac is exactly realized_n / calibration length. The
    # realized fraction is NOT asserted equal to the nominal fraction, which
    # would be false whenever frac * length is not an integer.
    nz = m[m.realized_n > 0]
    if len(nz):
        blk = (nz.realized_n / nz.realized_frac).round().astype(int)
        check("injection: one consistent calibration block length",
              blk.nunique() == 1, f"lengths {sorted(blk.unique())}")
        L_blk = int(blk.iloc[0])
        check("injection: requested_n == floor(frac * block length)",
              bool((m.requested_n == (m.frac * L_blk).astype(int)).all()))
        check("injection: realized_frac == realized_n / block length",
              bool(np.allclose(m.realized_frac, m.realized_n / L_blk)))

# --------------------------------------------------------------- ACI ------
p = RES / "delayed_aci" / "aci_feedback.csv"
if p.exists():
    a = pd.read_csv(p)
    need_cols(a, ["base", "horizon_h", "horizon_steps", "cov_immediate", "cov_delayed",
                  "bias_pp", "n_feedback_consumed", "n_observable_before_last",
                  "first_feedback_index", "median_pending",
                  "frac_before_any_feedback"], "aci_feedback")
    check("ACI: every base x horizon present", len(a) == a.base.nunique() * a.horizon_h.nunique())
    check("ACI: coverages in [0,1]",
          bool(((a.cov_immediate.between(0, 1)) & (a.cov_delayed.between(0, 1))).all()))
    check("ACI: bias equals immediate - observable",
          bool(np.allclose(a.bias_pp, 100 * (a.cov_immediate - a.cov_delayed))))
    check("ACI: horizon equals horizon in 5-min steps",
          bool((a.horizon_steps == a.horizon_h * 12).all()))
    check("ACI: feedback consumed never exceeds what is observable",
          bool((a.n_feedback_consumed <= a.n_observable_before_last).all()))
    check("ACI: feedback-availability fractions in [0,1]",
          bool(a.frac_before_any_feedback.between(0, 1).all()))

# --------------------------------------------------- target isolation -----
p = RES / "target_isolation" / "target_isolation.csv"
if p.exists():
    t = pd.read_csv(p)
    need_cols(t, ["horizon_h", "n_train_leaky", "n_train_isolated", "n_leaking_train_seq",
                  "picp_leaky", "picp_isolated"], "target_isolation")
    check("leakage: isolated split never larger than leaky",
          bool((t.n_train_isolated <= t.n_train_leaky).all()))
    check("leakage: leaking count equals sequences dropped",
          bool((t.n_leaking_train_seq == t.n_train_leaky - t.n_train_isolated).all()))
    check("leakage: leaking count grows with horizon",
          bool(t.sort_values("horizon_h").n_leaking_train_seq.is_monotonic_increasing))
    check("leakage: coverages in [0,1]",
          bool((t.picp_leaky.between(0, 1) & t.picp_isolated.between(0, 1)).all()))

# --------------------------------------------------------------- USCRN ----
p = RES / "uscrn" / "uscrn_injection.csv"
if p.exists():
    u = pd.read_csv(p)
    need_cols(u, ["frac", "horizon_h", "seed", "arm", "picp", "mpiw",
                  "requested_n", "realized_n"], "uscrn_injection")
    check("USCRN: realized count equals requested count in every row",
          bool((u.realized_n == u.requested_n).all()))
    nzu = u[u.realized_n > 0]
    if len(nzu):
        blku = (nzu.realized_n / nzu.realized_frac).round().astype(int)
        check("USCRN: requested_n == floor(frac * block length) and "
              "realized_frac == realized_n / block length",
              blku.nunique() == 1
              and bool((u.requested_n == (u.frac * int(blku.iloc[0])).astype(int)).all())
              and bool(np.allclose(u.realized_frac, u.realized_n / int(blku.iloc[0]))))
    check("USCRN: both arms present", set(u.arm) == {"linear", "gap_aware"})
    check("USCRN grid complete",
          len(u) == u.frac.nunique() * u.horizon_h.nunique() * u.seed.nunique() * u.arm.nunique())
    check("USCRN: 20 placements", u.seed.nunique() == 20)
    check("USCRN: no NaN", not u[["picp", "mpiw"]].isna().any().any())
    zero = u[u.frac == 0.0]
    check("USCRN: arms coincide at 0% injection",
          bool(np.allclose(zero[zero.arm == "linear"].sort_values(["horizon_h", "seed"]).picp.to_numpy(),
                           zero[zero.arm == "gap_aware"].sort_values(["horizon_h", "seed"]).picp.to_numpy())))

# ------------------------------------------------------ same-model shift --
p = RES / "same_model_shift" / "same_model_shift.csv"
if p.exists():
    s = pd.read_csv(p)
    need_cols(s, ["base", "horizon_h", "naive_val_med", "naive_test_med", "naive_ratio",
                  "clean_val_med", "clean_test_med", "clean_ratio"], "same_model_shift")
    check("shift: ratios equal test/val medians (unrounded)",
          bool(np.allclose(s.naive_ratio, s.naive_test_med / s.naive_val_med, rtol=1e-6))
          and bool(np.allclose(s.clean_ratio, s.clean_test_med / s.clean_val_med, rtol=1e-6)))
    check("shift: both bases and all horizons present",
          len(s) == s.base.nunique() * s.horizon_h.nunique())

# ------------------------------------- cap sensitivity vs clean protocol --
cap_p = RES / "cap_sensitivity" / "cap_sensitivity.csv"
cln_p = RES / "clean_protocol" / "clean_cross_site.csv"
if cap_p.exists():
    e8 = pd.read_csv(cap_p)
    need_cols(e8, ["cap_min", "horizon_h", "n_seq", "n_train", "n_cal", "n_test",
                   "n_train_dropped", "n_cal_dropped",
                   "scp_picp", "scp_mpiw", "aci_picp", "aci_mpiw"], "cap_sensitivity")
    check("caps: admissible sequences grow with the cap",
          bool(all(e8[e8.horizon_h == h].sort_values("cap_min").n_seq.is_monotonic_increasing
                   for h in e8.horizon_h.unique())))
    if cln_p.exists():
        cln = pd.read_csv(cln_p)
        mor = cln[cln.site == "Morocco"].set_index("horizon_h")
        c30 = e8[e8.cap_min == 30].set_index("horizon_h")
        same = all(np.isclose(mor.loc[h].scp_picp, c30.loc[h].scp_picp) and
                   np.isclose(mor.loc[h].aci_mpiw, c30.loc[h].aci_mpiw)
                   for h in mor.index)
        check("Morocco clean-protocol rows agree between the two scripts", same)

# --------------------------------------------------------- deep 20 seeds --
p = RES / "deep_20seed" / "seed_list.json"
if p.exists():
    seeds = json.load(open(p))["seeds"]
    check("20 unique seeds", len(seeds) == 20 and len(set(seeds)) == 20, len(seeds))
    check("seed list matches the published list", seeds == EXPECTED_SEEDS)
p = RES / "deep_20seed" / "morocco_seed_results.csv"
if p.exists():
    msr = pd.read_csv(p)
    need_cols(msr, ["site", "seed", "horizon", "model", "rmse", "r2", "converged"],
              "morocco_seed_results")
    check("Morocco seed grid complete (seeds x models x horizons)",
          len(msr) == msr.seed.nunique() * msr.model.nunique() * msr.horizon.nunique())
    check("Morocco: 20 seeds present", msr.seed.nunique() == 20)
    check("Morocco: no NaN RMSE", not msr.rmse.isna().any())
    check("Morocco: all runs converged (no seed excluded)", bool(msr.converged.all()))

# ------------------------------------- methodological invariants (synthetic) --
try:
    from test_methodology import run_all as _run_synthetic
    for name, ok, detail in _run_synthetic():
        check(name, ok, detail)
except Exception as e:  # a failure to even run the suite is itself a failure
    check("synthetic methodological test suite runs", False, repr(e))

# --------------------------- methodological invariants (real data) -----------
# These recompute the protocol's properties from the raw records rather than
# trusting the stored outputs. Each block degrades to a skip message when the
# record it needs is absent (e.g. the non-redistributed Iraq file).

# 1. whole-gap interpolation on the real Morocco grid
try:
    from preprocess import (load_clean, nan_runs, MAX_GAP_BINS,
                            build_horizon, STEPS, HORIZONS_H,
                            TRAIN_RATIO, VAL_RATIO)
    res_m, observed_m, valid_win_m, valid_tgt_m = load_clean()
    runs = nan_runs(~observed_m)
    n_bins = len(observed_m)
    partial, bad_short, bad_long = 0, 0, 0
    for start, stop in runs:
        seg = valid_tgt_m[start:stop]
        if seg.any() and not seg.all():
            partial += 1
        bounded = start > 0 and stop < n_bins
        if stop - start <= MAX_GAP_BINS and bounded and not seg.all():
            bad_short += 1
        if stop - start > MAX_GAP_BINS and seg.any():
            bad_long += 1
    check("whole-gap: no missing run is partially filled (real grid)",
          partial == 0, f"{partial} partial runs")
    check("whole-gap: every bounded run within the cap is fully interpolated",
          bad_short == 0, f"{bad_short} short runs not filled")
    check("whole-gap: every run longer than the cap stays entirely excluded",
          bad_long == 0, f"{bad_long} long runs partly valid")

    # 2. target isolation recomputed for Morocco, every horizon, plus the
    #    stored clean-protocol counts
    from splits import isolated_split
    cln = pd.read_csv(cln_p) if cln_p.exists() else None
    grid_dates_m = res_m.index.to_numpy()
    for h in HORIZONS_H:
        _, y_h, _, _, ob_h, tb_h = build_horizon(res_m, valid_win_m, valid_tgt_m, STEPS[h])
        check(f"horizon: Morocco {h}h target-origin gap is exactly {STEPS[h]} bins",
              bool(np.all(tb_h - ob_h == STEPS[h])),
              f"diffs {np.unique(tb_h - ob_h)}")
        check(f"horizon: Morocco {h}h target timestamps exactly {h}h after origins",
              bool(np.all((grid_dates_m[tb_h] - grid_dates_m[ob_h])
                          == np.timedelta64(h, "h"))))
        m_tr, m_cal, m_te, st = isolated_split(ob_h, tb_h, TRAIN_RATIO, VAL_RATIO)
        ok_tr = (not m_tr.any()) or tb_h[m_tr].max() < st["val_origin_boundary"]
        ok_ca = (not m_cal.any()) or tb_h[m_cal].max() < st["test_origin_boundary"]
        check(f"isolation: Morocco {h}h train targets before validation boundary", ok_tr)
        check(f"isolation: Morocco {h}h calibration targets before test boundary", ok_ca)
        check(f"ACI pool legality: Morocco {h}h calibration outcomes observable "
              "before the first test origin",
              bool(tb_h[m_cal].max() < ob_h[m_te][0]))
        if cln is not None:
            row = cln[(cln.site == "Morocco") & (cln.horizon_h == h)]
            check(f"isolation: Morocco {h}h stored counts match recomputation",
                  len(row) == 1 and int(row.n_cal.iloc[0]) == st["n_cal"]
                  and int(row.n_train.iloc[0]) == st["n_train"],
                  f"stored {row.n_cal.tolist()}, recomputed {st['n_cal']}")

    # the other clean-protocol sites, at every horizon
    from sites import available_sites, load_site, build_sequences
    for (sname, fname, dcol, targ, feats, rule, capb, enc, bnds, smap) in available_sites():
        if sname == "Morocco":
            continue                        # already covered above
        res_s, vw_s, vt_s = load_site(fname, dcol, targ, feats, rule, capb, enc, bnds)
        grid_dates_s = res_s.index.to_numpy()
        for h, steps in smap.items():
            _, y_s, _, ob_s, tb_s = build_sequences(res_s, vw_s, vt_s, feats, targ, steps)
            check(f"horizon: {sname} {h}h target-origin gap is exactly {steps} bins",
                  bool(np.all(tb_s - ob_s == steps)))
            check(f"horizon: {sname} {h}h target timestamps exactly {h}h after origins",
                  bool(np.all((grid_dates_s[tb_s] - grid_dates_s[ob_s])
                              == np.timedelta64(h, "h"))))
            m_tr, m_cal, m_te, st = isolated_split(ob_s, tb_s, 0.6, 0.2)
            ok_tr = (not m_tr.any()) or tb_s[m_tr].max() < st["val_origin_boundary"]
            ok_ca = (not m_cal.any()) or tb_s[m_cal].max() < st["test_origin_boundary"]
            check(f"isolation: {sname} {h}h train/calibration targets isolated",
                  ok_tr and ok_ca)
            check(f"ACI pool legality: {sname} {h}h calibration outcomes observable "
                  "before the first test origin",
                  bool(tb_s[m_cal].max() < ob_s[m_te][0]))
except Exception as e:
    check("whole-gap/isolation real-data checks run", False, repr(e))

# 3. controlled-injection isolation on the real substrate: arbitrary test
#    values must not move any calibration residual or conformal quantile
try:
    from controlled_injection import load_series as _load_mendeley, TR as _TR, VA as _VA
    from injection import injected_mask, fill_prefix
    from conformal import scp_quantile, ALPHA
    s_real = _load_mendeley()
    L = len(s_real)
    tr_i, va_i = int(L * _TR), int(L * _VA)
    rng_chk = np.random.RandomState(0)
    s_pert = s_real.copy()
    s_pert[va_i:] = rng_chk.uniform(-1e6, 1e6, L - va_i)
    inj_full = np.zeros(L, bool)
    inj_full[tr_i:va_i] = injected_mask(va_i - tr_i, 0.3, "distributed",
                                        np.random.RandomState(0))
    for method in ("linear", "ffill", "spline", "none"):
        f1 = fill_prefix(s_real[:va_i], inj_full[:va_i], method)
        f2 = fill_prefix(s_pert[:va_i], inj_full[:va_i], method)
        ok_all = True
        for h in (6, 48):
            oc = np.arange(tr_i, va_i - h)
            if method == "none":
                okm = ~inj_full[oc] & ~inj_full[oc + h]
                r1 = np.abs(s_real[oc[okm] + h] - s_real[oc[okm]])
                r2 = np.abs(s_pert[oc[okm] + h] - s_pert[oc[okm]])
            else:
                r1 = np.abs(f1[oc + h] - f1[oc])
                r2 = np.abs(f2[oc + h] - f2[oc])
            ok_all &= bool(np.allclose(r1, r2, equal_nan=True)
                           and np.isclose(scp_quantile(r1, ALPHA), scp_quantile(r2, ALPHA)))
        check(f"injection isolation: {method} fill unchanged by arbitrary test values",
              ok_all)
except Exception as e:
    check("injection-isolation real-data check runs", False, repr(e))

# 4. persistence: the stored raw forecast equals the feature-scaler inversion
#    of the last input-window soil-moisture value (needs the working arrays,
#    which are regenerable but not shipped; skipped when absent)
NAIVE = RES / "naive_pipeline"
pers_files = [NAIVE / "X_val.npy", NAIVE / "feature_scaler.pkl",
              NAIVE / "pred_persistence_val.npy"]
if all(f.exists() for f in pers_files):
    import pickle
    with open(NAIVE / "feature_scaler.pkl", "rb") as fh:
        fsc_chk = pickle.load(fh)
    X_va_chk = np.load(NAIVE / "X_val.npy")
    pers_chk = np.load(NAIVE / "pred_persistence_val.npy")
    ti_chk = 2                            # humiditysol is the third feature column
    raw_last = (X_va_chk[:, -1, ti_chk].astype(np.float64)
                * (fsc_chk.data_max_[ti_chk] - fsc_chk.data_min_[ti_chk])
                + fsc_chk.data_min_[ti_chk])
    check("persistence: stored forecast equals raw last-window value (all horizons)",
          bool(np.allclose(pers_chk, raw_last[:, None], atol=1e-3)))
    meta_naive = json.load(open(NAIVE / "meta.json"))
    check("persistence: meta.json declares raw units",
          meta_naive.get("persistence_units") == "raw_percent")
else:
    print("(persistence array check skipped: working arrays not present — "
          "regenerate with train_deep_models.py --recompute-outputs)")

# ------------------------------------------------------------- checksums --
EXTERNAL = {"iraq_IoTProcessed_Data.csv"}


def sha256_of(fh) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


sums = DATA / "checksums.sha256"
if sums.exists():
    n_ok, absent = 0, []
    for line in sums.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split()[0], line.split()[-1].lstrip("*")
        f = DATA / name
        if f.exists():
            with open(f, "rb") as fh:
                check(f"checksum: {name}", sha256_of(fh) == digest)
            n_ok += 1
        elif f.with_name(f.name + ".gz").exists():
            # the record is distributed compressed; verify the decompressed
            # content really is the file the experiments were run on
            import gzip
            with gzip.open(f.with_name(f.name + ".gz"), "rb") as fh:
                check(f"checksum: {name} (decompressed from {name}.gz)",
                      sha256_of(fh) == digest)
            n_ok += 1
        else:
            absent.append(name)
    print(f"(checksums: {n_ok} verified)")
    for name in absent:
        note = "external, see data/README.md" if name in EXTERNAL else "not present locally"
        print(f"(not checked: {name} — {note})")
    print()

# ----------------------------------------------------------------- report --
n_fail = sum(1 for _, ok, _ in checks if not ok)
for name, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL':4s} {name}" + (f"  [{detail}]" if detail and not ok else ""))
print()
if n_fail:
    print(f"FAILED: {n_fail} of {len(checks)} checks")
    sys.exit(1)
print(f"ALL CHECKS PASSED: {len(checks)}/{len(checks)}")
