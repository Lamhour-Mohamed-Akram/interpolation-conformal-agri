"""Paired inferential statistics for the reported coverage differences.

Scientific purpose
------------------
The controlled experiments report differences in measured coverage as point
estimates (means or medians over 20 gap placements, or single deterministic
values on one test stream). This script attaches paired significance tests and
confidence intervals to those differences, using the pairing structure that the
experimental designs already contain:

A. Controlled injection (Johannesburg, results/controlled_injection).
   For a given geometry and fraction, placement seed s produces the SAME
   injected gap mask for every fill method (the mask is drawn from
   RandomState(seed) before the method is applied), so fill methods are
   paired by placement. Two paired contrasts are tested per cell, on the
   placement-level coverage averaged over the four horizons (the unit used in
   the manuscript's injection table):
     (i)  fill method vs. the gap-aware exclusion arm at the same placement;
     (ii) fill method at fraction f vs. the 0 % baseline (a constant, since
          at 0 % every placement coincides), i.e. a one-sample test on the
          per-placement change.
   Test: two-sided Wilcoxon signed-rank (exact distribution, n = 20).
   Effect size: median paired difference in percentage points and a 95 %
   paired percentile-bootstrap interval of the mean difference (10,000
   resamples). Multiplicity: Holm step-down within each contrast family.
   Per-horizon contrasts at 50 % injection are tested in the same way.

B. USCRN replication (results/uscrn): linear fill vs. gap-aware exclusion at
   the same placement, per fraction and horizon, same test and effect sizes.

C. Deterministic defect experiments on the gap-cleaned Morocco stream
   (target-boundary leakage; immediate vs. availability-aware ACI feedback).
   These have no placements: each arm is a single run, but both arms score
   the SAME test forecasts, so their per-forecast coverage indicators are
   paired. Test: exact McNemar test on the discordant pairs. Interval: a
   circular moving-block bootstrap (block length = horizon steps, capped at
   n/5; 10,000 resamples) of the coverage difference, which respects the
   serial dependence of overlapping multi-horizon forecasts. The arms are
   recomputed here with the same shared implementations the original
   experiments use (conformal.py, splits.py, preprocess.py), and the
   recomputed coverages are asserted equal to the published CSV values. The
   leaky and isolated test blocks of the target-isolation experiment start at
   int(0.8 n) and int(0.6 n) + int(0.2 n) respectively, which can differ by
   one row; the paired statistics use the common forecasts.

Input       results/controlled_injection/injection_matrix.csv
            results/uscrn/uscrn_injection.csv
            results/target_isolation/target_isolation.csv (for assertion)
            results/delayed_aci/aci_feedback.csv (for assertion)
            data/data.csv(.gz) via preprocess.load_clean()
Output      results/significance/injection_vs_gapaware.csv
            results/significance/injection_vs_baseline.csv
            results/significance/injection_per_horizon_50pct.csv
            results/significance/uscrn_paired.csv
            results/significance/defects_paired.csv
            results/significance/deep_vs_gapaware.csv      (section D, if the
            results/significance/deep_vs_persistence.csv    deep run is present)
Manuscript  the inferential columns and sentences added at resubmission
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import aci, scp_quantile, ALPHA                       # noqa: E402
from preprocess import (load_clean, build_horizon, STEPS, HORIZONS_H,  # noqa: E402
                        TRAIN_RATIO, VAL_RATIO)
from splits import isolated_split                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "significance"
N_BOOT = 10_000
BOOT_SEED = 0


# ----------------------------------------------------------------- helpers --
def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def wilcoxon_paired(d: np.ndarray) -> tuple[float, float]:
    """Two-sided exact Wilcoxon signed-rank p-value and rank-biserial r.

    Zero differences are dropped (Wilcoxon's original treatment) so the exact
    distribution applies; if every difference is zero the test is undefined
    and p = 1 is returned.
    """
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0, 0.0
    res = stats.wilcoxon(d, alternative="two-sided", method="exact")
    ranks = stats.rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    rb = (r_plus - r_minus) / ranks.sum()
    return float(res.pvalue), float(rb)


def boot_ci_mean(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95 % percentile bootstrap interval of the mean of paired differences."""
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test on paired binary outcomes a, b.

    Returns (n_10, n_01, p): n_10 = covered by a only, n_01 = covered by b
    only, p = two-sided exact binomial p-value on the discordant pairs.
    """
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum())
    n_disc = n10 + n01
    if n_disc == 0:
        return n10, n01, 1.0
    p = float(stats.binomtest(min(n10, n01), n_disc, 0.5, alternative="two-sided").pvalue)
    return n10, n01, p


def block_boot_diff(a: np.ndarray, b: np.ndarray, block: int,
                    rng: np.random.Generator) -> tuple[float, float]:
    """Circular moving-block bootstrap 95 % CI of mean(a) - mean(b) in pp."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = len(a)
    d = a - b
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    offs = np.arange(block)
    idx = (starts[:, :, None] + offs[None, None, :]).reshape(N_BOOT, -1)[:, :n] % n
    means = d[idx].mean(axis=1) * 100
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------- A. controlled injection --
def section_a(rng: np.random.Generator) -> None:
    df = pd.read_csv(ROOT / "results" / "controlled_injection" / "injection_matrix.csv")
    # placement-level coverage averaged over horizons (the manuscript's unit)
    pl = (df.groupby(["geom", "method", "frac", "seed"])["picp"].mean()
            .rename("picp").reset_index())
    geoms = ["single", "distributed", "isolated"]
    fills = ["linear", "ffill", "spline"]
    fracs = sorted(f for f in pl.frac.unique() if f > 0)

    # (i) fill vs gap-aware at the same placement
    rows = []
    for g in geoms:
        for m in fills:
            for f in fracs:
                a = pl[(pl.geom == g) & (pl.method == m) & (pl.frac == f)].sort_values("seed")
                b = pl[(pl.geom == g) & (pl.method == "none") & (pl.frac == f)].sort_values("seed")
                assert (a.seed.values == b.seed.values).all() and len(a) == 20
                d = 100 * (a.picp.values - b.picp.values)          # pp, fill - gap-aware
                p, rb = wilcoxon_paired(d)
                lo, hi = boot_ci_mean(d, rng)
                rows.append(dict(geom=g, method=m, frac=f, n=len(d),
                                 mean_fill=100 * a.picp.mean(), mean_gapaware=100 * b.picp.mean(),
                                 median_diff_pp=float(np.median(d)), mean_diff_pp=float(d.mean()),
                                 ci_lo_pp=lo, ci_hi_pp=hi, wilcoxon_p=p, rank_biserial=rb,
                                 n_positive=int((d > 0).sum()), n_negative=int((d < 0).sum())))
    A1 = pd.DataFrame(rows)
    A1["holm_p"] = holm(A1.wilcoxon_p.values)
    A1.to_csv(OUT / "injection_vs_gapaware.csv", index=False)

    # (ii) fill (and gap-aware) at fraction f vs the 0 % baseline constant
    rows = []
    for g in geoms:
        for m in fills + ["none"]:
            base = pl[(pl.geom == g) & (pl.method == m) & (pl.frac == 0.0)].picp
            assert base.nunique() == 1, "0 % placements must coincide"
            b0 = float(base.iloc[0])
            for f in fracs:
                a = pl[(pl.geom == g) & (pl.method == m) & (pl.frac == f)].sort_values("seed")
                d = 100 * (a.picp.values - b0)
                p, rb = wilcoxon_paired(d)
                lo, hi = boot_ci_mean(d, rng)
                rows.append(dict(geom=g, method=m, frac=f, n=len(d), baseline_cov=100 * b0,
                                 mean_cov=100 * a.picp.mean(),
                                 median_diff_pp=float(np.median(d)), mean_diff_pp=float(d.mean()),
                                 ci_lo_pp=lo, ci_hi_pp=hi, wilcoxon_p=p, rank_biserial=rb,
                                 n_positive=int((d > 0).sum()), n_negative=int((d < 0).sum())))
    A2 = pd.DataFrame(rows)
    A2["holm_p"] = holm(A2.wilcoxon_p.values)
    A2.to_csv(OUT / "injection_vs_baseline.csv", index=False)

    # (iii) per horizon at 50 % injection: fill vs gap-aware, all geometries
    rows = []
    for g in geoms:
        for m in fills:
            for h in sorted(df.h.unique()):
                a = df[(df.geom == g) & (df.method == m) & (df.frac == 0.5) & (df.h == h)].sort_values("seed")
                b = df[(df.geom == g) & (df.method == "none") & (df.frac == 0.5) & (df.h == h)].sort_values("seed")
                assert (a.seed.values == b.seed.values).all() and len(a) == 20
                d = 100 * (a.picp.values - b.picp.values)
                p, rb = wilcoxon_paired(d)
                lo, hi = boot_ci_mean(d, rng)
                rows.append(dict(geom=g, method=m, horizon_h=int(h), n=len(d),
                                 mean_fill=100 * a.picp.mean(), mean_gapaware=100 * b.picp.mean(),
                                 median_diff_pp=float(np.median(d)), mean_diff_pp=float(d.mean()),
                                 ci_lo_pp=lo, ci_hi_pp=hi, wilcoxon_p=p, rank_biserial=rb))
    A3 = pd.DataFrame(rows)
    A3["holm_p"] = holm(A3.wilcoxon_p.values)
    A3.to_csv(OUT / "injection_per_horizon_50pct.csv", index=False)

    print("A. Controlled injection, distributed geometry: fill vs gap-aware (paired by placement)")
    print(f"{'method':>8} {'frac':>5} {'fill':>6} {'gap-aw':>7} {'med d':>7} {'95% CI (mean d)':>18} "
          f"{'p (exact)':>10} {'Holm p':>9} {'r_rb':>6}")
    for _, r in A1[A1.geom == "distributed"].iterrows():
        print(f"{r.method:>8} {r.frac:5.1f} {r.mean_fill:6.1f} {r.mean_gapaware:7.1f} "
              f"{r.median_diff_pp:+7.2f} [{r.ci_lo_pp:+6.2f}, {r.ci_hi_pp:+6.2f}] "
              f"{r.wilcoxon_p:10.2e} {r.holm_p:9.2e} {r.rank_biserial:+6.2f}")
    print()


# --------------------------------------------------- B. USCRN replication --
def section_b(rng: np.random.Generator) -> None:
    df = pd.read_csv(ROOT / "results" / "uscrn" / "uscrn_injection.csv")
    rows = []
    for f in sorted(x for x in df.frac.unique() if x > 0):
        for h in sorted(df.horizon_h.unique()):
            a = df[(df.frac == f) & (df.horizon_h == h) & (df.arm == "linear")].sort_values("seed")
            b = df[(df.frac == f) & (df.horizon_h == h) & (df.arm == "gap_aware")].sort_values("seed")
            assert (a.seed.values == b.seed.values).all() and len(a) == 20
            d = 100 * (a.picp.values - b.picp.values)
            p, rb = wilcoxon_paired(d)
            lo, hi = boot_ci_mean(d, rng)
            base = df[(df.frac == 0.0) & (df.horizon_h == h) & (df.arm == "linear")].picp
            assert base.nunique() == 1
            d0 = 100 * (a.picp.values - float(base.iloc[0]))
            p0, rb0 = wilcoxon_paired(d0)
            lo0, hi0 = boot_ci_mean(d0, rng)
            rows.append(dict(frac=f, horizon_h=int(h), n=len(d),
                             mean_linear=100 * a.picp.mean(), mean_gapaware=100 * b.picp.mean(),
                             baseline_cov=100 * float(base.iloc[0]),
                             median_diff_pp=float(np.median(d)), mean_diff_pp=float(d.mean()),
                             ci_lo_pp=lo, ci_hi_pp=hi, wilcoxon_p=p, rank_biserial=rb,
                             vs0_median_diff_pp=float(np.median(d0)), vs0_ci_lo_pp=lo0,
                             vs0_ci_hi_pp=hi0, vs0_wilcoxon_p=p0))
    B = pd.DataFrame(rows)
    B["holm_p"] = holm(B.wilcoxon_p.values)
    B["vs0_holm_p"] = holm(B.vs0_wilcoxon_p.values)
    B.to_csv(OUT / "uscrn_paired.csv", index=False)
    print("B. USCRN: linear vs gap-aware (paired by placement)")
    print(f"{'frac':>5} {'H':>3} {'linear':>7} {'gap-aw':>7} {'med d':>7} {'95% CI':>18} {'Holm p':>9}")
    for _, r in B.iterrows():
        print(f"{r.frac:5.1f} {int(r.horizon_h):3d} {r.mean_linear:7.1f} {r.mean_gapaware:7.1f} "
              f"{r.median_diff_pp:+7.2f} [{r.ci_lo_pp:+6.2f}, {r.ci_hi_pp:+6.2f}] {r.holm_p:9.2e}")
    print()


# -------------------------------------------- C. deterministic defect arms --
def section_c(rng: np.random.Generator) -> None:
    ti_pub = pd.read_csv(ROOT / "results" / "target_isolation" / "target_isolation.csv")
    aci_pub = pd.read_csv(ROOT / "results" / "delayed_aci" / "aci_feedback.csv")
    res, _, valid_win, valid_tgt = load_clean()
    rows = []
    for h in HORIZONS_H:
        X, y, pers, _, ob, tb = build_horizon(res, valid_win, valid_tgt, STEPS[h])
        n = len(y)
        tr, va = int(n * TRAIN_RATIO), int(n * (TRAIN_RATIO + VAL_RATIO))
        block = int(min(STEPS[h], max(1, (n - va) // 5)))

        # --- target-boundary leakage: leaky vs isolated (ridge base, SCP) ---
        leaky_tr = np.zeros(n, bool); leaky_tr[:tr] = True
        rg = Ridge(alpha=1.0).fit(X[leaky_tr].reshape(leaky_tr.sum(), -1), y[leaky_tr])
        pred_leaky = rg.predict(X.reshape(n, -1)).astype(np.float32)
        q = scp_quantile(y[tr:va] - pred_leaky[tr:va], ALPHA)
        cov_leaky = (y[va:] >= pred_leaky[va:] - q) & (y[va:] <= pred_leaky[va:] + q)

        m_tr, m_cal, m_te, st = isolated_split(ob, tb, TRAIN_RATIO, VAL_RATIO)
        rg2 = Ridge(alpha=1.0).fit(X[m_tr].reshape(m_tr.sum(), -1), y[m_tr])
        pred_iso = rg2.predict(X.reshape(n, -1)).astype(np.float32)
        q2 = scp_quantile(y[m_cal] - pred_iso[m_cal], ALPHA)
        cov_iso = (y[m_te] >= pred_iso[m_te] - q2) & (y[m_te] <= pred_iso[m_te] + q2)

        pub = ti_pub[ti_pub.horizon_h == h].iloc[0]
        assert abs(cov_leaky.mean() - pub.picp_leaky) < 1e-9, (h, cov_leaky.mean(), pub.picp_leaky)
        assert abs(cov_iso.mean() - pub.picp_isolated) < 1e-9, (h, cov_iso.mean(), pub.picp_isolated)
        # The two arms' test blocks start at int(0.8 n) and int(0.6 n) + int(0.2 n)
        # respectively, which can differ by one row; pair on the common forecasts.
        idx_leaky = np.arange(va, n)
        idx_iso = np.where(m_te)[0]
        common = np.intersect1d(idx_leaky, idx_iso)
        assert len(common) >= min(len(idx_leaky), len(idx_iso)) - 1
        a = cov_leaky[np.searchsorted(idx_leaky, common)]
        b = cov_iso[np.searchsorted(idx_iso, common)]
        n10, n01, p = mcnemar_exact(a, b)
        lo, hi = block_boot_diff(a, b, block, rng)
        rows.append(dict(experiment="target_boundary_leakage", base="ridge", horizon_h=h,
                         n_test=int(len(common)), cov_arm1=100 * a.mean(),
                         cov_arm2=100 * b.mean(), arm1="leaky", arm2="isolated",
                         diff_pp=100 * (a.mean() - b.mean()),
                         n_arm1_only=n10, n_arm2_only=n01, mcnemar_p=p,
                         block_len=block, blockboot_ci_lo_pp=lo, blockboot_ci_hi_pp=hi))

        # --- immediate vs availability-aware ACI (persistence and ridge) ---
        rg3 = Ridge(alpha=1.0).fit(X[m_tr].reshape(m_tr.sum(), -1), y[m_tr])
        ridge_pred = rg3.predict(X.reshape(n, -1)).astype(np.float32)
        for base, pred in [("persistence", pers), ("ridge", ridge_pred)]:
            res_cal = y[m_cal] - pred[m_cal]
            _, _, c_imm = aci(res_cal, pred[m_te], y[m_te], feedback_mode="immediate",
                              return_traces=True)
            _, _, c_obs = aci(res_cal, pred[m_te], y[m_te], origin_bin=ob[m_te],
                              target_bin=tb[m_te], feedback_mode="observable",
                              return_traces=True)
            pub = aci_pub[(aci_pub.base == base) & (aci_pub.horizon_h == h)].iloc[0]
            assert abs(c_imm.mean() - pub.cov_immediate) < 1e-9, (base, h)
            assert abs(c_obs.mean() - pub.cov_delayed) < 1e-9, (base, h)
            n10, n01, p = mcnemar_exact(c_imm, c_obs)
            lo, hi = block_boot_diff(c_imm, c_obs, block, rng)
            rows.append(dict(experiment="premature_aci_feedback", base=base, horizon_h=h,
                             n_test=int(len(c_imm)), cov_arm1=100 * c_imm.mean(),
                             cov_arm2=100 * c_obs.mean(), arm1="immediate", arm2="observable",
                             diff_pp=100 * (c_imm.mean() - c_obs.mean()),
                             n_arm1_only=n10, n_arm2_only=n01, mcnemar_p=p,
                             block_len=block, blockboot_ci_lo_pp=lo, blockboot_ci_hi_pp=hi))
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "defects_paired.csv", index=False)
    print("C. Deterministic defects: paired per-forecast coverage (McNemar exact; block bootstrap)")
    print(f"{'experiment':>24} {'base':>11} {'H':>3} {'n':>4} {'arm1':>6} {'arm2':>6} {'diff':>6} "
          f"{'n10/n01':>9} {'McNemar p':>10} {'block CI':>18}")
    for _, r in C.iterrows():
        print(f"{r.experiment:>24} {r.base:>11} {int(r.horizon_h):3d} {int(r.n_test):4d} "
              f"{r.cov_arm1:6.1f} {r.cov_arm2:6.1f} {r.diff_pp:+6.1f} "
              f"{int(r.n_arm1_only):4d}/{int(r.n_arm2_only):<4d} {r.mcnemar_p:10.2e} "
              f"[{r.blockboot_ci_lo_pp:+6.1f}, {r.blockboot_ci_hi_pp:+6.1f}]")
    print()


# ------------------------------------ D. trained-forecaster replication --
def section_d(rng: np.random.Generator) -> None:
    """Paired inference for controlled_injection_deep.py, and a paired
    base-vs-base comparison of the linear-fill effect at identical placements.

    The LSTM family is summarised per placement by the mean over its five
    model seeds before testing, so the unit remains the placement (n = 20)
    and initialization variability is averaged out rather than counted as
    replication. Only geometry/fraction cells in which every placement has a
    defined gap-aware value are tested.
    """
    path = ROOT / "results" / "controlled_injection_deep" / "injection_matrix_deep.csv"
    if not path.exists():
        print("D. skipped: results/controlled_injection_deep/ not present")
        return
    df = pd.read_csv(path)
    df["family"] = np.where(df.model == "ridge", "ridge", "lstm")
    pl = (df.groupby(["family", "geom", "method", "frac", "seed"])["picp"].mean()
            .rename("picp").reset_index())
    pers = pd.read_csv(ROOT / "results" / "controlled_injection" / "injection_matrix.csv")
    pers = (pers.groupby(["geom", "method", "frac", "seed"])["picp"].mean()
                .rename("picp").reset_index())
    pers["family"] = "persistence"
    geoms = ["single", "distributed", "isolated"]
    fills = ["linear", "ffill", "spline"]
    fracs = sorted(f for f in pl.frac.unique() if f > 0)

    rows = []
    for fam in ("ridge", "lstm"):
        for g in geoms:
            for m in fills:
                for f in fracs:
                    a = pl[(pl.family == fam) & (pl.geom == g) & (pl.method == m) & (pl.frac == f)].sort_values("seed")
                    b = pl[(pl.family == fam) & (pl.geom == g) & (pl.method == "none") & (pl.frac == f)].sort_values("seed")
                    if len(a) != 20 or len(b) != 20 or a.picp.isna().any() or b.picp.isna().any():
                        continue
                    d = 100 * (a.picp.values - b.picp.values)
                    p, rb = wilcoxon_paired(d)
                    lo, hi = boot_ci_mean(d, rng)
                    rows.append(dict(family=fam, geom=g, method=m, frac=f, n=len(d),
                                     mean_fill=100 * a.picp.mean(), mean_gapaware=100 * b.picp.mean(),
                                     median_diff_pp=float(np.median(d)), mean_diff_pp=float(d.mean()),
                                     ci_lo_pp=lo, ci_hi_pp=hi, wilcoxon_p=p, rank_biserial=rb))
    D1 = pd.DataFrame(rows)
    D1["holm_p"] = holm(D1.wilcoxon_p.values)
    D1.to_csv(OUT / "deep_vs_gapaware.csv", index=False)

    # base-vs-base: is the linear-fill effect (fill - gap-aware, pp) different
    # between two bases at the same placements?
    rows = []
    allpl = pd.concat([pl, pers], ignore_index=True)
    def effect(fam, g, m, f):
        a = allpl[(allpl.family == fam) & (allpl.geom == g) & (allpl.method == m) & (allpl.frac == f)].sort_values("seed")
        b = allpl[(allpl.family == fam) & (allpl.geom == g) & (allpl.method == "none") & (allpl.frac == f)].sort_values("seed")
        if len(a) != 20 or len(b) != 20 or a.picp.isna().any() or b.picp.isna().any():
            return None
        return 100 * (a.picp.values - b.picp.values)
    for g in geoms:
        for m in fills:
            for f in fracs:
                e = {fam: effect(fam, g, m, f) for fam in ("persistence", "ridge", "lstm")}
                for fam1, fam2 in (("lstm", "persistence"), ("ridge", "persistence"), ("lstm", "ridge")):
                    if e[fam1] is None or e[fam2] is None:
                        continue
                    d = e[fam1] - e[fam2]
                    p, rb = wilcoxon_paired(d)
                    lo, hi = boot_ci_mean(d, rng)
                    rows.append(dict(geom=g, method=m, frac=f, base1=fam1, base2=fam2,
                                     effect_base1_pp=float(np.median(e[fam1])),
                                     effect_base2_pp=float(np.median(e[fam2])),
                                     median_diff_pp=float(np.median(d)), ci_lo_pp=lo, ci_hi_pp=hi,
                                     wilcoxon_p=p, rank_biserial=rb,
                                     same_sign=bool(np.sign(np.median(e[fam1])) == np.sign(np.median(e[fam2])))))
    D2 = pd.DataFrame(rows)
    D2["holm_p"] = holm(D2.wilcoxon_p.values)
    D2.to_csv(OUT / "deep_vs_persistence.csv", index=False)

    print("D. Trained forecasters, distributed geometry: fill vs protocol arm (paired by placement)")
    print(f"{'family':>6} {'method':>8} {'frac':>5} {'fill':>6} {'gap-aw':>7} {'med d':>7} {'95% CI':>18} {'Holm p':>9}")
    for _, r in D1[D1.geom == "distributed"].iterrows():
        print(f"{r.family:>6} {r.method:>8} {r.frac:5.1f} {r.mean_fill:6.1f} {r.mean_gapaware:7.1f} "
              f"{r.median_diff_pp:+7.2f} [{r.ci_lo_pp:+6.2f}, {r.ci_hi_pp:+6.2f}] {r.holm_p:9.2e}")
    print("   base-vs-base (linear, distributed): effect difference at identical placements")
    for _, r in D2[(D2.geom == "distributed") & (D2.method == "linear")].iterrows():
        print(f"   {r.base1:>11} vs {r.base2:<11} frac {r.frac:3.1f}: {r.effect_base1_pp:+6.2f} vs "
              f"{r.effect_base2_pp:+6.2f} pp, diff {r.median_diff_pp:+6.2f} [{r.ci_lo_pp:+6.2f}, {r.ci_hi_pp:+6.2f}], Holm p {r.holm_p:.2e}")
    print()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    section_a(rng)
    section_b(rng)
    section_c(rng)
    section_d(rng)
    print(f"wrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
