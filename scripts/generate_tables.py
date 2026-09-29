"""Regenerate the manuscript tables from the stored experiment outputs.

Every value is read from a result file under results/ and formatted through the
shared helpers below, so a quantity that appears in more than one table is read
from one source and rounded once. No manuscript number is written into this
file.

Rounding
--------
Displayed values are rounded exactly once, from the full-precision stored value.
Ratios are computed from unrounded medians and only then rounded, so a ratio
never disagrees with the medians shown beside it.

Shared quantities
-----------------
The Morocco clean-protocol table and the cap-sensitivity table report the same
calibration run at the 30-minute cap. Both are therefore read from
results/cap_sensitivity/cap_sensitivity.csv, so they cannot drift apart.
results/clean_protocol/clean_cross_site.csv independently recomputes the same
Morocco rows and is used for the cross-site table; verify_outputs.py checks that
the two agree.

Run: python scripts/generate_tables.py
Outputs: results/tables/*.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10
NOMINAL = 1 - ALPHA
written: list[str] = []


def fmt1(x) -> str:
    """One-decimal display, a single rounding step from the stored value."""
    return f"{float(x):.1f}"


def pct1(x, already_pct: bool = False) -> str:
    v = float(x) if already_pct else 100.0 * float(x)
    return f"{v:.1f}"


def write(df: pd.DataFrame, name: str):
    df.to_csv(OUT / name, index=False)
    written.append(name)


def exists(p: Path) -> bool:
    if p.exists():
        return True
    print(f"  [skip] {p.relative_to(ROOT)} not found")
    return False


# ------------------------------------------------- controlled injection ----
inj = RES / "controlled_injection" / "injection_matrix.csv"
if exists(inj):
    m = pd.read_csv(inj)

    def placement_ci(method: str, frac: float, geom: str = "distributed"):
        """Placement-level 95% CI.

        The 20 random placements are the independent replicates; the four
        horizons within a placement are not. The mean over horizons is therefore
        taken within each placement first, and the normal-approximation interval
        is formed over the 20 placement means.
        """
        g = m[(m.geom == geom) & (m.method == method) & (m.frac == frac)]
        per = g.groupby("seed").calib_err.mean() * 100
        sd = per.std(ddof=1)
        half = 1.96 * sd / np.sqrt(len(per))
        return per.mean(), per.mean() - half, per.mean() + half, sd

    rows = []
    for frac in sorted(m.frac.unique()):
        row = {"fraction_pct": int(round(frac * 100))}
        for meth in ("linear", "ffill", "spline", "none"):
            mean, lo, hi, sd = placement_ci(meth, frac)
            g = m[(m.geom == "distributed") & (m.method == meth) & (m.frac == frac)]
            row[f"{meth}_err_pp"] = fmt1(mean)
            # at 0% injection every placement is identical, so no interval is defined
            row[f"{meth}_ci"] = "---" if sd < 1e-9 else f"[{fmt1(lo)},{fmt1(hi)}]"
            row[f"{meth}_mpiw"] = f"{g.mpiw.mean():.0f}"
        rows.append(row)
    write(pd.DataFrame(rows), "controlled_injection_distributed.csv")

    rows = []
    for geom in ("single", "isolated"):
        for frac in sorted(m.frac.unique()):
            if frac == 0.0:
                continue
            row = {"geometry": geom, "fraction_pct": int(round(frac * 100))}
            for meth in ("linear", "ffill", "spline", "none"):
                g = m[(m.geom == geom) & (m.method == meth) & (m.frac == frac)]
                row[f"{meth}_err_pp"] = fmt1(100 * g.calib_err.mean())
                row[f"{meth}_mpiw"] = f"{g.mpiw.mean():.0f}"
            rows.append(row)
    write(pd.DataFrame(rows), "geometry_resolved.csv")

    d = m[(m.geom == "distributed") & (m.method == "linear")]
    ph = d.pivot_table(index="frac", columns="h", values="picp", aggfunc="mean") * 100
    ph = ph.round(1).reset_index()
    ph["frac"] = (ph["frac"] * 100).astype(int)
    write(ph.rename(columns={"frac": "fraction_pct"}), "per_horizon_linear.csv")

    rows = []
    for geom in ("single", "distributed", "isolated"):
        g = m[(m.geom == geom) & (m.method == "linear") & (m.frac == 0.5)]
        gf = m[(m.geom == geom) & (m.method == "ffill") & (m.frac == 0.5)]
        gs = m[(m.geom == geom) & (m.method == "spline") & (m.frac == 0.5)]
        rows.append(dict(geometry=geom, picp_linear=pct1(g.picp.mean()),
                         err_linear_pp=fmt1(100 * g.calib_err.mean()),
                         err_ffill_pp=fmt1(100 * gf.calib_err.mean()),
                         width_linear=f"{g.mpiw.mean():.0f}",
                         width_spline=f"{gs.mpiw.mean():.0f}"))
    write(pd.DataFrame(rows), "geometry_method_at_50pct.csv")

# ------------------------------------------------------ same-model shift ----
sm = RES / "same_model_shift" / "same_model_shift.csv"
if exists(sm):
    e7 = pd.read_csv(sm)
    rows = []
    for _, r in e7.iterrows():
        rows.append(dict(
            base=r.base, horizon_h=int(r.horizon_h),
            naive_val_median_raw=r.naive_val_med, naive_test_median_raw=r.naive_test_med,
            naive_ratio_raw=r.naive_ratio,
            naive_val_median_display=fmt1(r.naive_val_med) if r.naive_val_med >= 0.1
            else f"{r.naive_val_med:.2f}",
            naive_test_median_display=fmt1(r.naive_test_med),
            naive_ratio_display=fmt1(r.naive_ratio),
            clean_val_median_raw=r.clean_val_med, clean_test_median_raw=r.clean_test_med,
            clean_ratio_raw=r.clean_ratio,
            clean_val_median_display=fmt1(r.clean_val_med),
            clean_test_median_display=fmt1(r.clean_test_med),
            clean_ratio_display=fmt1(r.clean_ratio)))
    write(pd.DataFrame(rows), "same_model_shift.csv")

# ------------------------------------- cap sensitivity + clean protocol -----
cap = RES / "cap_sensitivity" / "cap_sensitivity.csv"
if exists(cap):
    e8 = pd.read_csv(cap)
    write(pd.DataFrame([dict(cap_min=int(r.cap_min), horizon_h=int(r.horizon_h),
                             n_seq=int(r.n_seq), n_cal=int(r.n_cal),
                             n_cal_dropped=int(r.n_cal_dropped),
                             scp_picp=pct1(r.scp_picp), scp_mpiw=fmt1(r.scp_mpiw),
                             aci_picp=pct1(r.aci_picp), aci_mpiw=fmt1(r.aci_mpiw))
                        for _, r in e8.iterrows()]), "cap_sensitivity.csv")

    # Morocco clean-protocol rows come from the same 30-min run as the table above
    cap30 = e8[e8.cap_min == 30].set_index("horizon_h")
    write(pd.DataFrame([dict(dataset="Morocco", horizon_h=h,
                             n_train=int(cap30.loc[h].n_train),
                             n_cal=int(cap30.loc[h].n_cal),
                             n_test=int(cap30.loc[h].n_test),
                             scp_picp=pct1(cap30.loc[h].scp_picp),
                             scp_mpiw=fmt1(cap30.loc[h].scp_mpiw),
                             aci_picp=pct1(cap30.loc[h].aci_picp),
                             aci_mpiw=fmt1(cap30.loc[h].aci_mpiw),
                             source="cap_sensitivity.csv (cap=30)")
                        for h in sorted(cap30.index)]), "clean_protocol_morocco.csv")

# --------------------------------------------------- clean cross-site ------
ccs = RES / "clean_protocol" / "clean_cross_site.csv"
if exists(ccs):
    c = pd.read_csv(ccs)
    write(pd.DataFrame([dict(site=r.site, horizon_h=int(r.horizon_h), n_seq=int(r.n_seq),
                             n_train=int(r.n_train), n_cal=int(r.n_cal),
                             n_test=int(r.n_test),
                             n_cal_dropped=int(r.n_cal_dropped),
                             scp_picp=pct1(r.scp_picp), scp_mpiw=fmt1(r.scp_mpiw),
                             aci_picp=pct1(r.aci_picp), aci_mpiw=fmt1(r.aci_mpiw))
                        for _, r in c.iterrows()]), "clean_protocol_cross_site.csv")

# ---------------------------- immediate vs availability-aware ACI ----------
# (raw experiment output lives under results/delayed_aci/ and keeps its
# historical field names cov_delayed/mpiw_delayed; the published table below
# uses the availability-aware terminology)
aci_f = RES / "delayed_aci" / "aci_feedback.csv"
if exists(aci_f):
    a = pd.read_csv(aci_f)
    write(pd.DataFrame([dict(base=r.base, horizon_h=int(r.horizon_h),
                             horizon_steps=int(r.horizon_steps),
                             n_test=int(r.n_test),
                             cov_immediate=pct1(r.cov_immediate),
                             cov_observable=pct1(r.cov_delayed),
                             difference_pp=fmt1(r.bias_pp),
                             n_feedback_consumed=int(r.n_feedback_consumed),
                             first_feedback_index=int(r.first_feedback_index),
                             frac_before_any_feedback=f"{float(r.frac_before_any_feedback):.3f}")
                        for _, r in a.iterrows()]), "delayed_aci.csv")

# --------------------------------------------------- target isolation ------
ti = RES / "target_isolation" / "target_isolation.csv"
if exists(ti):
    t = pd.read_csv(ti)
    write(pd.DataFrame([dict(horizon_h=int(r.horizon_h),
                             n_train_leaky=int(r.n_train_leaky),
                             n_train_isolated=int(r.n_train_isolated),
                             n_leaking_train_seq=int(r.n_leaking_train_seq),
                             picp_leaky=pct1(r.picp_leaky),
                             picp_isolated=pct1(r.picp_isolated),
                             rmse_leaky=fmt1(r.rmse_leaky),
                             rmse_isolated=fmt1(r.rmse_isolated))
                        for _, r in t.iterrows()]), "target_isolation.csv")

# ------------------------------------------------ boundary robustness ------
br = RES / "boundary_robustness" / "boundary_robustness.csv"
if exists(br):
    b = pd.read_csv(br)
    write(pd.DataFrame([dict(site=r.site, split=f"{r.train_frac:.2f}/{r.val_frac:.2f}",
                             horizon_h=int(r.horizon_h),
                             cov_aci_immediate=pct1(r.cov_aci_immediate),
                             cov_aci_observable=pct1(r.cov_aci_delayed),
                             immediate_minus_observable_pp=fmt1(r.aci_bias_pp),
                             n_leaking_train_seq=int(r.n_leaking_train_seq),
                             picp_leaky=pct1(r.picp_leaky),
                             picp_isolated=pct1(r.picp_isolated))
                        for _, r in b.iterrows()]), "boundary_robustness.csv")

# ------------------------------------------------------ USCRN replication --
us = RES / "uscrn" / "uscrn_injection.csv"
if exists(us):
    u = pd.read_csv(us)
    rows = []
    for frac in sorted(u.frac.unique()):
        row = {"fraction_pct": int(round(frac * 100))}
        for arm in ("linear", "gap_aware"):
            for h in sorted(u.horizon_h.unique()):
                g = u[(u.frac == frac) & (u.arm == arm) & (u.horizon_h == h)]
                row[f"{arm}_{int(h)}h_picp"] = pct1(g.picp.mean())
        rows.append(row)
    write(pd.DataFrame(rows), "uscrn_replication.csv")

# ------------------------------------------------ decision simulation ------
for label, path in (("morocco", RES / "naive_pipeline" / "uq_extensions" / "decision_sim.csv"),
                    ("iraq", RES / "multisite" / "iraq" / "uq_extensions" / "decision_sim.csv"),
                    ("johannesburg", RES / "multisite" / "mendeley" / "uq_extensions" / "decision_sim.csv")):
    if path.exists():
        ds = pd.read_csv(path)
        agg = (ds.groupby("method")
                 .agg(n_events_min=("n_events", "min"), n_events_max=("n_events", "max"),
                      miss_min=("miss_rate", "min"), miss_max=("miss_rate", "max"),
                      fa_min=("fa_rate", "min"), fa_max=("fa_rate", "max"),
                      cost_r50_min=("cost_r50", "min"), cost_r50_max=("cost_r50", "max"))
                 .reset_index().round(4))
        write(agg, f"decision_simulation_{label}.csv")

# --------------------------------------------- interpolation audit ---------
daily = RES / "interpolation_audit" / "daily_row_counts.csv"
summ = RES / "interpolation_audit" / "acquisition_summary.csv"
if exists(daily) and exists(summ):
    write(pd.read_csv(summ), "acquisition_summary.csv")

# ------------------------------------------------------- deep 20 seeds -----
msr = RES / "deep_20seed" / "morocco_seed_results.csv"
if exists(msr):
    s = pd.read_csv(msr)
    agg = (s.groupby(["site", "model", "horizon"])
             .agg(n_seeds=("seed", "nunique"),
                  rmse_mean=("rmse", "mean"), rmse_sd=("rmse", lambda v: v.std(ddof=1)),
                  r2_mean=("r2", "mean"), converged=("converged", "all"))
             .reset_index().round(4))
    write(agg, "deep_20seed_summary.csv")

# ------------------------------------------- decision sensitivity grid ----
for label, path in (("morocco", RES / "naive_pipeline" / "uq_extensions" / "decision_sensitivity.csv"),
                    ("iraq", RES / "multisite" / "iraq" / "uq_extensions" / "decision_sensitivity.csv"),
                    ("johannesburg", RES / "multisite" / "mendeley" / "uq_extensions" / "decision_sensitivity.csv")):
    if path.exists():
        ds = pd.read_csv(path)
        cost_cols = [c for c in ds.columns if c.startswith("cost_r")]
        agg = (ds.groupby(["threshold_pct", "method"])
                 .agg(n_events_min=("n_events", "min"), n_events_max=("n_events", "max"),
                      **{f"{c}_min": (c, "min") for c in cost_cols},
                      **{f"{c}_max": (c, "max") for c in cost_cols})
                 .reset_index().round(4))
        write(agg, f"decision_sensitivity_{label}.csv")

# ------------------------------------------------ paired inference ----------
sig = RES / "significance"
if exists(sig / "injection_vs_gapaware.csv"):
    g = pd.read_csv(sig / "injection_vs_gapaware.csv")
    g = g[g.geom == "distributed"]
    rows = []
    for frac in sorted(g.frac.unique()):
        row = {"fraction_pct": int(round(frac * 100))}
        for meth in ("linear", "ffill", "spline"):
            r = g[(g.method == meth) & (g.frac == frac)].iloc[0]
            row[f"{meth}_median_diff_pp"] = fmt1(r.median_diff_pp)
            row[f"{meth}_ci95_pp"] = f"[{r.ci_lo_pp:.1f}, {r.ci_hi_pp:.1f}]"
            row[f"{meth}_holm_p"] = f"{r.holm_p:.3g}"
        rows.append(row)
    write(pd.DataFrame(rows), "paired_inference_distributed.csv")
if exists(sig / "defects_paired.csv"):
    write(pd.read_csv(sig / "defects_paired.csv").round(6), "paired_inference_defects.csv")

# ------------------------------------ trained-forecaster injection --------
deep = RES / "controlled_injection_deep" / "injection_matrix_deep.csv"
if exists(deep):
    d = pd.read_csv(deep)
    d["family"] = np.where(d.model == "ridge", "ridge", "lstm")
    agg = (d.groupby(["family", "geom", "method", "frac"])
             .agg(picp_mean=("picp", "mean"), mpiw_mean=("mpiw", "mean"),
                  calib_err_pp=("calib_err", lambda v: 100 * v.mean()),
                  n_cal_mean=("n_cal", "mean"), n_nan=("picp", lambda v: int(v.isna().sum())))
             .reset_index().round(4))
    write(agg, "controlled_injection_deep_summary.csv")
    pa = RES / "controlled_injection_deep" / "point_accuracy_deep.csv"
    if pa.exists():
        write(pd.read_csv(pa).round(4), "controlled_injection_deep_point_accuracy.csv")

print(f"\nwrote {len(written)} tables -> {OUT.relative_to(ROOT)}")
for w in written:
    print("  ", w)
