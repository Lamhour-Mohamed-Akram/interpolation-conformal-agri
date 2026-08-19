"""Regenerate every manuscript figure from the stored experiment outputs.

Each figure reads the result file written by the experiment that produced it, so
no scientific value is typed into this file. The only constants here are
presentation parameters: figure sizes, colours, markers, font sizes, axis limits
chosen to leave headroom, and the nominal 0.90 reference line.

Figure -> source
----------------
fig1_dataset_split       results/naive_pipeline/ (target arrays and timestamps)
fig2_rmse_by_horizon     results/naive_pipeline/point_metrics.json
fig3_residual_shift      results/naive_pipeline/seed_*/pred_deterministic_*.npy
fig5_rolling_coverage    results/naive_pipeline/uq/rolling_coverage.npz
fig8_multisite_coverage  results/naive_pipeline/uq/uq_aggregate.csv,
                         results/multisite/<site>/site_results.json
fig_matrix               results/controlled_injection/injection_matrix.csv
fig_mechanism            results/controlled_injection/injection_matrix.csv
fig_uscrn                results/uscrn/uscrn_injection.csv
fig_leak                 results/target_isolation/target_isolation.csv
fig_aci                  results/delayed_aci/aci_feedback.csv

Run: python scripts/generate_figures.py
Outputs: results/figures/*.pdf (and *.png)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
NAIVE = RES / "naive_pipeline"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = ["6h", "12h", "24h", "48h"]
NOMINAL = 0.90

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 120, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


def have(*paths) -> bool:
    return all(Path(p).exists() for p in paths)


# =========================================================== naive pipeline ==
if have(NAIVE / "target_scaler.pkl", NAIVE / "y_train.npy"):
    with open(NAIVE / "target_scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    inv = scaler.inverse_transform

    def arr(name, seed=None):
        p = NAIVE / (f"seed_{seed}/{name}.npy" if seed is not None else f"{name}.npy")
        return np.load(p, allow_pickle=False)

    y_train, y_val, y_test = inv(arr("y_train")), inv(arr("y_val")), inv(arr("y_test"))
    dates_tr = arr("dates_train").astype("datetime64[s]")
    dates_va = arr("dates_val").astype("datetime64[s]")
    dates_te = arr("dates_test").astype("datetime64[s]")

    # ---- Fig 1: the record on the naive grid, with the chronological split ----
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for d, y, lab, c in [(dates_tr, y_train, "train (60%)", "#4878cf"),
                         (dates_va, y_val, "validation (20%)", "#ee854a"),
                         (dates_te, y_test, "test (20%)", "#d65f5f")]:
        ax.plot(d, y[:, 0], lw=0.6, color=c, label=lab)
    ax.set_ylabel("Soil moisture (%)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(ncol=3, loc="upper left", framealpha=0.9, fontsize=7)
    ax.set_title("Greenhouse soil moisture on the 5-min resampled grid, chronological split")
    save(fig, "fig1_dataset_split")

    # ---- Fig 3: validation vs test absolute residuals, seeds pooled ----
    seeds = json.load(open(NAIVE / "meta.json"))["seeds"]
    res_val, res_test = [], []
    for s in seeds:
        res_val.append(np.abs(y_val - inv(arr("pred_deterministic_val", s))))
        res_test.append(np.abs(y_test - inv(arr("pred_deterministic_test", s))))
    rv, rt = np.concatenate(res_val), np.concatenate(res_test)

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    data, positions, colors = [], [], []
    for i, h in enumerate(HORIZONS):
        data += [rv[:, i], rt[:, i]]
        positions += [i * 2.5, i * 2.5 + 1]
        colors += ["#4878cf", "#d65f5f"]
    bp = ax.boxplot(data, positions=positions, widths=0.8, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticks([i * 2.5 + 0.5 for i in range(len(HORIZONS))], HORIZONS)
    ax.set_ylabel("|residual| (% soil moisture)")
    ax.set_xlabel("Forecast horizon")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#4878cf", alpha=0.7),
               plt.Rectangle((0, 0), 1, 1, fc="#d65f5f", alpha=0.7)]
    ax.legend(handles, ["validation week", "test week"], loc="upper left")
    ax.set_title(f"LSTM absolute residuals: validation vs. test ({len(seeds)} seeds pooled)")
    save(fig, "fig3_residual_shift")
else:
    print("  [skip] fig1/fig3: naive-pipeline arrays absent (run train_deep_models.py)")

# ---- Fig 2: point-forecast accuracy by horizon ----
if have(NAIVE / "point_metrics.json"):
    M = json.load(open(NAIVE / "point_metrics.json"))
    x = np.arange(len(HORIZONS))
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    pers = [M["baselines"]["persistence"][h]["rmse"] for h in HORIZONS]
    ridge = [M["baselines"]["ridge"][h]["rmse"] for h in HORIZONS]
    det_m = [M["aggregate"]["deterministic"][h]["rmse"]["mean"] for h in HORIZONS]
    det_s = [M["aggregate"]["deterministic"][h]["rmse"]["std"] for h in HORIZONS]
    bay_m = [M["aggregate"]["bayesian"][h]["rmse"]["mean"] for h in HORIZONS]
    bay_s = [M["aggregate"]["bayesian"][h]["rmse"]["std"] for h in HORIZONS]
    n_seeds = len(M["per_seed"])
    ax.plot(x, pers, "o-", color="#333333", label="Persistence")
    ax.plot(x, ridge, "s-", color="#4878cf", label="Ridge")
    ax.errorbar(x, det_m, yerr=det_s, fmt="^-", color="#ee854a", capsize=3,
                label=f"LSTM ({n_seeds} seeds)")
    ax.errorbar(x, bay_m, yerr=bay_s, fmt="v-", color="#d65f5f", capsize=3,
                label=f"MC-dropout LSTM ({n_seeds} seeds)")
    ax.set_xticks(x, HORIZONS)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Test RMSE (% soil moisture)")
    ax.margins(y=0.10)                       # keep error-bar caps clear of the frame
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
              frameon=False, handlelength=2.0, columnspacing=1.6, borderaxespad=0.0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)         # room for the legend below the axes
    save(fig, "fig2_rmse_by_horizon")

# ---- Fig 5: rolling coverage across the test period ----
if have(NAIVE / "uq" / "rolling_coverage.npz", NAIVE / "dates_test.npy"):
    roll = np.load(NAIVE / "uq" / "rolling_coverage.npz")
    dates_te = np.load(NAIVE / "dates_test.npy", allow_pickle=False).astype("datetime64[s]")
    W = 200

    def rolling_mean(c, w=W):
        return np.convolve(c.astype(float), np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, h in zip(axes, ["6h", "24h"]):
        for key, lab, c in [(f"mc_gauss|{h}", "MC-dropout", "#d65f5f"),
                            (f"scp_persistence|{h}", "SCP (persistence)", "#ee854a"),
                            (f"aci_persistence|{h}", "ACI (persistence)", "#2e7d32")]:
            if key in roll:
                cov = rolling_mean(roll[key])
                ax.plot(dates_te[W - 1: W - 1 + len(cov)], cov, lw=1.2, color=c, label=lab)
        ax.axhline(NOMINAL, color="k", ls="--", lw=0.8)
        ax.set_title(f"{h} horizon")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel(f"Rolling coverage (window={W})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=7.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    save(fig, "fig5_rolling_coverage")

# ---- Fig 8: cross-site coverage at nominal 90% ----
MULTI_LABELS = {"iraq": "Iraq (5-min)", "mendeley": "Johannesburg (hourly)"}
METHODS_8 = [("mc_gauss", "MC-dropout"), ("scp_lstm", "SCP (LSTM)"),
             ("scp_persistence", "SCP (persistence)"), ("aci_persistence", "ACI (persistence)"),
             ("aci_mc_sigma", "ACI-σ (MC-dropout)")]
if have(NAIVE / "uq" / "uq_aggregate.csv"):
    sites = {}
    agg = pd.read_csv(NAIVE / "uq" / "uq_aggregate.csv")
    agg90 = agg[np.isclose(agg.iloc[:, 1].astype(float), NOMINAL)]
    sites["Morocco (5-min)"] = {f"{r.iloc[0]}|{r.iloc[2]}": {"picp_mean": float(r.picp_mean)}
                                for _, r in agg90.iterrows()}
    for site, label in MULTI_LABELS.items():
        f = RES / "multisite" / site / "site_results.json"
        if f.exists():
            sites[label] = json.load(open(f))["uq_90"]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(sites), figsize=(9.8, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, sdata) in zip(axes, sites.items()):
        width = 0.8 / len(METHODS_8)
        for j, (key, lab) in enumerate(METHODS_8):
            vals = [sdata.get(f"{key}|{h}", {}).get("picp_mean", np.nan) for h in HORIZONS]
            ax.bar(np.arange(4) + j * width, vals, width, color=cmap(j), label=lab)
        ax.axhline(NOMINAL, color="k", ls="--", lw=1)
        ax.set_xticks(np.arange(4) + 0.4 - width / 2, HORIZONS)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Coverage (PICP)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, fontsize=7.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    save(fig, "fig8_multisite_coverage")

# ==================================================== controlled injection ==
inj_file = RES / "controlled_injection" / "injection_matrix.csv"
if inj_file.exists():
    d = pd.read_csv(inj_file)
    fracs = sorted(d.frac.unique())
    x = [100 * f for f in fracs]

    # ---- Fig matrix: calibration error by fill method, and coverage by geometry ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for m, lab, c, mk in [("none", "gap-aware (proposed)", "tab:blue", "o"),
                          ("ffill", "forward-fill", "tab:orange", "s"),
                          ("linear", "linear", "tab:green", "^"),
                          ("spline", "cubic spline", "tab:red", "D")]:
        ys = [100 * d[(d.geom == "distributed") & (d.method == m) & (d.frac == f)].calib_err.mean()
              for f in fracs]
        ax1.plot(x, ys, mk + "-", color=c, ms=4, label=lab)
    ax1.set_xlabel("calibration-set interpolation fraction (%)")
    ax1.set_ylabel(r"|PICP $-$ 0.90| (pp)")
    ax1.set_title("(a) Fill method")
    ax1.margins(y=0.12)
    ax1.grid(alpha=.3)
    ax1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    for g, lab, c, mk in [("single", "single long block", "tab:blue", "s"),
                          ("distributed", "distributed blocks", "tab:orange", "o"),
                          ("isolated", "isolated points", "tab:green", "^")]:
        ys = [100 * d[(d.geom == g) & (d.method == "linear") & (d.frac == f)].picp.mean()
              for f in fracs]
        ax2.plot(x, ys, mk + "-", color=c, ms=4, label=lab)
    ax2.axhline(100 * NOMINAL, ls="--", c="k", lw=.8, label="nominal 90%")
    ax2.set_xlabel("calibration-set interpolation fraction (%)")
    ax2.set_ylabel("PICP (%)")
    ax2.set_title("(b) Gap geometry (linear fill)")
    ax2.margins(y=0.12)
    ax2.grid(alpha=.3)
    ax2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    save(fig, "fig_matrix")

    # ---- Fig mechanism: residual-scale distortion and the resulting width ----
    dd = d[d.geom == "distributed"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for meth, c, mk, lab in [("linear", "tab:red", "o", "linear"),
                             ("ffill", "tab:orange", "s", "forward-fill"),
                             ("spline", "tab:purple", "^", "cubic spline"),
                             ("none", "tab:green", "D", "gap-aware")]:
        g = dd[dd.method == meth].groupby("frac")
        ax1.plot(g.val_res_p90.mean().index * 100, g.val_res_p90.mean().values,
                 mk + "-", color=c, ms=4, label=lab)
        ax2.plot(g.mpiw.mean().index * 100, g.mpiw.mean().values, mk + "-", color=c, ms=4)
    ax1.set_xlabel("injected interpolation fraction (%)")
    ax1.set_ylabel("calibration-set residual p90")
    ax1.set_yscale("log"); ax1.set_title("(a) residual-scale distortion")
    ax1.grid(alpha=.3, which="both"); ax1.legend(fontsize=7.5)
    ax2.set_xlabel("injected interpolation fraction (%)")
    ax2.set_ylabel("mean interval width (MPIW)")
    ax2.set_yscale("log"); ax2.set_title("(b) resulting interval width")
    ax2.grid(alpha=.3, which="both")
    fig.tight_layout()
    save(fig, "fig_mechanism")

# ============================================================ USCRN replication ==
uscrn_file = RES / "uscrn" / "uscrn_injection.csv"
if uscrn_file.exists():
    u = pd.read_csv(uscrn_file)
    fracs = sorted(u.frac.unique())
    x = [100 * f for f in fracs]

    def cov(arm, h):
        return [100 * u[(u.arm == arm) & (u.horizon_h == h) & (u.frac == f)].picp.mean()
                for f in fracs]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(x, cov("linear", 48), "o--", color="tab:red", label="48 h, linear fill")
    ax.plot(x, cov("gap_aware", 48), "o-", color="tab:green", label="48 h, gap-aware")
    ax.plot(x, cov("linear", 6), "s--", color="tab:red", alpha=.55, label="6 h, linear fill")
    ax.plot(x, cov("gap_aware", 6), "s-", color="tab:green", alpha=.55, label="6 h, gap-aware")
    ax.axhline(100 * NOMINAL, color="k", lw=.8, ls=":")
    ax.text(1, 100 * NOMINAL + 0.3, "nominal 90%", fontsize=7.5)
    ax.set_xlabel("injected interpolation fraction (%)")
    ax.set_ylabel("test coverage (%)")
    ax.set_title("USCRN Fairhope replication (persistence SCP)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_uscrn")

# ========================================================== target isolation ==
leak_file = RES / "target_isolation" / "target_isolation.csv"
if leak_file.exists():
    t = pd.read_csv(leak_file).sort_values("horizon_h")
    xp = np.arange(len(t))
    fig, ax1 = plt.subplots(figsize=(5.2, 3.8))
    ax1.bar(xp, t.n_leaking_train_seq, width=.55, color="tab:gray", alpha=.45,
            label="leaking train seqs")
    ax1.set_ylabel("leaking training sequences")
    ax1.set_xticks(xp); ax1.set_xticklabels([f"{int(h)}h" for h in t.horizon_h])
    ax1.set_xlabel("horizon")
    ax1.set_ylim(0, 1.12 * t.n_leaking_train_seq.max())
    ax2 = ax1.twinx()
    ax2.plot(xp, 100 * t.picp_leaky, "o--", color="tab:red", label="PICP leaky split")
    ax2.plot(xp, 100 * t.picp_isolated, "o-", color="tab:green", label="PICP target-isolated")
    ax2.axhline(100 * NOMINAL, ls=":", c="k", lw=.8)
    ax2.set_ylabel("ridge test PICP (%)")
    lo = min(100 * t.picp_leaky.min(), 100 * NOMINAL)
    hi = max(100 * t.picp_isolated.max(), 100 * t.picp_leaky.max())
    ax2.set_ylim(lo - 2.5, hi + 2.5)
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    fig.legend(l1 + l2, la1 + la2, fontsize=7.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    ax1.set_title("Target-boundary leakage vs horizon")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)
    save(fig, "fig_leak")

# ============================== immediate vs availability-aware ACI (fig10) ==
aci_file = RES / "delayed_aci" / "aci_feedback.csv"
if aci_file.exists():
    a = pd.read_csv(aci_file)
    H = sorted(a.horizon_h.unique())
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    style = {("ridge", "cov_immediate"): ("o--", "tab:red", 1.0, "ridge, immediate"),
             ("ridge", "cov_delayed"): ("o-", "tab:red", .55, "ridge, observable (correct)"),
             ("persistence", "cov_immediate"): ("s--", "tab:blue", 1.0, "persistence, immediate"),
             ("persistence", "cov_delayed"): ("s-", "tab:blue", .55, "persistence, observable")}
    for (base, col), (fmt, c, alpha, lab) in style.items():
        ys = [100 * a[(a.base == base) & (a.horizon_h == h)][col].iloc[0] for h in H]
        ax.plot(H, ys, fmt, color=c, alpha=alpha, label=lab)
    ax.axhline(100 * NOMINAL, ls=":", c="k", lw=.8)
    ax.set_xticks(H)
    ax.set_xlabel("horizon (h)")
    ax.set_ylabel("ACI coverage (%)")
    ax.set_title("Immediate vs. availability-aware ACI")
    ax.margins(y=0.15)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    save(fig, "fig_aci")

print(f"\nfigures written to {OUT.relative_to(ROOT)}")
