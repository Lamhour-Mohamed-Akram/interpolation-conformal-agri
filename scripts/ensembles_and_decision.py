"""Ensemble uncertainty, EnbPI, a stronger point baseline, and the cost-loss study.

Scientific purpose
------------------
Completes the naive-pipeline comparison with the uncertainty families and the
decision-level analysis that residual-based conformal methods alone do not
cover, reusing the already-trained seeds so nothing is retrained.

Contents
--------
Deep ensembles      ens_gauss, a deterministic-LSTM ensemble with Gaussian
                    intervals; ens_mc_gauss, an MC-dropout mixture whose total
                    variance is mean(member sigma^2) + var(member means);
                    scp_ens_sigma, sigma-normalised split conformal on the
                    ensemble
EnbPI               bootstrap ensemble prediction intervals with out-of-bag
                    residuals and an online residual pool
Point baseline      gradient boosting on the flattened window, as a non-deep
                    reference stronger than ridge
Decision simulation an irrigation cost-loss simulation that converts nominal
                    90% intervals into irrigate/do-not-irrigate decisions. The
                    stress threshold is the 25th percentile of the TRAINING
                    target, and missed-stress and false-alarm rates are combined
                    under a range of cost ratios

The decision study is what turns a coverage number into an operational
consequence: a method can look acceptable on average coverage and still miss
nearly every stress event.

Input       results/naive_pipeline/
Output      results/naive_pipeline/uq_extensions/
Manuscript  the ensemble and EnbPI calibration rows, and the cost-loss table
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import conformal_quantile, aci as aci_shared    # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "naive_pipeline"
OUT = RESULTS / "uq_extensions"
os.makedirs(OUT, exist_ok=True)

HORIZONS = ["6h", "12h", "24h", "48h"]
LEVELS = {0.90: 1.6449, 0.95: 1.9600}
GAMMA = 0.02
WINDOW = 500
RNG = np.random.default_rng(42)

with open(RESULTS / "meta.json") as f:
    META = json.load(f)
SEEDS = META["seeds"]

with open(RESULTS / "target_scaler.pkl", "rb") as f:
    tsc = pickle.load(f)
inv = tsc.inverse_transform
SCALE = tsc.data_max_ - tsc.data_min_


def load(name, seed=None):
    p = RESULTS / (f"seed_{seed}/{name}.npy" if seed is not None else f"{name}.npy")
    return np.load(p)


y_train = inv(load("y_train"))
y_val = inv(load("y_val"))
y_test = inv(load("y_test"))
# persistence is stored in raw % units (see train_deep_models.py); ridge is target-scaled
pers_val, pers_test = load("pred_persistence_val"), load("pred_persistence_test")
ridge_val, ridge_test = inv(load("pred_ridge_val")), inv(load("pred_ridge_test"))

det_val = np.stack([inv(load("pred_deterministic_val", s)) for s in SEEDS])   # (n_seeds, n, 4)
det_test = np.stack([inv(load("pred_deterministic_test", s)) for s in SEEDS])
mc_mean_val = np.stack([inv(load("mc_mean_val", s)) for s in SEEDS])
mc_mean_test = np.stack([inv(load("mc_mean_test", s)) for s in SEEDS])
mc_std_val = np.stack([load("mc_std_val", s) * SCALE for s in SEEDS])
mc_std_test = np.stack([load("mc_std_test", s) * SCALE for s in SEEDS])

# ------------------------------------------------------------------ utils --

conformal_q = conformal_quantile         # canonical implementation (conformal.py)


def winkler(y, lo, hi, alpha):
    w = hi - lo
    below, above = y < lo, y > hi
    w = w + np.where(below, (2 / alpha) * (lo - y), 0) + np.where(above, (2 / alpha) * (y - hi), 0)
    return float(w.mean())


def evaluate(y, lo, hi, alpha):
    covered = (y >= lo) & (y <= hi)
    return float(covered.mean()), float((hi - lo).mean()), winkler(y, lo, hi, alpha)


def aci(res_cal, yhat, y, alpha, scale=None, gamma=GAMMA, window=WINDOW):
    """Immediate-feedback ACI traces via the shared implementation.

    The naive pipeline under study consumes each outcome's feedback at the
    next forecast regardless of observability (feedback_mode="immediate");
    the causally correct availability-aware variant lives in conformal.aci
    with feedback_mode="observable".
    """
    lo, hi, _ = aci_shared(res_cal, yhat, y, feedback_mode="immediate",
                           gamma=gamma, window=window, alpha=alpha, scale=scale,
                           return_traces=True)
    return lo, hi


rows = []          # method, level, horizon, picp, mpiw, winkler
intervals90 = {}   # (method, horizon) -> (lo, hi) at nominal 90% for the decision sim


def record(method, level, h, y, lo, hi):
    p, w, wk = evaluate(y, lo, hi, 1 - level)
    rows.append((method, level, h, p, w, wk))
    if level == 0.90:
        intervals90[(method, h)] = (np.asarray(lo, float), np.asarray(hi, float))


# =====================================================================
# A. Deep ensembles from all existing seeds (20)
# =====================================================================

ens_mu_val, ens_mu_test = det_val.mean(0), det_test.mean(0)
ens_sd_val = np.maximum(det_val.std(0), 1e-6)
ens_sd_test = np.maximum(det_test.std(0), 1e-6)

# MC-dropout mixture ensemble (Lakshminarayanan-style total variance)
mix_mu_val, mix_mu_test = mc_mean_val.mean(0), mc_mean_test.mean(0)
mix_sd_val = np.sqrt((mc_std_val ** 2).mean(0) + mc_mean_val.var(0))
mix_sd_test = np.sqrt((mc_std_test ** 2).mean(0) + mc_mean_test.var(0))
mix_sd_val = np.maximum(mix_sd_val, 1e-6)
mix_sd_test = np.maximum(mix_sd_test, 1e-6)

for level, z in LEVELS.items():
    alpha = 1 - level
    for h_i, h in enumerate(HORIZONS):
        yt, yv = y_test[:, h_i], y_val[:, h_i]

        record("ens_gauss", level, h, yt,
               ens_mu_test[:, h_i] - z * ens_sd_test[:, h_i],
               ens_mu_test[:, h_i] + z * ens_sd_test[:, h_i])
        record("ens_mc_gauss", level, h, yt,
               mix_mu_test[:, h_i] - z * mix_sd_test[:, h_i],
               mix_mu_test[:, h_i] + z * mix_sd_test[:, h_i])

        # sigma-normalized SCP / ACI on the mixture ensemble
        mu_t, sd_t = mix_mu_test[:, h_i], mix_sd_test[:, h_i]
        mu_v, sd_v = mix_mu_val[:, h_i], mix_sd_val[:, h_i]
        sc_val = np.abs(yv - mu_v) / sd_v
        q = conformal_q(sc_val, alpha)
        record("scp_ens_sigma", level, h, yt, mu_t - q * sd_t, mu_t + q * sd_t)

        lo, hi = aci(sc_val, mu_t, yt, alpha, scale=sd_t)
        record("aci_ens_sigma", level, h, yt, lo, hi)

print("A. ensembles done")

# =====================================================================
# B. Gradient boosting point baseline (+ SCP / ACI wrap)
# =====================================================================

Xtr = load("X_train").reshape(len(y_train), -1)
Xva = load("X_val").reshape(len(y_val), -1)
Xte = load("X_test").reshape(len(y_test), -1)
ytr_scaled = load("y_train")

gbr_val = np.zeros_like(y_val)
gbr_test = np.zeros_like(y_test)
for h_i, h in enumerate(HORIZONS):
    m = HistGradientBoostingRegressor(random_state=42)
    m.fit(Xtr, ytr_scaled[:, h_i])
    gbr_val[:, h_i] = m.predict(Xva)
    gbr_test[:, h_i] = m.predict(Xte)
np.save(RESULTS / "pred_gbr_val.npy", gbr_val.astype(np.float32))
np.save(RESULTS / "pred_gbr_test.npy", gbr_test.astype(np.float32))
gbr_val, gbr_test = inv(gbr_val), inv(gbr_test)

point_rows = []
for name, pv, pt in [("persistence", pers_val, pers_test), ("ridge", ridge_val, ridge_test),
                     ("gbr", gbr_val, gbr_test)]:
    for h_i, h in enumerate(HORIZONS):
        yt = y_test[:, h_i]
        e = yt - pt[:, h_i]
        rmse = float(np.sqrt((e ** 2).mean()))
        mae = float(np.abs(e).mean())
        r2 = float(1 - (e ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
        point_rows.append((name, h, rmse, mae, r2))

for level in LEVELS:
    alpha = 1 - level
    for h_i, h in enumerate(HORIZONS):
        yt, yv = y_test[:, h_i], y_val[:, h_i]
        res_val = np.abs(yv - gbr_val[:, h_i])
        q = conformal_q(res_val, alpha)
        record("scp_gbr", level, h, yt, gbr_test[:, h_i] - q, gbr_test[:, h_i] + q)
        lo, hi = aci(res_val, gbr_test[:, h_i], yt, alpha)
        record("aci_gbr", level, h, yt, lo, hi)

print("B. gradient boosting done")

# =====================================================================
# C. EnbPI (Xu & Xie 2021) with B=30 bootstrap ridge models
#    - OOB aggregation for calibration residuals on the training set
#    - online sliding update of the residual pool on the test stream
# =====================================================================

B = 30
n_tr = len(Xtr)
boot_idx = [RNG.choice(n_tr, n_tr, replace=True) for _ in range(B)]

for h_i, h in enumerate(HORIZONS):
    preds_tr = np.full((B, n_tr), np.nan)
    preds_te = np.zeros((B, len(y_test)))
    inb = np.zeros((B, n_tr), dtype=bool)
    for b in range(B):
        idx = boot_idx[b]
        inb[b, idx] = True
        m = Ridge(alpha=1.0)
        m.fit(Xtr[idx], ytr_scaled[idx, h_i])
        preds_tr[b] = m.predict(Xtr)
        preds_te[b] = m.predict(Xte)

    # OOB ensemble prediction per training point (mean over members not trained on it)
    oob_mask = ~inb
    with np.errstate(invalid="ignore"):
        oob_pred = np.where(oob_mask, preds_tr, np.nan)
        mu_oob = np.nanmean(oob_pred, axis=0)
    ok = ~np.isnan(mu_oob)
    # % units
    def inv_col(x, col=h_i):
        full = np.zeros((len(x), 4)); full[:, col] = x
        return inv(full)[:, col]
    res_oob = np.abs(inv_col(ytr_scaled[ok, h_i]) - inv_col(mu_oob[ok]))
    mu_te = inv_col(preds_te.mean(0))

    for level in LEVELS:
        alpha = 1 - level
        pool = list(res_oob[-WINDOW:])
        yt = y_test[:, h_i]
        lo = np.zeros(len(yt)); hi = np.zeros(len(yt))
        for t in range(len(yt)):
            q = float(np.quantile(pool, 1 - alpha))
            lo[t], hi[t] = mu_te[t] - q, mu_te[t] + q
            pool.append(abs(yt[t] - mu_te[t]))
            if len(pool) > WINDOW:
                pool.pop(0)
        record("enbpi_ridge", level, h, yt, lo, hi)

print("C. EnbPI done")

# =====================================================================
# Rebuild the headline existing methods at 90% for the decision sim
# (same code path as evaluate_uq.py; seed-mean where applicable)
# =====================================================================

alpha = 0.10
for h_i, h in enumerate(HORIZONS):
    yt, yv = y_test[:, h_i], y_val[:, h_i]

    # mc_gauss and aci_mc_sigma: use seed 42 (first seed) for trace-level sim,
    # decision metrics for per-seed methods are averaged over seeds below instead.
    per_seed = {"mc_gauss": [], "aci_mc_sigma": []}
    for s_i, s in enumerate(SEEDS):
        mu_t = mc_mean_test[s_i, :, h_i]
        sd_t = np.maximum(mc_std_test[s_i, :, h_i], 1e-6)
        mu_v = mc_mean_val[s_i, :, h_i]
        sd_v = np.maximum(mc_std_val[s_i, :, h_i], 1e-6)
        per_seed["mc_gauss"].append((mu_t - 1.6449 * sd_t, mu_t + 1.6449 * sd_t))
        sc_val = np.abs(yv - mu_v) / sd_v
        lo, hi = aci(sc_val, mu_t, yt, alpha, scale=sd_t)
        per_seed["aci_mc_sigma"].append((lo, hi))
    intervals90[("mc_gauss", h)] = per_seed["mc_gauss"]          # list of 5
    intervals90[("aci_mc_sigma", h)] = per_seed["aci_mc_sigma"]  # list of 5

    res_val = np.abs(yv - pers_val[:, h_i])
    q = conformal_q(res_val, alpha)
    intervals90[("scp_persistence", h)] = (pers_test[:, h_i] - q, pers_test[:, h_i] + q)
    lo, hi = aci(res_val, pers_test[:, h_i], yt, alpha)
    intervals90[("aci_persistence", h)] = (lo, hi)

# =====================================================================
# D. Irrigation cost-loss decision simulation (nominal 90% intervals)
#    Rule: irrigate at t if interval lower bound < tau (stress threshold).
#    tau = 25th percentile of TRAINING soil moisture (site-derived).
# =====================================================================

TAU = float(np.percentile(y_train[:, 0], 25))
COST_RATIOS = [5, 10, 20, 50]

sim_methods = ["mc_gauss", "scp_persistence", "enbpi_ridge",
               "aci_persistence", "aci_mc_sigma", "aci_ens_sigma"]
sim_rows = []
for h_i, h in enumerate(HORIZONS):
    yt = y_test[:, h_i]
    event = yt < TAU
    n_ev = int(event.sum())

    def decide(lo):
        irr = lo < TAU
        miss = event & ~irr
        fa = irr & ~event
        out = {"irr_rate": irr.mean(), "miss_rate": (miss.sum() / max(n_ev, 1)),
               "fa_rate": fa.mean()}
        for r in COST_RATIOS:
            out[f"cost_r{r}"] = (r * miss.sum() + irr.sum()) / len(yt)
        return out

    # reference policies
    oracle = {"irr_rate": event.mean(), "miss_rate": 0.0, "fa_rate": 0.0}
    for r in COST_RATIOS:
        oracle[f"cost_r{r}"] = event.mean()
    always = {"irr_rate": 1.0, "miss_rate": 0.0, "fa_rate": (~event).mean()}
    for r in COST_RATIOS:
        always[f"cost_r{r}"] = 1.0
    never = {"irr_rate": 0.0, "miss_rate": 1.0 if n_ev else 0.0, "fa_rate": 0.0}
    for r in COST_RATIOS:
        never[f"cost_r{r}"] = r * n_ev / len(yt)
    for name, d in [("oracle", oracle), ("always_irrigate", always), ("never_irrigate", never)]:
        sim_rows.append((name, h, n_ev, d))

    for m in sim_methods:
        iv = intervals90[(m, h)]
        if isinstance(iv, list):  # per-seed: average the metrics
            ds = [decide(lo) for lo, _ in iv]
            d = {k: float(np.mean([x[k] for x in ds])) for k in ds[0]}
        else:
            d = decide(iv[0])
        sim_rows.append((m, h, n_ev, d))

print(f"D. decision sim done (tau = {TAU:.1f}%)")

# ------------------------------------------------------------------ output --

with open(OUT / "uq_extensions_results.csv", "w") as f:
    f.write("method,level,horizon,picp,mpiw,winkler\n")
    for m, lv, h, p, w, wk in rows:
        f.write(f"{m},{lv},{h},{p:.4f},{w:.3f},{wk:.3f}\n")

with open(OUT / "point_forecasts_ext.csv", "w") as f:
    f.write("model,horizon,rmse,mae,r2\n")
    for m, h, rmse, mae, r2 in point_rows:
        f.write(f"{m},{h},{rmse:.3f},{mae:.3f},{r2:.4f}\n")

with open(OUT / "decision_sim.csv", "w") as f:
    keys = ["irr_rate", "miss_rate", "fa_rate"] + [f"cost_r{r}" for r in COST_RATIOS]
    f.write("method,horizon,n_events," + ",".join(keys) + "\n")
    for m, h, n_ev, d in sim_rows:
        f.write(f"{m},{h},{n_ev}," + ",".join(f"{d[k]:.4f}" for k in keys) + "\n")

with open(OUT / "uq_extensions_meta.json", "w") as f:
    json.dump({"tau_percent": TAU, "cost_ratios": COST_RATIOS, "enbpi_B": B,
               "enbpi_base": "ridge", "window": WINDOW, "gamma": GAMMA}, f, indent=1)

# quick readable summary at 90%
print("\n=== new UQ methods, nominal 90% (PICP% / MPIW) ===")
for m in ["ens_gauss", "ens_mc_gauss", "scp_ens_sigma", "aci_ens_sigma",
          "scp_gbr", "aci_gbr", "enbpi_ridge"]:
    cells = []
    for h in HORIZONS:
        r = [x for x in rows if x[0] == m and x[1] == 0.90 and x[2] == h][0]
        cells.append(f"{r[3]*100:5.1f}%/{r[4]:6.2f}")
    print(f"{m:15s} " + "  ".join(cells))

print("\n=== point forecasts (RMSE) ===")
for m in ["persistence", "ridge", "gbr"]:
    cells = [f"{r[2]:6.2f}" for r in point_rows if r[0] == m]
    print(f"{m:12s} " + "  ".join(cells))

print("\n=== decision sim @6h (miss% of events / unnecessary-irrigation% / cost r=10) ===")
for m, h, n_ev, d in sim_rows:
    if h == "6h":
        print(f"{m:18s} miss {d['miss_rate']*100:5.1f}%  fa {d['fa_rate']*100:5.1f}%  cost {d['cost_r10']:.3f}")
