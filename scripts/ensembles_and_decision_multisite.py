"""Ensemble uncertainty, EnbPI and the cost-loss study on the two external sites.

Scientific purpose
------------------
Applies the analyses of ensembles_and_decision.py to the Iraq and Johannesburg
records, using the predictions saved by train_deep_models_multisite.py so that
nothing is retrained.

Per site: deep-ensemble uncertainty (deterministic and MC-dropout mixture, with
sigma-normalised ACI), a gradient-boosting point baseline, EnbPI with bootstrap
ridge members and out-of-bag residuals, and the irrigation cost-loss simulation
at nominal 90% with the stress threshold set to the 25th percentile of the
training target.

Input       results/multisite/<site>/
Output      results/multisite/<site>/uq_extensions/
Manuscript  the cross-site calibration rows and the Johannesburg cost-loss table
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

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = ["6h", "12h", "24h", "48h"]
LEVELS = {0.90: 1.6449, 0.95: 1.9600}
GAMMA, WINDOW = 0.02, 500
# The 20 fixed seeds of the multi-seed protocol. Fixed and published so that
# every reported mean and spread can be reproduced exactly; 7 appears once.
SEEDS = [42, 123, 2024, 7, 99,
         0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
         11, 12, 13, 14, 15]
B = 30
COST_RATIOS = [5, 10, 20, 50]
RNG = np.random.default_rng(42)


conformal_q = conformal_quantile         # canonical implementation (conformal.py)


def winkler(y, lo, hi, alpha):
    w = hi - lo
    w = w + np.where(y < lo, (2 / alpha) * (lo - y), 0) + np.where(y > hi, (2 / alpha) * (y - hi), 0)
    return float(w.mean())


def evaluate(y, lo, hi, alpha):
    cov = (y >= lo) & (y <= hi)
    return float(cov.mean()), float((hi - lo).mean()), winkler(y, lo, hi, alpha)


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


def run_site(site: str) -> None:
    S = ROOT / "results" / "multisite" / site
    OUT = S / "uq_extensions"
    os.makedirs(OUT, exist_ok=True)

    with open(S / "target_scaler.pkl", "rb") as f:
        tsc = pickle.load(f)
    inv = tsc.inverse_transform

    y_train = inv(np.load(S / "y_train.npy"))
    y_val = inv(np.load(S / "y_val.npy"))
    y_test = inv(np.load(S / "y_test.npy"))
    pers_val = np.load(S / "pred_persistence_val.npy")
    pers_test = np.load(S / "pred_persistence_test.npy")
    ridge_val = np.load(S / "pred_ridge_val.npy")
    ridge_test = np.load(S / "pred_ridge_test.npy")
    det_val = np.stack([np.load(S / f"seed_{s}/det_va.npy") for s in SEEDS])
    det_test = np.stack([np.load(S / f"seed_{s}/det_te.npy") for s in SEEDS])
    mu_val = np.stack([np.load(S / f"seed_{s}/mu_va.npy") for s in SEEDS])
    mu_test = np.stack([np.load(S / f"seed_{s}/mu_te.npy") for s in SEEDS])
    sd_val = np.stack([np.load(S / f"seed_{s}/sd_va.npy") for s in SEEDS])
    sd_test = np.stack([np.load(S / f"seed_{s}/sd_te.npy") for s in SEEDS])

    X_tr = np.load(S / "X_train.npy").reshape(len(y_train), -1)
    X_va = np.load(S / "X_val.npy").reshape(len(y_val), -1)
    X_te = np.load(S / "X_test.npy").reshape(len(y_test), -1)
    ytr_scaled = np.load(S / "y_train.npy")

    rows = []
    intervals90 = {}

    def record(method, level, h, y, lo, hi):
        p, w, wk = evaluate(y, lo, hi, 1 - level)
        rows.append((method, level, h, p, w, wk))
        if level == 0.90:
            intervals90[(method, h)] = (np.asarray(lo, float), np.asarray(hi, float))

    # ---- A. ensembles ----
    ens_mu_t, ens_sd_t = det_test.mean(0), np.maximum(det_test.std(0), 1e-6)
    mix_mu_v, mix_mu_t = mu_val.mean(0), mu_test.mean(0)
    mix_sd_v = np.maximum(np.sqrt((sd_val ** 2).mean(0) + mu_val.var(0)), 1e-6)
    mix_sd_t = np.maximum(np.sqrt((sd_test ** 2).mean(0) + mu_test.var(0)), 1e-6)

    for level, z in LEVELS.items():
        alpha = 1 - level
        for h_i, h in enumerate(HORIZONS):
            yt, yv = y_test[:, h_i], y_val[:, h_i]
            record("ens_gauss", level, h, yt,
                   ens_mu_t[:, h_i] - z * ens_sd_t[:, h_i], ens_mu_t[:, h_i] + z * ens_sd_t[:, h_i])
            record("ens_mc_gauss", level, h, yt,
                   mix_mu_t[:, h_i] - z * mix_sd_t[:, h_i], mix_mu_t[:, h_i] + z * mix_sd_t[:, h_i])
            sc_val = np.abs(yv - mix_mu_v[:, h_i]) / mix_sd_v[:, h_i]
            q = conformal_q(sc_val, alpha)
            record("scp_ens_sigma", level, h, yt,
                   mix_mu_t[:, h_i] - q * mix_sd_t[:, h_i], mix_mu_t[:, h_i] + q * mix_sd_t[:, h_i])
            lo, hi = aci(sc_val, mix_mu_t[:, h_i], yt, alpha, scale=mix_sd_t[:, h_i])
            record("aci_ens_sigma", level, h, yt, lo, hi)

    # ---- B. gradient boosting ----
    gbr_val = np.zeros_like(y_val)
    gbr_test = np.zeros_like(y_test)
    for h_i in range(4):
        m = HistGradientBoostingRegressor(random_state=42)
        m.fit(X_tr, ytr_scaled[:, h_i])
        gbr_val[:, h_i] = m.predict(X_va)
        gbr_test[:, h_i] = m.predict(X_te)
    gbr_val, gbr_test = inv(gbr_val), inv(gbr_test)

    point_rows = []
    for name, pt in [("persistence", pers_test), ("ridge", ridge_test), ("gbr", gbr_test)]:
        for h_i, h in enumerate(HORIZONS):
            yt = y_test[:, h_i]
            e = yt - pt[:, h_i]
            point_rows.append((name, h, float(np.sqrt((e ** 2).mean())), float(np.abs(e).mean()),
                               float(1 - (e ** 2).sum() / ((yt - yt.mean()) ** 2).sum())))

    for level in LEVELS:
        alpha = 1 - level
        for h_i, h in enumerate(HORIZONS):
            yt, yv = y_test[:, h_i], y_val[:, h_i]
            res_val = np.abs(yv - gbr_val[:, h_i])
            q = conformal_q(res_val, alpha)
            record("scp_gbr", level, h, yt, gbr_test[:, h_i] - q, gbr_test[:, h_i] + q)
            lo, hi = aci(res_val, gbr_test[:, h_i], yt, alpha)
            record("aci_gbr", level, h, yt, lo, hi)

    # ---- C. EnbPI (bootstrap ridge) ----
    n_tr = len(X_tr)
    boot_idx = [RNG.choice(n_tr, n_tr, replace=True) for _ in range(B)]
    for h_i, h in enumerate(HORIZONS):
        preds_tr = np.full((B, n_tr), np.nan)
        preds_te = np.zeros((B, len(y_test)))
        inb = np.zeros((B, n_tr), dtype=bool)
        for b in range(B):
            idx = boot_idx[b]
            inb[b, idx] = True
            m = Ridge(alpha=1.0)
            m.fit(X_tr[idx], ytr_scaled[idx, h_i])
            preds_tr[b] = m.predict(X_tr)
            preds_te[b] = m.predict(X_te)
        with np.errstate(invalid="ignore"):
            mu_oob = np.nanmean(np.where(~inb, preds_tr, np.nan), axis=0)
        ok = ~np.isnan(mu_oob)

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

    # ---- existing headline methods at 90% for the decision sim ----
    alpha = 0.10
    for h_i, h in enumerate(HORIZONS):
        yt, yv = y_test[:, h_i], y_val[:, h_i]
        per_seed = {"mc_gauss": [], "aci_mc_sigma": []}
        for s_i in range(len(SEEDS)):
            mu_t = mu_test[s_i, :, h_i]
            sd_t = np.maximum(sd_test[s_i, :, h_i], 1e-6)
            mu_v = mu_val[s_i, :, h_i]
            sd_v = np.maximum(sd_val[s_i, :, h_i], 1e-6)
            per_seed["mc_gauss"].append((mu_t - 1.6449 * sd_t, mu_t + 1.6449 * sd_t))
            sc_val = np.abs(yv - mu_v) / sd_v
            lo, hi = aci(sc_val, mu_t, yt, alpha, scale=sd_t)
            per_seed["aci_mc_sigma"].append((lo, hi))
        intervals90[("mc_gauss", h)] = per_seed["mc_gauss"]
        intervals90[("aci_mc_sigma", h)] = per_seed["aci_mc_sigma"]

        res_val = np.abs(yv - pers_val[:, h_i])
        q = conformal_q(res_val, alpha)
        intervals90[("scp_persistence", h)] = (pers_test[:, h_i] - q, pers_test[:, h_i] + q)
        lo, hi = aci(res_val, pers_test[:, h_i], yt, alpha)
        intervals90[("aci_persistence", h)] = (lo, hi)

    # ---- D. decision simulation ----
    TAU = float(np.percentile(y_train[:, 0], 25))
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
            out = {"irr_rate": float(irr.mean()), "miss_rate": float(miss.sum() / max(n_ev, 1)),
                   "fa_rate": float(fa.mean())}
            for r in COST_RATIOS:
                out[f"cost_r{r}"] = float((r * miss.sum() + irr.sum()) / len(yt))
            return out

        oracle = {"irr_rate": float(event.mean()), "miss_rate": 0.0, "fa_rate": 0.0}
        for r in COST_RATIOS:
            oracle[f"cost_r{r}"] = float(event.mean())
        never = {"irr_rate": 0.0, "miss_rate": 1.0 if n_ev else 0.0, "fa_rate": 0.0}
        for r in COST_RATIOS:
            never[f"cost_r{r}"] = float(r * n_ev / len(yt))
        sim_rows.append(("oracle", h, n_ev, oracle))
        sim_rows.append(("never_irrigate", h, n_ev, never))

        for m in sim_methods:
            iv = intervals90[(m, h)]
            if isinstance(iv, list):
                ds = [decide(np.asarray(lo)) for lo, _ in iv]
                d = {k: float(np.mean([x[k] for x in ds])) for k in ds[0]}
            else:
                d = decide(np.asarray(iv[0]))
            sim_rows.append((m, h, n_ev, d))

    # ---- output ----
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
        json.dump({"tau_percent": TAU, "cost_ratios": COST_RATIOS, "enbpi_B": B}, f, indent=1)

    print(f"\n=== {site}: new UQ methods, nominal 90% (PICP%/MPIW) — tau={TAU:.1f} ===")
    for m in ["ens_gauss", "ens_mc_gauss", "aci_ens_sigma", "scp_gbr", "aci_gbr", "enbpi_ridge"]:
        cells = []
        for h in HORIZONS:
            r = [x for x in rows if x[0] == m and x[1] == 0.90 and x[2] == h][0]
            cells.append(f"{r[3]*100:5.1f}%/{r[4]:6.2f}")
        print(f"{m:15s} " + "  ".join(cells))
    print(f"--- point RMSE ---")
    for m in ["persistence", "ridge", "gbr"]:
        print(f"{m:12s} " + "  ".join(f"{r[2]:7.2f}" for r in point_rows if r[0] == m))
    print(f"--- decision sim @6h (n_events={sim_rows[0][2]}) ---")
    for m, h, n_ev, d in sim_rows:
        if h == "6h":
            print(f"{m:18s} miss {d['miss_rate']*100:5.1f}%  fa {d['fa_rate']*100:5.1f}%  cost@r10 {d['cost_r10']:.3f}")


if __name__ == "__main__":
    for site in ["iraq", "mendeley"]:
        run_site(site)
