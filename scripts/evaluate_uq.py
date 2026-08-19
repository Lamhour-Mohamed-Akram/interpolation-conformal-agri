"""Uncertainty-quantification evaluation of the naive-pipeline predictions.

Scientific purpose
------------------
Scores every uncertainty method reported for the naive pipeline on the same
predictions, at horizons 6/12/24/48 h and nominal levels 90% and 95%, so that
Bayesian, quantile and conformal families are compared like for like.

Methods
-------
Bayesian (per seed)
  mc_gauss          MC-dropout Gaussian intervals, mean +/- z * std
  mc_quant          MC-dropout empirical-quantile intervals
Split conformal (validation-calibrated, finite-sample corrected)
  scp_lstm          absolute-residual scores on the deterministic LSTM
  scp_ridge         absolute-residual scores on the ridge baseline
  scp_persistence   absolute-residual scores on persistence
  scp_mc_sigma      sigma-normalised scores |y-mu|/sigma on MC-dropout
Adaptive conformal inference
  online variants of the above, updated as outcomes arrive. This is the
  IMMEDIATE-feedback variant (conformal.aci with feedback_mode="immediate"),
  deliberately: the naive pipeline under study applies each outcome's feedback
  at the step it is issued, before it is observable. The corrected
  availability-aware (observable-feedback) protocol is evaluated in
  delayed_aci.py and clean_protocol.py.

All conformal quantiles and every ACI recursion come from conformal.py, the
repository's single implementation. Persistence predictions are stored in raw
% units by train_deep_models.py (see its docstring) and are used as-is; model
predictions are target-scaled and inverse-transformed here.

Reported per method: coverage (PICP), mean interval width (MPIW), and the
rolling-coverage trace used for the coverage-over-time figure.

Input       results/naive_pipeline/ (predictions from train_deep_models.py)
Output      results/naive_pipeline/uq/
Manuscript  the naive-pipeline calibration tables and the rolling-coverage figure
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import conformal_quantile, aci as aci_shared    # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "naive_pipeline"
OUT = RESULTS / "uq"
os.makedirs(OUT, exist_ok=True)

HORIZONS = ["6h", "12h", "24h", "48h"]
LEVELS = {0.90: 1.6449, 0.95: 1.9600}
GAMMA = 0.02
WINDOW = 500

with open(RESULTS / "meta.json") as f:
    META = json.load(f)
SEEDS = META["seeds"]

with open(RESULTS / "target_scaler.pkl", "rb") as f:
    tsc = pickle.load(f)
inv = tsc.inverse_transform
SCALE = (tsc.data_max_ - tsc.data_min_)  # per-horizon scale for std arrays


def load(name: str, seed=None) -> np.ndarray:
    p = RESULTS / (f"seed_{seed}/{name}.npy" if seed is not None else f"{name}.npy")
    return np.load(p)


y_val = inv(load("y_val"))
y_test = inv(load("y_test"))
n_test = len(y_test)

# persistence is stored in raw % units; ridge is target-scaled
pers_val, pers_test = load("pred_persistence_val"), load("pred_persistence_test")
ridge_val, ridge_test = inv(load("pred_ridge_val")), inv(load("pred_ridge_test"))

seed_data = {}
for s in SEEDS:
    seed_data[s] = {
        "det_val": inv(load("pred_deterministic_val", s)),
        "det_test": inv(load("pred_deterministic_test", s)),
        "mc_mean_val": inv(load("mc_mean_val", s)),
        "mc_mean_test": inv(load("mc_mean_test", s)),
        "mc_std_val": load("mc_std_val", s) * SCALE,
        "mc_std_test": load("mc_std_test", s) * SCALE,
        "q90lo": inv(load("mc_q90lo_test", s)), "q90hi": inv(load("mc_q90hi_test", s)),
        "q95lo": inv(load("mc_q95lo_test", s)), "q95hi": inv(load("mc_q95hi_test", s)),
    }

# ------------------------------------------------------------------ utils --


def winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    w = hi - lo
    below, above = y < lo, y > hi
    w = w + np.where(below, (2 / alpha) * (lo - y), 0) + np.where(above, (2 / alpha) * (y - hi), 0)
    return float(w.mean())


def evaluate(y, lo, hi, alpha):
    covered = (y >= lo) & (y <= hi)
    return float(covered.mean()), float((hi - lo).mean()), winkler(y, lo, hi, alpha), covered


def aci_immediate(res_cal, yhat, y, alpha, scale=None, gamma=GAMMA, window=WINDOW):
    """Immediate-feedback ACI traces via the shared implementation.

    This is the deliberately defective naive-pipeline variant
    (feedback_mode="immediate"): each outcome updates the online state at the
    next forecast regardless of when it becomes physically observable.
    """
    return aci_shared(res_cal, yhat, y, feedback_mode="immediate", gamma=gamma,
                      window=window, alpha=alpha, scale=scale, return_traces=True)


rows = []          # method, seed, level, horizon, picp, mpiw, winkler
rolling = {}       # (method, horizon) -> covered bool array (level 0.90, first seed / seedless)


def record(method, seed, level, h, y, lo, hi, keep_rolling=False):
    p, w, wk, cov = evaluate(y, lo, hi, 1 - level)
    rows.append((method, seed, level, h, p, w, wk))
    if keep_rolling and level == 0.90:
        rolling[f"{method}|{h}"] = cov


for level, z in LEVELS.items():
    alpha = 1 - level
    for h_i, h in enumerate(HORIZONS):
        yt = y_test[:, h_i]
        yv = y_val[:, h_i]

        # ---- seed-independent methods ----
        for name, pv, pt in [("persistence", pers_val, pers_test), ("ridge", ridge_val, ridge_test)]:
            res_val = np.abs(yv - pv[:, h_i])
            q = conformal_quantile(res_val, alpha)
            record(f"scp_{name}", None, level, h, yt, pt[:, h_i] - q, pt[:, h_i] + q,
                   keep_rolling=(name == "persistence"))

            lo, hi, _ = aci_immediate(res_val, pt[:, h_i], yt, alpha)
            record(f"aci_{name}", None, level, h, yt, lo, hi,
                   keep_rolling=(name == "persistence"))

        # ---- per-seed methods ----
        for s in SEEDS:
            d = seed_data[s]
            mu_t, sd_t = d["mc_mean_test"][:, h_i], np.maximum(d["mc_std_test"][:, h_i], 1e-6)
            mu_v, sd_v = d["mc_mean_val"][:, h_i], np.maximum(d["mc_std_val"][:, h_i], 1e-6)

            record("mc_gauss", s, level, h, yt, mu_t - z * sd_t, mu_t + z * sd_t,
                   keep_rolling=(s == SEEDS[0]))
            qlo = d["q90lo"] if level == 0.90 else d["q95lo"]
            qhi = d["q90hi"] if level == 0.90 else d["q95hi"]
            record("mc_quant", s, level, h, yt, qlo[:, h_i], qhi[:, h_i])

            # split conformal on deterministic LSTM
            res_val = np.abs(yv - d["det_val"][:, h_i])
            q = conformal_quantile(res_val, alpha)
            record("scp_lstm", s, level, h, yt,
                   d["det_test"][:, h_i] - q, d["det_test"][:, h_i] + q,
                   keep_rolling=(s == SEEDS[0]))

            # sigma-normalized split conformal calibrated on validation scores
            sc_val = np.abs(yv - mu_v) / sd_v
            q = conformal_quantile(sc_val, alpha)
            record("scp_mc_sigma", s, level, h, yt, mu_t - q * sd_t, mu_t + q * sd_t)

            # ACI on deterministic LSTM
            lo, hi, _ = aci_immediate(res_val, d["det_test"][:, h_i], yt, alpha)
            record("aci_lstm", s, level, h, yt, lo, hi, keep_rolling=(s == SEEDS[0]))

            # sigma-normalized ACI on MC-dropout
            lo, hi, _ = aci_immediate(sc_val, mu_t, yt, alpha, scale=sd_t)
            record("aci_mc_sigma", s, level, h, yt, lo, hi, keep_rolling=(s == SEEDS[0]))

# --------------------------------------------------- sensitivity analyses --
sens_rows = []
alpha = 0.10
for h_i, h in enumerate(HORIZONS):
    yt = y_test[:, h_i]
    yv = y_val[:, h_i]
    res_val = np.abs(yv - pers_val[:, h_i])
    for gamma in [0.005, 0.02, 0.05]:
        for window in [250, 500, 1000]:
            lo, hi, _ = aci_immediate(res_val, pers_test[:, h_i], yt, alpha,
                                      gamma=gamma, window=window)
            p, w, wk, _ = evaluate(yt, lo, hi, alpha)
            sens_rows.append(("aci_persistence", h, gamma, window, p, w, wk))

# ------------------------------------------------------------------ output --
with open(OUT / "uq_results.csv", "w") as f:
    f.write("method,seed,level,horizon,picp,mpiw,winkler\n")
    for m, s, lv, h, p, w, wk in rows:
        f.write(f"{m},{'' if s is None else s},{lv},{h},{p:.4f},{w:.3f},{wk:.3f}\n")

# aggregate across seeds
agg = {}
for m, s, lv, h, p, w, wk in rows:
    agg.setdefault((m, lv, h), []).append((p, w, wk))
with open(OUT / "uq_aggregate.csv", "w") as f:
    f.write("method,level,horizon,picp_mean,picp_std,mpiw_mean,mpiw_std,winkler_mean,winkler_std,n_seeds\n")
    for (m, lv, h), vals in agg.items():
        a = np.array(vals)
        f.write(f"{m},{lv},{h},{a[:,0].mean():.4f},{a[:,0].std():.4f},"
                f"{a[:,1].mean():.3f},{a[:,1].std():.3f},"
                f"{a[:,2].mean():.3f},{a[:,2].std():.3f},{len(vals)}\n")

with open(OUT / "aci_sensitivity.csv", "w") as f:
    f.write("method,horizon,gamma,window,picp,mpiw,winkler\n")
    for m, h, g, w_, p, w, wk in sens_rows:
        f.write(f"{m},{h},{g},{w_},{p:.4f},{w:.3f},{wk:.3f}\n")

np.savez(OUT / "rolling_coverage.npz", **{k: v for k, v in rolling.items()})

# human-readable summary at 90%
lines = ["UQ evaluation — aggregate across seeds, nominal 90% coverage", ""]
lines.append(f"{'method':16s} " + " ".join(f"{h:>16s}" for h in HORIZONS))
order = ["mc_gauss", "mc_quant", "scp_lstm", "scp_ridge", "scp_persistence",
         "scp_mc_sigma", "aci_lstm", "aci_ridge", "aci_persistence", "aci_mc_sigma"]
for m in order:
    cells = []
    for h in HORIZONS:
        a = np.array(agg.get((m, 0.90, h), [(np.nan,) * 3]))
        cells.append(f"{a[:,0].mean()*100:5.1f}%/{a[:,1].mean():6.2f}")
    lines.append(f"{m:16s} " + " ".join(f"{c:>16s}" for c in cells))
lines.append("")
lines.append("cells: mean PICP / mean MPIW (% units) across seeds where applicable")
summary = "\n".join(lines)
with open(OUT / "uq_summary.txt", "w") as f:
    f.write(summary + "\n")
print(summary)
