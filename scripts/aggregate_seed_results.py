"""Per-seed metric tables for the 20-seed deep case study.

Scientific purpose
------------------
Turns the raw per-seed outputs of the training scripts into tidy per-seed metric
tables, so every reported deep-model mean and standard deviation can be traced
back to the individual runs behind it, and so it is visible that no run was
dropped.

For each site, seed, horizon and model variant the tables record RMSE, R2 and
(for Morocco) MAE, the number of epochs actually run, and whether the run
converged. Convergence is recorded rather than used as a filter: no seed is
excluded from any reported aggregate, so a seed that trained poorly still
contributes to the spread.

Standard deviations reported downstream use the sample convention (ddof=1).

Input       results/naive_pipeline/ (Morocco), results/multisite/<site>/,
            results/deep_20seed/seed_list.json
Output      results/deep_20seed/morocco_seed_results.csv
            results/deep_20seed/iraq_lstm_20seed.csv
            results/deep_20seed/johannesburg_lstm_20seed.csv
            results/deep_20seed/training_runs.csv
Manuscript  the per-seed spread quoted for every deep-model result
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "deep_20seed"
NAIVE = ROOT / "results" / "naive_pipeline"
MULTI = ROOT / "results" / "multisite"
HORIZON_NAMES = ["6h", "12h", "24h", "48h"]

SEEDS = json.load(open(OUT / "seed_list.json"))["seeds"]

rows, runs = [], []

# ---------------------------------------------------------------- Morocco --
metrics = json.load(open(NAIVE / "point_metrics.json"))
for s in SEEDS:
    per_seed = metrics["per_seed"][str(s)] if str(s) in metrics["per_seed"] else metrics["per_seed"][s]
    for variant in ("deterministic", "bayesian"):
        history = json.load(open(NAIVE / f"seed_{s}" / f"history_{variant}.json"))
        n_epochs = len(history["loss"])
        nan_loss = any(np.isnan(v) for v in history["loss"] + history["val_loss"])
        for h in HORIZON_NAMES:
            m = per_seed[variant][h]
            rows.append(dict(site="morocco", seed=s, horizon=h, model=variant,
                             rmse=m["rmse"], r2=m["r2"], mae=m["mae"],
                             epochs_run=n_epochs,
                             converged=bool(np.isfinite(m["rmse"]) and not nan_loss)))
        runs.append(dict(site="morocco", seed=s, model=variant, epochs_run=n_epochs,
                         nan_loss=bool(nan_loss), status="ok" if not nan_loss else "nan_loss"))
pd.DataFrame(rows).to_csv(OUT / "morocco_seed_results.csv", index=False)

# ------------------------------------------------------------- other sites --
for site, fname in (("iraq", "iraq_lstm_20seed.csv"),
                    ("mendeley", "johannesburg_lstm_20seed.csv")):
    sdir = MULTI / site
    with open(sdir / "target_scaler.pkl", "rb") as fh:
        target_scaler = pickle.load(fh)
    y_test = target_scaler.inverse_transform(np.load(sdir / "y_test.npy"))
    site_rows = []
    for s in SEEDS:
        for variant, key in (("deterministic", "det_te"), ("bayesian", "bay_te")):
            pred = np.load(sdir / f"seed_{s}" / f"{key}.npy")
            for h_i, h in enumerate(HORIZON_NAMES):
                err = y_test[:, h_i] - pred[:, h_i]
                rmse = float(np.sqrt(np.mean(err ** 2)))
                ss_tot = float(np.sum((y_test[:, h_i] - y_test[:, h_i].mean()) ** 2))
                r2 = float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan")
                site_rows.append(dict(site=site, seed=s, horizon=h, model=variant,
                                      rmse=rmse, r2=r2, converged=bool(np.isfinite(rmse))))
            finite = bool(np.all(np.isfinite(pred)))
            runs.append(dict(site=site, seed=s, model=variant, epochs_run=-1,
                             nan_loss=not finite,
                             status="ok" if finite else "nonfinite_predictions"))
    pd.DataFrame(site_rows).to_csv(OUT / fname, index=False)

pd.DataFrame(runs).to_csv(OUT / "training_runs.csv", index=False)

n_bad = sum(1 for r in runs if r["status"] != "ok")
print(f"seeds: {len(SEEDS)} | training runs recorded: {len(runs)} | non-converged: {n_bad}")
print(f"wrote 4 tables -> {OUT.relative_to(ROOT)}")
