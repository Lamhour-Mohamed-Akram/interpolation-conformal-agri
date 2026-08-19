"""Target-boundary leakage versus target-isolated chronological splitting.

Scientific purpose
------------------
When multi-horizon targets are created by shifting the full series before the
chronological split is applied, a training sequence's h-step-ahead target can
fall inside the validation period. The split then looks clean at the level of
sequence indices while the targets themselves cross the boundary.

This experiment quantifies the difference on the gap-cleaned 5-min Morocco
record:

leaky split      sequences are split by index only, so a training sequence may
                 carry a target beyond its own block's origin boundary
isolated split   after the index split, any sequence whose TARGET bin falls
                 beyond its block's origin boundary is dropped

Both arms are otherwise identical (same ridge base, same evaluation set, same
split-conformal calibration). The number of leaking training sequences grows
with the horizon, because a longer horizon pushes more targets across the
boundary.

Input       data/data.csv.gz (or data/data.csv) via preprocess.load_clean()
Computation ridge fitted per arm, split-conformal coverage and RMSE on the test
            block
Output      results/target_isolation/target_isolation.csv
Manuscript  leaking-sequence counts and the leaky-vs-isolated coverage figure
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import scp_quantile, ALPHA                     # noqa: E402
from preprocess import load_clean, build_horizon, STEPS, HORIZONS_H   # noqa: E402
from splits import isolated_split                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "target_isolation"


def coverage_and_rmse(res_cal, yhat, y):
    q = scp_quantile(res_cal, ALPHA)
    picp = float(((y >= yhat - q) & (y <= yhat + q)).mean())
    return picp, float(np.sqrt(np.mean((y - yhat) ** 2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res, _, valid_win, valid_tgt = load_clean()
    rows = []
    for h in HORIZONS_H:
        X, y, pers, _, origin_bin, target_bin = build_horizon(res, valid_win, valid_tgt, STEPS[h])
        n = len(y)
        tr, va = int(n * 0.6), int(n * 0.8)

        # leaky arm: index split only (the defect under study)
        leaky_tr = np.zeros(n, bool); leaky_tr[:tr] = True
        rg = Ridge(alpha=1.0).fit(X[leaky_tr].reshape(leaky_tr.sum(), -1), y[leaky_tr])
        pred_leaky = rg.predict(X.reshape(len(X), -1)).astype(np.float32)
        picp_leaky, rmse_leaky = coverage_and_rmse(
            y[tr:va] - pred_leaky[tr:va], pred_leaky[va:], y[va:])

        # isolated arm: the shared clean-protocol rule
        clean_tr, clean_va, m_te, st = isolated_split(origin_bin, target_bin, 0.6, 0.2)
        n_leaking = st["n_train_dropped"]
        rg2 = Ridge(alpha=1.0).fit(X[clean_tr].reshape(clean_tr.sum(), -1), y[clean_tr])
        pred_clean = rg2.predict(X.reshape(len(X), -1)).astype(np.float32)
        picp_iso, rmse_iso = coverage_and_rmse(
            y[clean_va] - pred_clean[clean_va], pred_clean[m_te], y[m_te])

        rows.append(dict(horizon_h=h, n_seq=n, n_test=n - va,
                         n_train_leaky=int(leaky_tr.sum()),
                         n_train_isolated=st["n_train"],
                         n_leaking_train_seq=n_leaking,
                         n_cal_leaky=va - tr, n_cal_isolated=st["n_cal"],
                         picp_leaky=picp_leaky, picp_isolated=picp_iso,
                         rmse_leaky=rmse_leaky, rmse_isolated=rmse_iso))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "target_isolation.csv", index=False)

    print("Target-boundary leakage vs isolation (gap-cleaned Morocco, ridge base)")
    print(f"{'H':>4} {'train leaky/iso':>18} {'leaking':>8} {'PICP leaky/iso':>18} {'RMSE leaky/iso':>16}")
    for _, r in df.iterrows():
        print(f"{int(r.horizon_h):>3}h {f'{int(r.n_train_leaky)}/{int(r.n_train_isolated)}':>18} "
              f"{int(r.n_leaking_train_seq):>8} "
              f"{f'{100*r.picp_leaky:.1f}/{100*r.picp_isolated:.1f}%':>18} "
              f"{f'{r.rmse_leaky:.2f}/{r.rmse_isolated:.2f}':>16}")
    print(f"\nwrote {OUT.relative_to(ROOT)}/target_isolation.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
