"""Robustness of the two evaluation defects across sites and split boundaries.

Scientific purpose
------------------
The premature-feedback (immediate) defect and the target-boundary leakage defect are each
demonstrated on one site at one split. This experiment repeats both across three
sites and three chronological split boundaries (60/20, 50/25, 70/15) to show
that neither is an artefact of a particular record or of one arbitrary split
position.

For every (site, split) pair at the 48 h horizon it reports

  ACI arm       coverage with immediate feedback and with availability-aware
                feedback (conformal.aci, feedback_mode="observable"), and the
                difference between them. Both variants run under the
                target-isolated split, so the calibration pool contains only
                outcomes observable before the first test origin and the two
                arms differ ONLY in feedback timing.
  leakage arm   the number of leaking training sequences, and split-conformal
                coverage under the leaky and target-isolated splits

The ridge base is used in both arms so the comparison is against a fitted model
rather than only a training-free one.

Input       data/*.csv (Iraq is skipped when absent)
Output      results/boundary_robustness/boundary_robustness.csv
Manuscript  the breadth table supporting the robustness claim
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import aci, scp_quantile, ALPHA                # noqa: E402
from sites import available_sites, load_site, build_sequences  # noqa: E402
from splits import isolated_split                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "boundary_robustness"
SPLITS = [(0.6, 0.2), (0.5, 0.25), (0.7, 0.15)]
HORIZON = 48


def scp_coverage(res_cal, yhat, y):
    q = scp_quantile(res_cal, ALPHA)
    return float(((y >= yhat - q) & (y <= yhat + q)).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for (name, fname, dcol, target, feats, rule, cap, enc, bounds, steps_map) in available_sites():
        res, valid_win, valid_tgt = load_site(fname, dcol, target, feats, rule, cap, enc, bounds)
        steps = steps_map[HORIZON]
        X, y, pers, origin_bin, target_bin = build_sequences(res, valid_win, valid_tgt,
                                                             feats, target, steps)
        n = len(y)
        flat = X.reshape(len(X), -1)
        for (a, b) in SPLITS:
            tr = int(n * a); va = tr + int(n * b)
            clean_tr, clean_va, m_te, st = isolated_split(origin_bin, target_bin, a, b)
            n_leaking = st["n_train_dropped"]
            rg2 = Ridge(alpha=1.0).fit(flat[clean_tr], y[clean_tr])
            pred_c = rg2.predict(flat).astype(np.float32)

            # ACI arm: identical isolated calibration pool (legal by
            # construction), immediate vs availability-aware feedback only
            assert target_bin[clean_va].max() < origin_bin[m_te][0]
            res_cal_iso = y[clean_va] - pred_c[clean_va]
            cov_imm, _ = aci(res_cal_iso, pred_c[m_te], y[m_te],
                             feedback_mode="immediate")
            cov_del, _ = aci(res_cal_iso, pred_c[m_te], y[m_te],
                             origin_bin=origin_bin[m_te], target_bin=target_bin[m_te],
                             feedback_mode="observable")

            # leakage arm: leaky index split vs the target-isolated split
            rg = Ridge(alpha=1.0).fit(flat[:tr], y[:tr])
            pred = rg.predict(flat).astype(np.float32)
            cov_leaky = scp_coverage(y[tr:va] - pred[tr:va], pred[va:], y[va:])
            cov_iso = scp_coverage(res_cal_iso, pred_c[m_te], y[m_te])

            rows.append(dict(site=name, horizon_h=HORIZON, train_frac=a, val_frac=b,
                             n_seq=n, cov_aci_immediate=cov_imm, cov_aci_delayed=cov_del,
                             aci_bias_pp=100 * (cov_imm - cov_del),
                             n_leaking_train_seq=n_leaking,
                             picp_leaky=cov_leaky, picp_isolated=cov_iso))
            print(f"{name:14s} split {a:.2f}/{b:.2f}  ACI {100*cov_imm:.0f}->{100*cov_del:.0f}% "
                  f"(bias {100*(cov_imm-cov_del):+.1f} pp)  leakage {n_leaking:4d} seq, "
                  f"PICP {100*cov_leaky:.0f}->{100*cov_iso:.0f}%")
    pd.DataFrame(rows).to_csv(OUT / "boundary_robustness.csv", index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}/boundary_robustness.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
