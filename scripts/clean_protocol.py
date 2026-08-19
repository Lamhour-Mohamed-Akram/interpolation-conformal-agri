"""Cross-site evaluation under the leakage-free protocol.

Scientific purpose
------------------
Applies the full protocol (capped interpolation, gap-aware windowing,
chronological target isolation, availability-aware online feedback) to all
available sites and reports what static split-conformal and availability-aware
(observable-feedback) ACI actually deliver once the evaluation defects are
removed.

This is the counterpart to the naive-pipeline results: the same two calibration
methods, the same horizons and the same persistence base, differing only in the
protocol under which they are evaluated.

The chronological split is target-isolated (splits.isolated_split): a
calibration sequence whose target bin falls beyond the test-origin boundary is
excluded from calibration, so no calibration score is computed from a test-side
observation. The counts of sequences removed by that rule are recorded per row.

Reported per site and horizon: split-conformal coverage and mean interval width,
availability-aware ACI coverage and mean interval width, and the isolated
train/calibration/test counts.

Input       data/*.csv (Iraq is skipped when absent)
Output      results/clean_protocol/clean_cross_site.csv
Manuscript  the clean-protocol cross-site table
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import aci, scp_evaluate                        # noqa: E402
from sites import available_sites, load_site, build_sequences  # noqa: E402
from splits import isolated_split                              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "clean_protocol"
HORIZONS = (6, 12, 24, 48)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    sites = available_sites()
    print(f"{'site':14}{'H':>4} {'SCP cov/width':>18} {'observable ACI cov/width':>26}")
    for (name, fname, dcol, target, feats, rule, cap, enc, bounds, steps_map) in sites:
        res, valid_win, valid_tgt = load_site(fname, dcol, target, feats, rule, cap, enc, bounds)
        for h in HORIZONS:
            steps = steps_map[h]
            X, y, pers, ob, tb = build_sequences(res, valid_win, valid_tgt, feats, target, steps)
            n = len(y)
            m_tr, m_cal, m_te, st = isolated_split(ob, tb, 0.6, 0.2)
            # calibration outcomes are all observable before the first test origin
            assert tb[m_cal].max() < ob[m_te][0]
            res_cal = y[m_cal] - pers[m_cal]
            scp_cov, scp_wid = scp_evaluate(res_cal, pers[m_te], y[m_te])
            aci_cov, aci_wid = aci(res_cal, pers[m_te], y[m_te],
                                   origin_bin=ob[m_te], target_bin=tb[m_te],
                                   feedback_mode="observable")
            rows.append(dict(site=name, horizon_h=h, n_seq=n,
                             n_train=st["n_train"], n_cal=st["n_cal"],
                             n_test=st["n_test"],
                             n_train_dropped=st["n_train_dropped"],
                             n_cal_dropped=st["n_cal_dropped"],
                             scp_picp=scp_cov, scp_mpiw=scp_wid,
                             aci_picp=aci_cov, aci_mpiw=aci_wid))
            print(f"{name:14}{h:>3}h {f'{100*scp_cov:.0f}% / {scp_wid:.1f}':>18} "
                  f"{f'{100*aci_cov:.0f}% / {aci_wid:.1f}':>24}")
    pd.DataFrame(rows).to_csv(OUT / "clean_cross_site.csv", index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}/clean_cross_site.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
