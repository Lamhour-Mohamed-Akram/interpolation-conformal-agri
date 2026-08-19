"""Matched-model-definition naive-versus-clean distribution-shift ablation (Morocco).

Scientific purpose
------------------
Isolates how much of the apparent validation-to-test residual shift is produced
by the preprocessing pipeline rather than by the data or the forecaster. The
same model definitions and hyperparameters are evaluated under both pipelines
(persistence is identical by construction, as no fitting occurs; ridge is
refitted on each arm's own training set), so a change in model class or
hyperparameters cannot account for any difference between the arms — the
comparison attributes the difference to the pipeline as a whole, including the
training sample it induces.

naive arm   resample to 5-min medians -> physical-bounds screening ->
            time-interpolation with NO cap (plus edge fill) -> clip ->
            24-step windows at every index -> targets shifted before the split
            -> 60/20/20 index split, cross-boundary sequences included
clean arm   identical resampling and bounds, but whole-gap interpolation capped
            at 30 min, gap-aware admissible windows, and the 60/20/20 split
            taken on the admissible-sequence stream with chronological target
            isolation (splits.isolated_split): sequences whose target crosses
            their block's boundary are excluded from that block

Model definitions are identical in both arms: persistence is the last target
value of the input window, and ridge (alpha=1.0) is fitted on the flattened
24x6 raw window using each arm's own training split.

Reported per base and horizon: median absolute residual on validation and on
test, and their ratio. The ratio is stored unrounded; any rounding happens once,
at display time, in generate_tables.py.

Input       data/data.csv.gz or data/data.csv (either form is accepted)
Output      results/same_model_shift/same_model_shift.csv
Manuscript  the same-model shift table and the naive-vs-clean ratio ranges
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import (load_clean, build_horizon, split_idx, STEPS,      # noqa: E402
                        HORIZONS_H, RAW_CSV, FEATURE_COLS, TARGET_COL,
                        DATE_COL, SEQ_LEN, PHYSICAL_BOUNDS,
                        TRAIN_RATIO, VAL_RATIO)
from splits import isolated_split                                         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "same_model_shift"


def load_naive():
    """Same parsing and bounds screening as the clean arm, but uncapped interpolation."""
    df = pd.read_csv(RAW_CSV, encoding="latin1")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
    df = df.drop_duplicates(subset=[DATE_COL], keep="last").set_index(DATE_COL)
    df = df[FEATURE_COLS].copy()
    for c, (lo, hi) in PHYSICAL_BOUNDS.items():
        df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
    res = df.resample("5min").median()
    res = res.interpolate(method="time").ffill().bfill()      # uncapped: the naive step
    for c, (lo, hi) in PHYSICAL_BOUNDS.items():
        res[c] = res[c].clip(lo, hi)
    return res


def build_naive(res, steps):
    """Windows at every index, with targets shifted before any split is applied.

    The naive arm keeps its studied defects, but the horizon itself is exact:
    the target lies ``steps`` bins after the origin (the final input row),
    identical to the clean arm's convention.
    """
    feat = res[FEATURE_COLS].to_numpy(np.float32)
    targ = res[TARGET_COL].to_numpy(np.float32)
    N = len(res)
    idx = np.arange(N - SEQ_LEN - steps + 1)
    X = np.stack([feat[i:i + SEQ_LEN] for i in idx]).astype(np.float32)
    return X, targ[idx + SEQ_LEN - 1 + steps], targ[idx + SEQ_LEN - 1]


def arm_metrics(X, y, pers, masks=None):
    """Median absolute validation/test residuals and their ratio, per base model.

    ``masks`` is (train, val, test) boolean masks for the clean arm's
    target-isolated split; the naive arm passes None and keeps its defining
    index-only split (cross-boundary sequences included, as the defect under
    study).
    """
    n = len(y)
    if masks is None:
        s_tr, s_va, s_te = split_idx(n)
        m_tr = np.zeros(n, bool); m_tr[s_tr] = True
        m_va = np.zeros(n, bool); m_va[s_va] = True
        m_te = np.zeros(n, bool); m_te[s_te] = True
    else:
        m_tr, m_va, m_te = masks
    rg = Ridge(alpha=1.0).fit(X[m_tr].reshape(m_tr.sum(), -1), y[m_tr])
    rid = rg.predict(X.reshape(len(X), -1)).astype(np.float32)
    out = {}
    for name, pred in [("persistence", pers), ("ridge", rid)]:
        va = float(np.median(np.abs(y[m_va] - pred[m_va])))
        te = float(np.median(np.abs(y[m_te] - pred[m_te])))
        out[name] = (va, te, te / max(va, 1e-9))
    return n, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res_c, _, valid_win, valid_tgt = load_clean()
    res_n = load_naive()
    rows = []
    print("Same-model naive vs clean shift ratios (Morocco)")
    print(f"{'base':12s}{'H':>4} {'naive val/test/ratio':>26} {'clean val/test/ratio':>26}")
    for h in HORIZONS_H:
        Xn, yn, pn = build_naive(res_n, STEPS[h])
        Xc, yc, pc, _, ob, tb = build_horizon(res_c, valid_win, valid_tgt, STEPS[h])
        m_tr, m_va, m_te, _ = isolated_split(ob, tb, TRAIN_RATIO, VAL_RATIO)
        n_n, mn = arm_metrics(Xn, yn, pn)
        n_c, mc = arm_metrics(Xc, yc, pc, masks=(m_tr, m_va, m_te))
        for base in ("persistence", "ridge"):
            nv, nt, nr = mn[base]
            cv, ct, cr = mc[base]
            rows.append(dict(base=base, horizon_h=h, n_naive=n_n, n_clean=n_c,
                             naive_val_med=nv, naive_test_med=nt, naive_ratio=nr,
                             clean_val_med=cv, clean_test_med=ct, clean_ratio=cr))
            print(f"{base:12s}{h:>3}h {nv:>8.2f}{nt:>8.2f}{nr:>8.1f}x "
                  f"{cv:>10.2f}{ct:>8.2f}{cr:>8.1f}x")
    pd.DataFrame(rows).to_csv(OUT / "same_model_shift.csv", index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}/same_model_shift.csv")


if __name__ == "__main__":
    main()
