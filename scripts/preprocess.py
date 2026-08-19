"""Gap-aware preprocessing of the Morocco greenhouse record.

Scientific purpose
------------------
Sparse, irregularly logged sensor records are routinely resampled onto a regular
grid and interpolated without a cap, which converts long acquisition outages
into smooth synthetic segments. This module implements the alternative used
throughout the paper: resample to a fixed 5-min grid, interpolate ONLY missing
runs whose ENTIRE length is at most ``MAX_GAP_MIN`` minutes, and leave longer
runs entirely NaN so that they break the sequence stream instead of being
silently filled.

Whole-gap rule
--------------
The cap applies to the whole missing run, not to a fill budget:
a contiguous NaN run is interpolated only when its full length is <= the cap
AND it is bounded by observed values on both sides; a longer run keeps every
one of its bins NaN, and unbounded leading/trailing runs are never
extrapolated. (``pandas.interpolate(limit=N)`` does NOT implement this rule:
it partially fills the first N bins of arbitrarily long runs.)
``interpolate_capped`` is the canonical implementation used by every capped
loader in this repository.

A sequence is retained only if its whole input window and its h-step-ahead
target bin are observed (or short-interpolated) data, so no fully synthetic
straight line enters either the inputs or the targets. The chronological split
is taken on the stream of valid sequences, and split boundaries are enforced
with target isolation (see splits.py): a sequence whose target crosses its
block's boundary is excluded from that block.

Inputs
------
data/data.csv.gz : raw greenhouse log (irregular cadence, latin-1 encoded).
                   The plain data/data.csv is used instead when present;
                   data_file() resolves either form.

Parameters
----------
SEQ_LEN       length of the input window, in 5-min bins (24 bins = 2 h)
HORIZONS_H    forecast horizons in hours
MAX_GAP_MIN   longest missing run that may be interpolated (minutes)
TRAIN/VAL     chronological split fractions

Outputs (when run as a script)
------------------------------
results/interpolation_audit/daily_row_counts.csv   rows logged per calendar day
results/interpolation_audit/acquisition_summary.csv aggregate acquisition profile
results/interpolation_audit/grid_summary.csv       observed vs interpolated bins
results/interpolation_audit/shift_diagnostic.csv   validation->test residual
                                                   medians for the two
                                                   training-free bases

Run: python scripts/preprocess.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_file(name: str) -> Path:
    """Locate a raw record, accepting either the plain or gzip-compressed form.

    The Morocco record is distributed gzip-compressed because it is large;
    pandas reads either form transparently, so no caller needs to care which is
    present.
    """
    base = ROOT / "data" / name
    for candidate in (base, base.with_name(base.name + ".gz")):
        if candidate.exists():
            return candidate
    return base


RAW_CSV = data_file("data.csv")
OUT = ROOT / "results" / "interpolation_audit"

FEATURE_COLS = ["humidity", "temperature", "humiditysol", "temperaturesol", "co2", "lumière"]
TARGET_COL = "humiditysol"
DATE_COL = "date"
SEQ_LEN = 24
HORIZONS_H = (6, 12, 24, 48)
STEPS = {h: h * 12 for h in HORIZONS_H}
MAX_GAP_MIN = 30                       # interpolate whole runs <= 30 min (6 bins); longer stay NaN
MAX_GAP_BINS = MAX_GAP_MIN // 5
TRAIN_RATIO, VAL_RATIO = 0.6, 0.2
PHYSICAL_BOUNDS = {
    "humidity": (0.0, 100.0), "humiditysol": (0.0, 100.0),
    "temperature": (-10.0, 60.0), "temperaturesol": (-10.0, 60.0),
    "co2": (0.0, 5000.0), "lumière": (0.0, 200000.0),
}


def nan_runs(isna: np.ndarray):
    """Contiguous NaN runs of a boolean missing mask, as (start, stop) pairs."""
    idx = np.flatnonzero(np.asarray(isna, bool))
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    return [(run[0], run[-1] + 1) for run in np.split(idx, breaks)]


def interpolate_capped(res: pd.DataFrame, cap_bins: int) -> pd.DataFrame:
    """Whole-gap-aware capped interpolation on a regular-grid frame.

    For each column independently:
    * a contiguous NaN run of length <= ``cap_bins`` that is bounded by
      observed values on both sides is time-interpolated in full;
    * a run longer than ``cap_bins`` keeps ALL of its bins NaN (no partial
      filling of long runs);
    * unbounded leading/trailing runs are never extrapolated.

    This is the canonical capped-interpolation rule; every loader in the
    repository that claims a gap cap goes through this function.
    """
    out = res.copy()
    for c in out.columns:
        col = out[c]
        isna = col.isna().to_numpy()
        if not isna.any():
            continue
        # limit_area="inside" guarantees no extrapolation across the edges;
        # the interpolation itself is uncapped and long runs are re-blanked
        # in full below, so no run is ever partially filled.
        filled = col.interpolate(method="time", limit_area="inside").to_numpy(copy=True)
        for start, stop in nan_runs(isna):
            if stop - start > cap_bins:
                filled[start:stop] = np.nan
        out[c] = filled
    return out


def load_raw() -> pd.DataFrame:
    """Read the raw log, parse timestamps and drop unparseable/duplicate rows."""
    df = pd.read_csv(RAW_CSV, encoding="latin1")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)


def load_clean_capped(cap_bins: int):
    """Resample to the 5-min grid with whole-gap capped interpolation.

    Returns
    -------
    res       : DataFrame on the regular 5-min grid, short gaps interpolated
    observed  : bool array, True where the target bin holds real data
    valid_win : bool array, True where EVERY feature is real or
                short-interpolated (admissible input-window bin)
    valid_tgt : bool array, True where the target is real or short-interpolated
                (admissible target bin)
    """
    df = load_raw().drop_duplicates(subset=[DATE_COL], keep="last").set_index(DATE_COL)
    df = df[FEATURE_COLS].copy()
    for c, (lo, hi) in PHYSICAL_BOUNDS.items():
        df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
    res = df.resample("5min").median()
    observed = res[TARGET_COL].notna().to_numpy()          # real bins, before interpolation
    res_i = interpolate_capped(res, cap_bins)
    valid_win = res_i[FEATURE_COLS].notna().all(axis=1).to_numpy()
    valid_tgt = res_i[TARGET_COL].notna().to_numpy()
    for c, (lo, hi) in PHYSICAL_BOUNDS.items():
        res_i[c] = res_i[c].clip(lo, hi)
    return res_i, observed, valid_win, valid_tgt


def load_clean():
    """Whole-gap capped preprocessing at the paper's 30-minute cap."""
    return load_clean_capped(MAX_GAP_BINS)


def build_horizon(res, valid_win, valid_tgt, steps):
    """Valid single-horizon sequences, with each sequence's origin/target bin.

    Horizon convention (used everywhere in this repository): the forecast
    ORIGIN is the final observed bin of the input window,
    origin_bin = i + SEQ_LEN - 1, and the h-step-ahead TARGET is exactly
    ``steps`` grid bins after that origin,

        target_bin = origin_bin + steps,

    so a nominal 6-hour horizon on the 5-minute grid is exactly 72 bins
    (6 h 00 min), never 73.

    A sequence is kept when its input bins [i, i+SEQ_LEN) are all
    window-admissible and its target bin is target-admissible. Origin and
    target bins are returned so callers can enforce chronological target
    isolation at split boundaries.

    Returns X (n, SEQ_LEN, n_feat), y (n,), persistence (n,), target dates (n,),
    origin_bin (n,), target_bin (n,).
    """
    feat = res[FEATURE_COLS].to_numpy(np.float32)
    targ = res[TARGET_COL].to_numpy(np.float32)
    dates = res.index.to_numpy()
    N = len(res)
    Xs, ys, pers, ds, ob, tb_ = [], [], [], [], [], []
    csum = np.concatenate([[0], np.cumsum(valid_win.astype(np.int64))])   # O(1) window check
    for i in range(N - SEQ_LEN - steps + 1):
        o = i + SEQ_LEN - 1
        tb = o + steps
        win_ok = (csum[i + SEQ_LEN] - csum[i]) == SEQ_LEN
        if not win_ok or not valid_tgt[tb]:
            continue
        Xs.append(feat[i:i + SEQ_LEN]); ys.append(targ[tb])
        pers.append(targ[o]); ds.append(dates[tb])
        ob.append(o); tb_.append(tb)
    ob = np.asarray(ob, np.int64)
    tb_ = np.asarray(tb_, np.int64)
    # horizon-semantics invariants: exact bin count and exact grid time
    assert np.all(tb_ - ob == steps), "horizon bin convention violated"
    if len(ob):
        step_delta = dates[1] - dates[0]
        assert np.all((dates[tb_] - dates[ob]) == steps * step_delta), \
            "horizon timestamp convention violated"
    return (np.asarray(Xs, np.float32), np.asarray(ys, np.float32),
            np.asarray(pers, np.float32), np.asarray(ds), ob, tb_)


def split_idx(n):
    """Chronological train/validation/test slices over the valid-sequence stream."""
    tr = int(n * TRAIN_RATIO); va = tr + int(n * VAL_RATIO)
    return slice(0, tr), slice(tr, va), slice(va, n)


def acquisition_audit(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows logged per calendar day, and the aggregate acquisition profile.

    The logger's cadence is highly non-uniform: a short dense window holds most
    of the record while several calendar days carry no rows at all. Both the
    per-day series and the summary are written so the acquisition claims in the
    paper can be recomputed from the raw file.
    """
    day = raw[DATE_COL].dt.floor("D")
    counts = day.value_counts().sort_index()
    full = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    counts = counts.reindex(full, fill_value=0)
    daily = pd.DataFrame({"date": counts.index.strftime("%Y-%m-%d"),
                          "row_count": counts.to_numpy(int)})

    dense = daily[(daily.date >= "2022-04-28") & (daily.date <= "2022-05-10")]
    peak = dense[dense.row_count > 70_000]
    summary = pd.DataFrame([dict(
        total_rows=int(daily.row_count.sum()),
        n_days=len(daily),
        dense_window="2022-04-28..2022-05-10",
        dense_days=len(dense),
        dense_rows=int(dense.row_count.sum()),
        dense_share_pct=round(100 * dense.row_count.sum() / daily.row_count.sum(), 1),
        peak_days=int(len(peak)),
        peak_mean_rows_per_day=int(round(peak.row_count.mean())),
        peak_median_rows_per_day=int(peak.row_count.median()),
        zero_row_dates=";".join(daily[daily.row_count == 0].date.tolist()))])
    return daily, summary


def main():
    from sklearn.linear_model import Ridge
    from splits import isolated_split

    OUT.mkdir(parents=True, exist_ok=True)

    daily, summary = acquisition_audit(load_raw())
    daily.to_csv(OUT / "daily_row_counts.csv", index=False)
    summary.to_csv(OUT / "acquisition_summary.csv", index=False)
    s = summary.iloc[0]
    print(f"raw rows {s.total_rows:,} over {s.n_days} days | "
          f"{s.dense_days}-day dense window holds {s.dense_share_pct}% | "
          f"days with no rows: {s.zero_row_dates}")

    res, observed, valid_win, valid_tgt = load_clean()
    grid = pd.DataFrame([dict(
        n_bins=len(res),
        observed_bins=int(observed.sum()),
        observed_pct=round(100 * observed.mean(), 1),
        valid_bins=int(valid_tgt.sum()),
        valid_pct=round(100 * valid_tgt.mean(), 1),
        dropped_long_gap_bins=int((~valid_tgt).sum()),
        max_gap_min=MAX_GAP_MIN)])
    grid.to_csv(OUT / "grid_summary.csv", index=False)
    print(f"5-min bins {len(res)} | observed {observed.sum()} ({100*observed.mean():.1f}%) | "
          f"valid {valid_tgt.sum()} ({100*valid_tgt.mean():.1f}%) | long-gap {int((~valid_tgt).sum())}")

    # validation -> test residual medians for the two training-free bases,
    # under the target-isolated chronological split
    rows = []
    for h in HORIZONS_H:
        X, y, pers, _, ob, tb = build_horizon(res, valid_win, valid_tgt, STEPS[h])
        n = len(y)
        m_tr, m_va, m_te, st = isolated_split(ob, tb, TRAIN_RATIO, VAL_RATIO)
        rg = Ridge(alpha=1.0).fit(X[m_tr].reshape(m_tr.sum(), -1), y[m_tr])
        pr = rg.predict(X.reshape(n, -1))
        for base, pred in [("persistence", pers), ("ridge", pr)]:
            rv = np.abs(y[m_va] - pred[m_va])
            rt = np.abs(y[m_te] - pred[m_te])
            rows.append(dict(horizon_h=h, base=base, n_seq=n,
                             n_train=int(m_tr.sum()), n_val=int(m_va.sum()),
                             n_test=int(m_te.sum()),
                             val_res_median=float(np.median(rv)),
                             test_res_median=float(np.median(rt)),
                             ratio=float(np.median(rt) / max(np.median(rv), 1e-9))))
    shift = pd.DataFrame(rows)
    shift.to_csv(OUT / "shift_diagnostic.csv", index=False)
    print(shift.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote 4 files -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
