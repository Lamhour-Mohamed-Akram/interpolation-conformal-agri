"""Site definitions and generic loading for the cross-site experiments.

Three records are used for the cross-site checks. They differ in cadence,
target variable and sensor suite, which is the point: an evaluation defect that
survives all three is not a property of one installation.

Morocco       greenhouse, 5-min logging, soil-moisture target
Iraq          greenhouse, 5-min logging, water-level target
Johannesburg  field station, hourly logging, soil-moisture target

Each site is preprocessed with the same gap-aware rule used for Morocco
(preprocess.interpolate_capped): resample to the site's native cadence, screen
physically impossible values where bounds are known, and interpolate only
missing runs whose WHOLE length is at most the site's cap (30 min for the
5-min sites, 2 h for the hourly site); a longer run keeps every bin NaN.

The Iraq record is not redistributed with this repository because its
redistribution terms are not stated explicitly; see data/README.md for the
download link and checksum. If a record is absent the remaining sites still
run, but the omission is reported prominently so that a partial run is never
mistaken for a full reproduction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import data_file, interpolate_capped           # noqa: E402
from preprocess import PHYSICAL_BOUNDS as MOROCCO_BOUNDS       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Records that are not redistributed with this repository, and how to obtain them
EXTERNAL_SOURCES = {
    "iraq_IoTProcessed_Data.csv": (
        "IoT Agriculture 2024 (W. D. Abdullah, Tikrit University)\n"
        "        download: https://www.kaggle.com/datasets/wisam1985/iot-agriculture-2024\n"
        "        save the processed file as: data/iraq_IoTProcessed_Data.csv\n"
        "        expected SHA-256: b6485b4322662ec6beeb986bf4adf2dfccfd3c3cf2e0bbe514439d3b517e8a82"),
}

MOROCCO_FEATURES = ["humidity", "temperature", "humiditysol", "temperaturesol", "co2", "lumière"]
SEQ_LEN = 24

# name, filename, date column, target, features, resample rule, interpolation cap
# (in bins), encoding, physical bounds, horizon -> steps
SITES = [
    ("Morocco", "data.csv", "date", "humiditysol", MOROCCO_FEATURES, "5min", 6, "latin1",
     MOROCCO_BOUNDS,                       # identical screening to preprocess.py, so the
                                           # Morocco rows here match cap_sensitivity's
     {6: 72, 12: 144, 24: 288, 48: 576}),
    ("Iraq", "iraq_IoTProcessed_Data.csv", "date", "water_level",
     ["tempreature", "humidity", "water_level"], "5min", 6, "utf-8", None,
     {6: 72, 12: 144, 24: 288, 48: 576}),
    ("Johannesburg", "mendeley_soil_data_incl_rain_v3.csv", "Time", "Soil_Moisture",
     ["Humidity", "Atmospheric_Temp", "Soil_Temp", "Soil_Moisture", "Dew_Point"], "1h", 2,
     "utf-8", None, {6: 6, 12: 12, 24: 24, 48: 48}),
]


def available_sites():
    """Site definitions whose raw file is present, reporting any that are not.

    A missing record is announced loudly rather than passed over, so that a run
    covering fewer sites than the paper can never be mistaken for a complete
    one. The instructions for obtaining the record are printed with it.
    """
    out, missing = [], []
    for site in SITES:
        if data_file(site[1]).exists():
            out.append(site)
        else:
            missing.append(site)
    if missing:
        print("=" * 72)
        print(f"NOT REPRODUCED: {len(missing)} of {len(SITES)} sites are missing their raw data.")
        for name, fname, *_ in missing:
            print(f"\n  {name}  (expects data/{fname})")
            print(f"        {EXTERNAL_SOURCES.get(fname, 'see data/README.md')}")
        print("\n  Results below cover only: " + ", ".join(s[0] for s in out))
        print("  The shipped files under results/ DO include the missing site(s).")
        print("=" * 72)
    return out


def load_site(filename, date_col, target, features, rule, cap_bins, encoding, bounds):
    """Resample one site onto its regular grid with whole-gap capped interpolation.

    Returns the resampled frame, the window-admissibility mask (every feature
    real or short-interpolated) and the target-admissibility mask.
    """
    df = pd.read_csv(data_file(filename), encoding=encoding)
    df.columns = df.columns.str.strip()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = (df.dropna(subset=[date_col]).sort_values(date_col)
            .drop_duplicates(date_col, keep="last").set_index(date_col))
    df = df[features].copy()
    if bounds:
        for c, (lo, hi) in bounds.items():
            df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
    res = interpolate_capped(df.resample(rule).median(), cap_bins)
    valid_win = res[features].notna().all(axis=1).to_numpy()
    valid_tgt = res[target].notna().to_numpy()
    return res, valid_win, valid_tgt


def build_sequences(res, valid_win, valid_tgt, features, target, steps):
    """Gap-aware sequences, plus each sequence's origin bin and target bin.

    Horizon convention (identical to preprocess.build_horizon): the forecast
    origin is the final observed bin of the input window and the target is
    exactly ``steps`` grid bins after it, target_bin = origin_bin + steps —
    a nominal 6-hour horizon on an hourly grid is exactly 6 bins, never 7.

    A sequence is admissible when its input window is entirely
    window-admissible and its target bin is target-admissible. The origin and
    target bins are returned so callers can enforce chronological target
    isolation at split boundaries (see splits.py).
    """
    feat = res[features].to_numpy(np.float32)
    targ = res[target].to_numpy(np.float32)
    dates = res.index.to_numpy()
    N = len(res)
    csum = np.concatenate([[0], np.cumsum(valid_win.astype(np.int64))])
    X, y, pers, origin_bin, target_bin = [], [], [], [], []
    for i in range(N - SEQ_LEN - steps + 1):
        o = i + SEQ_LEN - 1
        t = o + steps
        if (csum[i + SEQ_LEN] - csum[i]) != SEQ_LEN or not valid_tgt[t]:
            continue
        X.append(feat[i:i + SEQ_LEN]); y.append(targ[t]); pers.append(targ[o])
        origin_bin.append(o); target_bin.append(t)
    origin_bin = np.array(origin_bin, np.int64)
    target_bin = np.array(target_bin, np.int64)
    assert np.all(target_bin - origin_bin == steps), "horizon bin convention violated"
    if len(origin_bin):
        step_delta = dates[1] - dates[0]
        assert np.all((dates[target_bin] - dates[origin_bin]) == steps * step_delta), \
            "horizon timestamp convention violated"
    return (np.array(X, np.float32), np.array(y, np.float32), np.array(pers, np.float32),
            origin_bin, target_bin)
