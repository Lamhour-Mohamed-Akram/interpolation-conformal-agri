"""Multi-seed deep forecaster training on the two external sites.

Scientific purpose
------------------
Repeats the naive-pipeline deep case study on two independent public records so
that its conclusions are not tied to a single greenhouse:

Iraq          IoT Agriculture 2024, greenhouse, ~5-min logging,
              27 Nov 2023 - 30 Mar 2024, target water_level (%)
Johannesburg  field station, hourly logging, target Soil_Moisture

The protocol, architecture, seed list and training settings are identical to
train_deep_models.py, so differences between sites are attributable to the data
rather than to the setup. All intermediate predictions are saved per site so the
ensemble, EnbPI and decision-simulation analyses can be computed without
retraining.

Input       data/iraq_IoTProcessed_Data.csv (see data/README.md; not
            redistributed with this repository),
            data/mendeley_soil_data_incl_rain_v3.csv
Output      results/multisite/<site>/
Manuscript  the cross-site point-accuracy and calibration tables
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_BASE = ROOT / "results" / "multisite"

# The 20 fixed seeds of the multi-seed protocol. Fixed and published so that
# every reported mean and spread can be reproduced exactly; 7 appears once.
SEEDS = [42, 123, 2024, 7, 99,
         0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
         11, 12, 13, 14, 15]
MC_SAMPLES = 50
SEQ_LEN = 24
HORIZON_NAMES = ["6h", "12h", "24h", "48h"]
LEVEL, Z90 = 0.90, 1.6449
GAMMA, WINDOW = 0.02, 500

SITES = {
    "iraq": {
        "csv": DATA / "iraq_IoTProcessed_Data.csv",
        "date_col": "date",
        "features": ["tempreature", "humidity", "water_level"],
        "target": "water_level",
        "resample": "5min",
        "steps_per_hour": 12,
        "bounds": {"tempreature": (-10, 60), "humidity": (0, 100), "water_level": (0, 100)},
    },
    "mendeley": {
        "csv": DATA / "mendeley_soil_data_incl_rain_v3.csv",
        "date_col": "Time",
        "features": ["Humidity", "Atmospheric_Temp", "Soil_Temp", "Soil_Moisture", "Dew_Point"],
        "target": "Soil_Moisture",
        "resample": "1h",
        "steps_per_hour": 1,
        "bounds": {"Humidity": (0, 100), "Soil_Moisture": (0, 100),
                   "Atmospheric_Temp": (-20, 60), "Soil_Temp": (-20, 60), "Dew_Point": (-30, 60)},
    },
}


def build_lstm(n_features: int, dropout_rate: float) -> Model:
    inputs = Input(shape=(SEQ_LEN, n_features))
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(dropout_rate)(x)
    x = LSTM(32)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(len(HORIZON_NAMES), activation="linear")(x)
    m = Model(inputs=inputs, outputs=outputs)
    m.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return m


def metrics_ph(y, p):
    return {h: {"rmse": float(np.sqrt(mean_squared_error(y[:, i], p[:, i]))),
                "mae": float(mean_absolute_error(y[:, i], p[:, i])),
                "r2": float(r2_score(y[:, i], p[:, i]))}
            for i, h in enumerate(HORIZON_NAMES)}


import sys                                                     # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import conformal_quantile as conformal_q        # noqa: E402
from conformal import aci as aci_shared                        # noqa: E402


def winkler(y, lo, hi, alpha):
    w = hi - lo
    w = w + np.where(y < lo, (2 / alpha) * (lo - y), 0) + np.where(y > hi, (2 / alpha) * (y - hi), 0)
    return float(w.mean())


def evaluate(y, lo, hi, alpha):
    cov = (y >= lo) & (y <= hi)
    return float(cov.mean()), float((hi - lo).mean()), winkler(y, lo, hi, alpha), cov


def aci(res_cal, yhat, y, alpha, scale=None, gamma=GAMMA, window=WINDOW):
    """Immediate-feedback ACI traces (the naive-pipeline defect under study),
    via the shared implementation."""
    lo, hi, _ = aci_shared(res_cal, yhat, y, feedback_mode="immediate",
                           gamma=gamma, window=window, alpha=alpha,
                           scale=scale, return_traces=True)
    return lo, hi


def run_site(site: str) -> None:
    cfg = SITES[site]
    out = OUT_BASE / site
    os.makedirs(out, exist_ok=True)
    feats, targ, dcol = cfg["features"], cfg["target"], cfg["date_col"]

    # ---------------- preprocessing (identical policy to main site) ----------
    df = pd.read_csv(cfg["csv"])
    df.columns = df.columns.str.strip()
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=[dcol]).sort_values(dcol).drop_duplicates(subset=[dcol], keep="last")
    df = df.set_index(dcol)[feats].astype(float)
    for c, (lo, hi) in cfg["bounds"].items():
        df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
    res = df.resample(cfg["resample"]).median().interpolate(method="time").ffill().bfill()
    for c, (lo, hi) in cfg["bounds"].items():
        res[c] = res[c].clip(lo, hi)
    res = res.reset_index()

    sph = cfg["steps_per_hour"]
    steps = {h: int(h[:-1]) * sph for h in HORIZON_NAMES}
    for h, st in steps.items():
        res[f"target_{h}"] = res[targ].shift(-st)
    res = res.iloc[: -max(steps.values())].copy()
    tcols = [f"target_{h}" for h in HORIZON_NAMES]

    n = len(res)
    tr, va = int(n * 0.6), int(n * 0.6) + int(n * 0.2)
    blocks_raw = {"train": res.iloc[:tr], "val": res.iloc[tr:va], "test": res.iloc[va:]}

    fsc, tsc = MinMaxScaler(), MinMaxScaler()
    fsc.fit(blocks_raw["train"][feats])
    tsc.fit(blocks_raw["train"][tcols])

    def windows(d):
        # origin-aligned targets: the shifted target columns are read at the
        # final observed input row, so a nominal h-hour horizon is exactly h
        # hours after the forecast origin (same convention as build_horizon)
        F = fsc.transform(d[feats]).astype(np.float32)
        T = tsc.transform(d[tcols]).astype(np.float32)
        m = len(d) - SEQ_LEN + 1
        X = np.stack([F[i : i + SEQ_LEN] for i in range(m)])
        y = T[SEQ_LEN - 1 : SEQ_LEN - 1 + m]
        return X, y, d[dcol].to_numpy()[SEQ_LEN - 1 : SEQ_LEN - 1 + m]

    X_tr, y_tr, _ = windows(blocks_raw["train"])
    X_va, y_va, _ = windows(blocks_raw["val"])
    X_te, y_te, dates_te = windows(blocks_raw["test"])
    np.save(out / "dates_test.npy", dates_te)
    np.save(out / "X_train.npy", X_tr)
    np.save(out / "X_val.npy", X_va)
    np.save(out / "X_test.npy", X_te)
    np.save(out / "y_train.npy", y_tr)
    np.save(out / "y_val.npy", y_va)
    np.save(out / "y_test.npy", y_te)
    with open(out / "target_scaler.pkl", "wb") as f:
        pickle.dump(tsc, f)

    inv = tsc.inverse_transform
    yv, yt = inv(y_va), inv(y_te)
    SCALE = tsc.data_max_ - tsc.data_min_

    meta = {
        "site": site, "rows_raw": int(len(df)), "resample": cfg["resample"],
        "time_range": (str(res[dcol].min()), str(res[dcol].max())),
        "resampled_timestamps": n, "features": feats, "target": targ,
        "train/val/test": [len(X_tr), len(X_va), len(X_te)], "seeds": SEEDS,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(site, "meta:", meta, flush=True)

    # ---------------- baselines ----------------
    ti = feats.index(targ)
    pers_va = np.repeat(X_va[:, -1, ti : ti + 1], 4, axis=1)
    pers_te = np.repeat(X_te[:, -1, ti : ti + 1], 4, axis=1)
    # persistence features are scaled by fsc; map to target scale via raw value
    # NOTE: feature ti and targets share raw units, but different scalers ->
    # invert through the feature scaler for correctness
    def unscale_pers(P):
        raw = P[:, 0] * (fsc.data_max_[ti] - fsc.data_min_[ti]) + fsc.data_min_[ti]
        return np.repeat(raw[:, None], 4, axis=1)

    pers_va_o, pers_te_o = unscale_pers(pers_va), unscale_pers(pers_te)

    ridge = Ridge(alpha=1.0).fit(X_tr.reshape(len(X_tr), -1), y_tr)
    ridge_va_o = inv(ridge.predict(X_va.reshape(len(X_va), -1)).astype(np.float32))
    ridge_te_o = inv(ridge.predict(X_te.reshape(len(X_te), -1)).astype(np.float32))
    np.save(out / "pred_persistence_val.npy", pers_va_o)
    np.save(out / "pred_persistence_test.npy", pers_te_o)
    np.save(out / "pred_ridge_val.npy", ridge_va_o)
    np.save(out / "pred_ridge_test.npy", ridge_te_o)

    results = {"meta": meta,
               "point": {"persistence": metrics_ph(yt, pers_te_o),
                         "ridge": metrics_ph(yt, ridge_te_o)}}

    # ---------------- multi-seed LSTMs + UQ (20 seeds) ----------------
    rows = []
    alpha = 1 - LEVEL

    def record(method, seed, h, y, lo, hi):
        p, w, wk, _ = evaluate(y, lo, hi, alpha)
        rows.append((method, seed, h, p, w, wk))

    # seed-independent: persistence SCP/ACI
    for h_i, h in enumerate(HORIZON_NAMES):
        res_val = np.abs(yv[:, h_i] - pers_va_o[:, h_i])
        q = conformal_q(res_val, alpha)
        record("scp_persistence", None, h, yt[:, h_i],
               pers_te_o[:, h_i] - q, pers_te_o[:, h_i] + q)
        lo, hi = aci(res_val, pers_te_o[:, h_i], yt[:, h_i], alpha)
        record("aci_persistence", None, h, yt[:, h_i], lo, hi)

    point_det, point_bay = [], []
    for seed in SEEDS:
        preds = {}
        for variant, dr in [("det", 0.2), ("bay", 0.15)]:
            tf.keras.backend.clear_session()
            np.random.seed(seed)
            tf.keras.utils.set_random_seed(seed)
            m = build_lstm(len(feats), dr)
            m.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=80, batch_size=128,
                  verbose=0, callbacks=[
                      EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                      ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)])
            preds[f"{variant}_va"] = inv(m.predict(X_va, verbose=0))
            preds[f"{variant}_te"] = inv(m.predict(X_te, verbose=0))
            if variant == "bay":
                s_va = np.stack([m(X_va, training=True).numpy() for _ in range(MC_SAMPLES)])
                s_te = np.stack([m(X_te, training=True).numpy() for _ in range(MC_SAMPLES)])
                preds["mu_va"], preds["sd_va"] = inv(s_va.mean(0)), s_va.std(0) * SCALE
                preds["mu_te"], preds["sd_te"] = inv(s_te.mean(0)), s_te.std(0) * SCALE

        sd = out / f"seed_{seed}"
        os.makedirs(sd, exist_ok=True)
        for k in ["det_va", "det_te", "bay_va", "bay_te", "mu_va", "mu_te", "sd_va", "sd_te"]:
            np.save(sd / f"{k}.npy", preds[k])

        point_det.append(metrics_ph(yt, preds["det_te"]))
        point_bay.append(metrics_ph(yt, preds["bay_te"]))

        for h_i, h in enumerate(HORIZON_NAMES):
            y1 = yt[:, h_i]
            mu_t = preds["mu_te"][:, h_i]
            sd_t = np.maximum(preds["sd_te"][:, h_i], 1e-6)
            mu_v = preds["mu_va"][:, h_i]
            sd_v = np.maximum(preds["sd_va"][:, h_i], 1e-6)

            record("mc_gauss", seed, h, y1, mu_t - Z90 * sd_t, mu_t + Z90 * sd_t)

            res_val = np.abs(yv[:, h_i] - preds["det_va"][:, h_i])
            q = conformal_q(res_val, alpha)
            record("scp_lstm", seed, h, y1,
                   preds["det_te"][:, h_i] - q, preds["det_te"][:, h_i] + q)

            sc_val = np.abs(yv[:, h_i] - mu_v) / sd_v
            q = conformal_q(sc_val, alpha)
            record("scp_mc_sigma", seed, h, y1, mu_t - q * sd_t, mu_t + q * sd_t)

            lo, hi = aci(res_val, preds["det_te"][:, h_i], y1, alpha)
            record("aci_lstm", seed, h, y1, lo, hi)

            lo, hi = aci(sc_val, mu_t, y1, alpha, scale=sd_t)
            record("aci_mc_sigma", seed, h, y1, lo, hi)
        print(f"{site} seed {seed} done", flush=True)

    def agg_seed(entries):
        return {h: {m: {"mean": float(np.mean([e[h][m] for e in entries])),
                        "std": float(np.std([e[h][m] for e in entries], ddof=1))}
                    for m in ["rmse", "mae", "r2"]} for h in HORIZON_NAMES}

    results["point"]["lstm"] = agg_seed(point_det)
    results["point"]["mc_lstm"] = agg_seed(point_bay)

    with open(out / "uq_results.csv", "w") as f:
        f.write("method,seed,horizon,picp,mpiw,winkler\n")
        for m, s, h, p, w, wk in rows:
            f.write(f"{m},{'' if s is None else s},{h},{p:.4f},{w:.3f},{wk:.3f}\n")

    aggu = {}
    for m, s, h, p, w, wk in rows:
        aggu.setdefault((m, h), []).append((p, w, wk))
    results["uq_90"] = {f"{m}|{h}": {"picp_mean": float(np.mean([v[0] for v in vals])),
                                     "picp_std": float(np.std([v[0] for v in vals], ddof=1)),
                                     "mpiw_mean": float(np.mean([v[1] for v in vals])),
                                     "winkler_mean": float(np.mean([v[2] for v in vals]))}
                        for (m, h), vals in aggu.items()}
    with open(out / "site_results.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [f"=== {site} — nominal 90% (means across seeds) ===",
             f"{'method':16s} " + " ".join(f"{h:>14s}" for h in HORIZON_NAMES)]
    for m in ["mc_gauss", "scp_lstm", "scp_persistence", "scp_mc_sigma",
              "aci_lstm", "aci_persistence", "aci_mc_sigma"]:
        cells = []
        for h in HORIZON_NAMES:
            v = aggu.get((m, h))
            a = np.array(v)
            cells.append(f"{a[:,0].mean()*100:5.1f}%/{a[:,1].mean():6.2f}")
        lines.append(f"{m:16s} " + " ".join(f"{c:>14s}" for c in cells))
    summary = "\n".join(lines)
    with open(out / "uq_summary.txt", "w") as f:
        f.write(summary + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT_BASE, exist_ok=True)
    for site in SITES:
        run_site(site)
    print("multisite validation complete", flush=True)
