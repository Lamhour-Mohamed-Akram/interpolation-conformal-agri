"""Multi-seed deep forecaster training on the Morocco record (naive pipeline).

Scientific purpose
------------------
Trains the deep models whose uncertainty estimates are studied in the
naive-pipeline case study: a deterministic LSTM and an MC-dropout LSTM, each
over 20 independent random seeds, jointly predicting the 6/12/24/48 h horizons.
A ridge regressor on the flattened input window and a persistence forecaster are
trained alongside as reference points that require no deep learning.

This script deliberately follows the NAIVE preprocessing pipeline (uncapped
interpolation onto the 5-min grid, targets shifted before the chronological
split), because the purpose of the case study is to characterise what such a
pipeline reports. The leakage-free protocol is implemented separately in
preprocess.py and clean_protocol.py.

Twenty seeds are used so that seed-to-seed spread is reported rather than a
single lucky run; the exact seed list is written to
results/deep_20seed/seed_list.json.

Architecture   LSTM(64) -> dropout(0.2) -> LSTM(32) -> dropout(0.2) -> Dense(4)
Training       Adam 1e-3, batch 128, <=80 epochs, early stopping patience 15
MC-dropout     dropout left active at inference, 50 stochastic passes

Validation-set MC-dropout means and standard deviations are saved as well as
test-set ones, because the sigma-normalised conformal methods must be
calibrated on validation predictions.

Persistence units
-----------------
The persistence forecast is the last soil-moisture value of the input window.
That value lives in FEATURE-scaled space, and the feature scaler and the
per-horizon target scaler have different ranges, so it must be mapped back to
raw physical units through the FEATURE scaler. pred_persistence_val/test.npy
are therefore stored in RAW % units, unlike the model predictions, which are
target-scaled; every consumer treats them accordingly. A regression check
asserts that the stored persistence forecast equals the last raw observed
soil-moisture value of each input window.

Run
---
python scripts/train_deep_models.py                       full 20-seed training
python scripts/train_deep_models.py --recompute-outputs   redo preprocessing,
    baselines and every aggregate from the saved per-seed prediction files,
    without retraining (used after a fix that does not touch the networks)

Input       data/data.csv.gz or data/data.csv (either form is accepted)
Output      results/naive_pipeline/ (per-seed predictions, metrics, histories,
            scalers, target timestamps)
Manuscript  the naive-pipeline point-accuracy tables and every uncertainty
            result derived from these predictions
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------- config ----
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import data_file                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = data_file("data.csv")          # accepts data.csv or data.csv.gz
RESULTS = ROOT / "results" / "naive_pipeline"

# The 20 fixed seeds of the multi-seed protocol. Fixed and published so that
# every reported mean and spread can be reproduced exactly; 7 appears once.
SEEDS = [42, 123, 2024, 7, 99,
         0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
         11, 12, 13, 14, 15]
MC_SAMPLES = 50
SEQ_LEN = 24  # 24 x 5 min = 2 h input window
HORIZONS_H = (6, 12, 24, 48)
HORIZON_NAMES = [f"{h}h" for h in HORIZONS_H]
TRAIN_RATIO, VAL_RATIO = 0.6, 0.2
DROPOUT_BASELINE = 0.2
DROPOUT_BAYES = 0.15

FEATURE_COLS = ["humidity", "temperature", "humiditysol", "temperaturesol", "co2", "lumière"]
TARGET_COL = "humiditysol"
DATE_COL = "date"
PHYSICAL_BOUNDS = {
    "humidity": (0.0, 100.0),
    "humiditysol": (0.0, 100.0),
    "temperature": (-10.0, 60.0),
    "temperaturesol": (-10.0, 60.0),
    "co2": (0.0, 5000.0),
    "lumière": (0.0, 200000.0),
}

# --------------------------------------------------------- preprocessing ----


def load_and_resample() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, encoding="latin1")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
    df = df.drop_duplicates(subset=[DATE_COL], keep="last").set_index(DATE_COL)
    df = df[FEATURE_COLS].copy()
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    df_res = df.resample("5min").median()
    df_res = df_res.interpolate(method="time").ffill().bfill()
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        df_res[col] = df_res[col].clip(lo, hi)
    return df_res.reset_index()


def create_sequences(data: pd.DataFrame, target_cols: list[str]):
    """Windows plus multi-horizon targets, aligned to the forecast origin.

    The forecast origin is the FINAL observed row of the input window
    (row i + SEQ_LEN - 1). The shifted target columns were built with
    shift(-steps), so reading them AT the origin row yields the raw value
    exactly ``steps`` grid bins after the origin: a nominal h-hour horizon is
    exactly h hours after the last observed input, on every grid. The naive
    pipeline keeps its studied defects (uncapped interpolation, targets
    shifted before the split, immediate feedback) but its nominal horizon is
    exact. The returned dates are the origin timestamps.
    """
    feat = data[FEATURE_COLS].to_numpy(dtype=np.float32)
    targ = data[target_cols].to_numpy(dtype=np.float32)
    dates = data[DATE_COL].to_numpy()
    n = len(data) - SEQ_LEN + 1
    X = np.zeros((n, SEQ_LEN, len(FEATURE_COLS)), dtype=np.float32)
    y = np.zeros((n, len(target_cols)), dtype=np.float32)
    for i in range(n):
        X[i] = feat[i : i + SEQ_LEN]
        y[i] = targ[i + SEQ_LEN - 1]
    return X, y, dates[SEQ_LEN - 1 : SEQ_LEN - 1 + n]


def mc_predict(model, X: np.ndarray, T: int) -> np.ndarray:
    return np.stack([model(X, training=True).numpy() for _ in range(T)], axis=0)


def metrics_per_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for i, h in enumerate(HORIZON_NAMES):
        out[h] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
        }
    return out


def prepare_data():
    """Naive-pipeline preprocessing, splits, scaling and baseline forecasts.

    Saves every array, both scalers, the baseline predictions and meta.json,
    and returns what the training loop needs. Deterministic: no random state.
    """
    steps_by_horizon = {f"{h}h": h * 12 for h in HORIZONS_H}
    target_cols = [f"target_{k}" for k in steps_by_horizon]

    df_res = load_and_resample()
    df_h = df_res.copy()
    for name, steps in steps_by_horizon.items():
        df_h[f"target_{name}"] = df_h[TARGET_COL].shift(-steps)
    df_h = df_h.iloc[: -max(steps_by_horizon.values())].copy()

    n = len(df_h)
    tr_end = int(n * TRAIN_RATIO)
    va_end = tr_end + int(n * VAL_RATIO)
    df_tr_raw, df_va_raw, df_te_raw = df_h.iloc[:tr_end], df_h.iloc[tr_end:va_end], df_h.iloc[va_end:]

    fsc, tsc = MinMaxScaler(), MinMaxScaler()
    fsc.fit(df_tr_raw[FEATURE_COLS])
    tsc.fit(df_tr_raw[target_cols])

    parts = {}
    raw_targets = {}
    for split, d in [("train", df_tr_raw), ("val", df_va_raw), ("test", df_te_raw)]:
        dd = d.copy()
        dd[FEATURE_COLS] = fsc.transform(d[FEATURE_COLS])
        dd[target_cols] = tsc.transform(d[target_cols])
        parts[split] = create_sequences(dd, target_cols)
        raw_targets[split] = d[TARGET_COL].to_numpy()

    (X_tr, y_tr, dates_tr), (X_va, y_va, dates_va), (X_te, y_te, dates_te) = (
        parts["train"], parts["val"], parts["test"],
    )

    for name, arr in [
        ("X_train", X_tr), ("y_train", y_tr), ("X_val", X_va), ("y_val", y_va),
        ("X_test", X_te), ("y_test", y_te),
        ("dates_train", dates_tr), ("dates_val", dates_va), ("dates_test", dates_te),
    ]:
        np.save(RESULTS / f"{name}.npy", arr)
    with open(RESULTS / "feature_scaler.pkl", "wb") as f:
        pickle.dump(fsc, f)
    with open(RESULTS / "target_scaler.pkl", "wb") as f:
        pickle.dump(tsc, f)

    meta = {
        "raw_time_range": (str(df_res[DATE_COL].min()), str(df_res[DATE_COL].max())),
        "resampled_timestamps": int(len(df_res)),
        "sequence_length": SEQ_LEN,
        "time_horizons_steps": steps_by_horizon,
        "seeds": SEEDS,
        "mc_samples": MC_SAMPLES,
        "train_samples": int(len(X_tr)),
        "val_samples": int(len(X_va)),
        "test_samples": int(len(X_te)),
        "split_policy": "time_split_before_scaling_and_windowing",
        "persistence_units": "raw_percent",
    }
    with open(RESULTS / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    inv = tsc.inverse_transform
    y_te_orig = inv(y_te)

    # ---------- seed-independent baselines ----------
    hs_idx = FEATURE_COLS.index(TARGET_COL)

    def persistence_raw(X):
        # X is feature-scaled; the last soil-moisture value must be mapped back
        # to raw units through the FEATURE scaler — the target scaler was
        # fitted on the shifted targets and has different per-horizon ranges.
        last = X[:, -1, hs_idx].astype(np.float64)
        raw = last * (fsc.data_max_[hs_idx] - fsc.data_min_[hs_idx]) + fsc.data_min_[hs_idx]
        return np.repeat(raw[:, None], len(HORIZONS_H), axis=1)

    pers_va, pers_te = persistence_raw(X_va), persistence_raw(X_te)

    # regression check: the raw persistence forecast must equal the last raw
    # observed soil-moisture value of each input window (float32 tolerance)
    for pers, raw in [(pers_va, raw_targets["val"]), (pers_te, raw_targets["test"])]:
        expect = raw[SEQ_LEN - 1 : SEQ_LEN - 1 + len(pers)]
        assert np.allclose(pers[:, 0], expect, atol=1e-3), \
            "persistence forecast does not equal the last raw observed value"

    np.save(RESULTS / "pred_persistence_val.npy", pers_va)
    np.save(RESULTS / "pred_persistence_test.npy", pers_te)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr.reshape(len(X_tr), -1), y_tr)
    ridge_va = ridge.predict(X_va.reshape(len(X_va), -1)).astype(np.float32)
    ridge_te = ridge.predict(X_te.reshape(len(X_te), -1)).astype(np.float32)
    np.save(RESULTS / "pred_ridge_val.npy", ridge_va)
    np.save(RESULTS / "pred_ridge_test.npy", ridge_te)

    baseline_metrics = {
        "persistence": metrics_per_horizon(y_te_orig, pers_te),
        "ridge": metrics_per_horizon(y_te_orig, inv(ridge_te)),
    }
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), y_te_orig, inv, baseline_metrics


def aggregate_and_write(all_metrics: dict) -> None:
    agg = {}
    for variant in ["deterministic", "bayesian"]:
        agg[variant] = {}
        for h in HORIZON_NAMES:
            for m in ["rmse", "mae", "r2"]:
                vals = [all_metrics["per_seed"][s][variant][h][m] for s in SEEDS]
                agg[variant].setdefault(h, {})[m] = {
                    "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                    "min": float(np.min(vals)), "max": float(np.max(vals)),
                }
    all_metrics["aggregate"] = agg
    with open(RESULTS / "point_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)


def recompute_outputs() -> None:
    """Recompute baselines and every aggregate from the saved seed predictions.

    Used after a fix that changes derived quantities (e.g. persistence units)
    but not the trained networks: preprocessing, scalers, baselines and
    point_metrics.json are rebuilt; the per-seed prediction files must already
    exist and are NOT retrained.
    """
    _, _, _, y_te_orig, inv, baseline_metrics = prepare_data()
    all_metrics: dict = {"baselines": baseline_metrics, "per_seed": {}}
    for seed in SEEDS:
        sdir = RESULTS / f"seed_{seed}"
        seed_entry = {}
        for variant in ("deterministic", "bayesian"):
            p = sdir / f"pred_{variant}_test.npy"
            assert p.exists(), f"missing {p}; run the full training first"
            seed_entry[variant] = metrics_per_horizon(y_te_orig, inv(np.load(p)))
        all_metrics["per_seed"][seed] = seed_entry
    aggregate_and_write(all_metrics)
    print("recomputed baselines, per-seed metrics and aggregates "
          "from saved predictions", flush=True)


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    if "--recompute-outputs" in sys.argv[1:]:
        recompute_outputs()
        return

    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    def build_lstm(dropout_rate: float) -> Model:
        inputs = Input(shape=(SEQ_LEN, len(FEATURE_COLS)))
        x = LSTM(64, return_sequences=True)(inputs)
        x = Dropout(dropout_rate)(x)
        x = LSTM(32)(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(len(HORIZONS_H), activation="linear")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
        return model

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), y_te_orig, inv, baseline_metrics = prepare_data()

    # ---------- multi-seed LSTM training ----------
    all_metrics: dict = {"baselines": baseline_metrics, "per_seed": {}}
    callbacks = lambda: [  # noqa: E731
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    for seed in SEEDS:
        sdir = RESULTS / f"seed_{seed}"
        os.makedirs(sdir, exist_ok=True)
        seed_entry = {}

        for variant, dr in [("deterministic", DROPOUT_BASELINE), ("bayesian", DROPOUT_BAYES)]:
            tf.keras.backend.clear_session()
            np.random.seed(seed)
            tf.keras.utils.set_random_seed(seed)
            model = build_lstm(dr)
            hist = model.fit(
                X_tr, y_tr, validation_data=(X_va, y_va),
                epochs=80, batch_size=128, verbose=0, callbacks=callbacks(),
            )
            with open(sdir / f"history_{variant}.json", "w") as f:
                json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)

            pred_va = model.predict(X_va, verbose=0)
            pred_te = model.predict(X_te, verbose=0)
            np.save(sdir / f"pred_{variant}_val.npy", pred_va)
            np.save(sdir / f"pred_{variant}_test.npy", pred_te)
            seed_entry[variant] = metrics_per_horizon(y_te_orig, inv(pred_te))

            if variant == "bayesian":
                for split, X in [("val", X_va), ("test", X_te)]:
                    samples = mc_predict(model, X, MC_SAMPLES)
                    np.save(sdir / f"mc_mean_{split}.npy", samples.mean(axis=0))
                    np.save(sdir / f"mc_std_{split}.npy", samples.std(axis=0))
                    # 90/95% central quantiles for quantile-based intervals
                    for lvl, (qlo, qhi) in {"90": (0.05, 0.95), "95": (0.025, 0.975)}.items():
                        np.save(sdir / f"mc_q{lvl}lo_{split}.npy", np.quantile(samples, qlo, axis=0))
                        np.save(sdir / f"mc_q{lvl}hi_{split}.npy", np.quantile(samples, qhi, axis=0))

        all_metrics["per_seed"][seed] = seed_entry
        print(f"seed {seed} done: det 6h RMSE "
              f"{seed_entry['deterministic']['6h']['rmse']:.2f}, "
              f"bayes 6h RMSE {seed_entry['bayesian']['6h']['rmse']:.2f}", flush=True)

    aggregate_and_write(all_metrics)
    print("deep-model training complete", flush=True)


if __name__ == "__main__":
    main()
