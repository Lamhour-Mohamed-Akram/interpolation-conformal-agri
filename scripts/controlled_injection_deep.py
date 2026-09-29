"""Controlled calibration-only interpolation injection with trained forecasters.

Scientific purpose
------------------
The central causal experiment (controlled_injection.py) uses a training-free
persistence base so that no fitted model can absorb or mask the effect of
calibration-set interpolation. This companion experiment asks whether the
direction and magnitude of that effect carry over to trained forecasters: a
ridge regressor and a multivariate LSTM of the same architecture as the
naive-pipeline case study (two stacked LSTM layers, 64/32 units, dropout 0.2,
24-step input window, one linear output per horizon).

Design (identical to the persistence experiment wherever it can be)
-------------------------------------------------------------------
substrate   Johannesburg hourly record, every feature complete on the hourly
            grid (asserted), so all interpolation present is injected here
split       train = first 60 % of bins, calibration = next 20 %, test = last 20 %
injection   the SAME masks as controlled_injection.py: placement seed s with
            RandomState(s) draws the identical gap set for a given geometry
            and fraction, so every cell here is paired bin-for-bin with the
            persistence cell; the mask is applied to EVERY feature column
            (a logger outage removes all channels at once)
fill        linear | forward-fill | cubic spline | none (gap-aware exclusion),
            each operating on the pre-test prefix only (injection.fill_prefix)
models      fixed before any injection: each model is fitted ONCE on the real
            training block (targets confined to the training block; the LSTM's
            early stopping uses the last 15 % of training origins, never the
            calibration block), so across cells only the calibration residuals
            change. Five LSTM seeds (the first five of the published seed
            list) quantify initialization variability; ridge is deterministic.
calibration origins in [tr, va - h); inputs are the 24 bins ending at the
            origin, read from the filled prefix. The gap-aware arm applies
            protocol rules P1-P2 exactly as a windowed model must: injected
            runs no longer than the hourly whole-gap cap (2 bins, as for this
            site in sites.py) are interpolated, longer runs stay missing, an
            origin is kept only if its input window contains no long-gap bin,
            and its target bin must be real. Strict exclusion of every window
            touching any injected bin is infeasible for a 24-bin window under
            dense isolated missingness (no window survives at 50 %), which is
            precisely why the protocol caps rather than forbids interpolation.
test        held real and fixed; the same test predictions serve every cell
metric      split-conformal coverage on the real test block at nominal 90 %,
            mean width, |PICP - 0.90|, calibration-residual 90th percentile,
            and the surviving calibration count

Input       data/mendeley_soil_data_incl_rain_v3.csv
Output      results/controlled_injection_deep/injection_matrix_deep.csv
            results/controlled_injection_deep/point_accuracy_deep.csv
            results/controlled_injection_deep/pred_test_<model>.npy
Manuscript  the trained-forecaster replication of the injection experiment
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import scp_quantile, ALPHA                    # noqa: E402
from injection import injected_mask, fill_prefix             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "mendeley_soil_data_incl_rain_v3.csv"
OUT = ROOT / "results" / "controlled_injection_deep"

FEATURES = ["Humidity", "Atmospheric_Temp", "Soil_Temp", "Soil_Moisture", "Dew_Point"]
TARGET = "Soil_Moisture"
TI = FEATURES.index(TARGET)
HORIZONS = (6, 12, 24, 48)
FRACS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
GEOMS = ("single", "distributed", "isolated")
METHODS = ("linear", "ffill", "spline", "none")
REPS = 20
TR, VA = 0.6, 0.8
SEQ_LEN = 24
LSTM_SEEDS = [42, 123, 2024, 7, 99]
ES_HOLDOUT = 0.15
CAP_BINS = 2          # whole-gap cap on the hourly grid (2 h), as in sites.py
MIN_CAL = 20          # a cell with fewer surviving calibration scores is recorded as NaN


def short_gap_mask(inj: np.ndarray, cap: int) -> np.ndarray:
    """True on injected bins that belong to a missing run of length <= cap."""
    out = np.zeros_like(inj)
    idx = np.flatnonzero(inj)
    if len(idx) == 0:
        return out
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    for run in np.split(idx, breaks):
        if len(run) <= cap:
            out[run] = True
    return out


def load_matrix() -> np.ndarray:
    """Hourly feature matrix (L, n_features) on a complete grid."""
    df = pd.read_csv(CSV)
    df.columns = df.columns.str.strip()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = (df.dropna(subset=["Time"]).sort_values("Time")
            .drop_duplicates("Time", keep="last").set_index("Time"))
    res = df[FEATURES].resample("1h").median()
    n_missing = int(res.isna().sum().sum())
    assert n_missing == 0, (
        f"substrate is not gap-free: {n_missing} missing hourly cells. The "
        "controlled injection design assumes a complete record.")
    return res.to_numpy(np.float64)


def windows(F: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Input windows ending at each origin (inclusive)."""
    return np.stack([F[o - SEQ_LEN + 1: o + 1] for o in origins]).astype(np.float32)


def build_lstm(n_features: int, seed: int):
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    inputs = Input(shape=(SEQ_LEN, n_features))
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(len(HORIZONS), activation="linear")(x)
    m = Model(inputs=inputs, outputs=outputs)
    m.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = load_matrix()
    L = len(S)
    tr, va = int(L * TR), int(L * VA)
    hmax = max(HORIZONS)

    # ---- fixed models, fitted once on the real training block ----
    fsc = MinMaxScaler().fit(S[:tr])
    tsc = MinMaxScaler().fit(S[:tr, TI:TI + 1])
    F_real = fsc.transform(S).astype(np.float32)

    def targets(Smat, origins):
        Y = np.stack([Smat[origins + h, TI] for h in HORIZONS], axis=1)
        return tsc.transform(Y.reshape(-1, 1)).reshape(Y.shape).astype(np.float32)

    def inv_t(P):
        return tsc.inverse_transform(P.reshape(-1, 1)).reshape(P.shape)

    train_origins = np.arange(SEQ_LEN - 1, tr - hmax)        # every target inside train
    n_fit = int(len(train_origins) * (1 - ES_HOLDOUT))
    X_fit, y_fit = windows(F_real, train_origins[:n_fit]), targets(S, train_origins[:n_fit])
    X_es, y_es = windows(F_real, train_origins[n_fit:]), targets(S, train_origins[n_fit:])

    models = {}
    ridge = Ridge(alpha=1.0).fit(np.concatenate([X_fit, X_es]).reshape(n_fit + len(X_es), -1),
                                 np.concatenate([y_fit, y_es]))
    models["ridge"] = lambda X, m=ridge: m.predict(X.reshape(len(X), -1))

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    for seed in LSTM_SEEDS:
        m = build_lstm(len(FEATURES), seed)
        m.fit(X_fit, y_fit, validation_data=(X_es, y_es), epochs=80, batch_size=128,
              verbose=0, callbacks=[
                  EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                  ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)])
        models[f"lstm_{seed}"] = lambda X, m=m: m.predict(X, batch_size=2048, verbose=0)
        print(f"trained lstm seed {seed}", flush=True)

    # ---- test predictions: real inputs, fixed for every cell ----
    test_origins = np.arange(va, L - min(HORIZONS))
    X_te = windows(F_real, test_origins)
    pred_te = {name: inv_t(f(X_te)) for name, f in models.items()}
    point_rows = []
    for name in list(models) + ["persistence"]:
        for h_i, h in enumerate(HORIZONS):
            ot = test_origins[test_origins < L - h]
            yt = S[ot + h, TI]
            p = S[ot, TI] if name == "persistence" else pred_te[name][: len(ot), h_i]
            point_rows.append(dict(model=name, h=h, rmse=float(np.sqrt(np.mean((yt - p) ** 2))),
                                   mae=float(np.mean(np.abs(yt - p)))))
        if name != "persistence":
            np.save(OUT / f"pred_test_{name}.npy", pred_te[name])
    pd.DataFrame(point_rows).to_csv(OUT / "point_accuracy_deep.csv", index=False)

    # ---- injection cells ----
    cal_origins = np.arange(tr, va - min(HORIZONS))
    rows = []
    for geom in GEOMS:
        for method in METHODS:
            for frac in FRACS:
                for seed in range(REPS):
                    rng = np.random.RandomState(seed)             # identical mask to the
                    inj = np.zeros(L, bool)                       # persistence experiment
                    inj[tr:va] = injected_mask(va - tr, frac, geom, rng)
                    requested_n = int(frac * (va - tr))
                    realized_n = int(inj.sum())
                    Sf = S.copy()
                    if method == "none":
                        # protocol arm (P1-P2 for a windowed model): gaps no longer
                        # than the hourly cap are interpolated, longer gaps stay
                        # missing, a window is admissible only if it contains no
                        # long-gap bin, and the target bin must be real
                        inj_short = short_gap_mask(inj, CAP_BINS)
                        inj_long = inj & ~inj_short
                        for j in range(S.shape[1]):
                            Sf[:va, j] = fill_prefix(S[:va, j], inj_short[:va], "linear")
                        Sf[inj_long] = np.nan
                        csum = np.concatenate([[0], np.cumsum(inj_long.astype(np.int64))])
                    else:
                        for j in range(S.shape[1]):               # outage hits every channel
                            Sf[:va, j] = fill_prefix(S[:va, j], inj[:va], method)
                        csum = None
                    F_fill = fsc.transform(np.nan_to_num(Sf, nan=0.0)).astype(np.float32)
                    X_cal = windows(F_fill, cal_origins)
                    preds = {name: inv_t(f(X_cal)) for name, f in models.items()}
                    for h_i, h in enumerate(HORIZONS):
                        sel = cal_origins < va - h
                        oc = cal_origins[sel]
                        if method == "none":
                            win_ok = (csum[oc + 1] - csum[oc - SEQ_LEN + 1]) == 0
                            ok = win_ok & ~inj[oc + h]
                        else:
                            ok = np.ones(len(oc), bool)
                        y_cal = Sf[oc[ok] + h, TI]
                        ot = test_origins[test_origins < L - h]
                        yt = S[ot + h, TI]
                        for name in models:
                            rc = np.abs(y_cal - preds[name][sel][ok, h_i])
                            if len(rc) < MIN_CAL:
                                q = float("nan"); picp = float("nan"); p90 = float("nan")
                            else:
                                q = scp_quantile(rc, ALPHA)
                                rt = np.abs(yt - pred_te[name][: len(ot), h_i])
                                picp = float(np.mean(rt <= q))
                                p90 = float(np.percentile(rc, 90))
                            rows.append(dict(geom=geom, method=method, frac=frac, seed=seed,
                                             model=name, requested_frac=frac,
                                             realized_frac=realized_n / (va - tr),
                                             requested_n=requested_n, realized_n=realized_n,
                                             h=h, n_cal=int(ok.sum()), picp=picp, mpiw=2 * q,
                                             calib_err=abs(picp - (1 - ALPHA)),
                                             val_res_p90=p90))
                print(f"{geom:12s} {method:7s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "injection_matrix_deep.csv", index=False)

    df["family"] = np.where(df.model == "ridge", "ridge", "lstm")
    print("\nfill method x fraction (distributed geometry, mean |PICP-0.90| over horizons, "
          "placements and model seeds)")
    for fam in ("ridge", "lstm"):
        print(f"--- {fam}")
        print(f"{'frac':>6} " + " ".join(f"{m:>12}" for m in METHODS))
        for frac in FRACS:
            cells = []
            for m in METHODS:
                g = df[(df.family == fam) & (df.geom == "distributed") & (df.method == m) & (df.frac == frac)]
                cells.append(f"{100*g.calib_err.mean():.1f} pp")
            print(f"{frac:6.2f} " + " ".join(f"{c:>12}" for c in cells))
    print("\ntest RMSE per horizon")
    for _, r in pd.DataFrame(point_rows).iterrows():
        print(f"{r.model:12s} {int(r.h):3d}h  rmse {r.rmse:7.2f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}/injection_matrix_deep.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
