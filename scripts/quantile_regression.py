"""Conformalised quantile regression on the naive-pipeline predictions.

Scientific purpose
------------------
Adds a quantile-based uncertainty family to the comparison, so the naive
pipeline's calibration picture is not drawn only from residual-based conformal
methods and MC-dropout.

A quantile-output LSTM shares the backbone of train_deep_models.py
(LSTM 64 -> dropout 0.2 -> LSTM 32 -> dropout 0.2) and predicts 4 horizons x 4
quantiles {0.025, 0.05, 0.95, 0.975} under the pinball loss, with the same seed
list and training settings.

Evaluated three ways per horizon and nominal level:
  qr_raw    raw quantile-regression intervals, uncalibrated
  cqr       split-conformalised on validation, score = max(lo - y, y - hi)
  aci_cqr   adaptive conformal inference over the CQR score stream

Input       results/naive_pipeline/
Output      results/naive_pipeline/uq_extensions/
Manuscript  the quantile-family rows of the calibration tables
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal import conformal_quantile as conformal_q        # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "naive_pipeline"
OUT = RESULTS / "uq_extensions"
os.makedirs(OUT, exist_ok=True)

HORIZONS = ["6h", "12h", "24h", "48h"]
QUANTILES = [0.025, 0.05, 0.95, 0.975]           # (lo95, lo90, hi90, hi95)
LEVELS = {0.90: (1, 2), 0.95: (0, 3)}            # level -> (lo idx, hi idx)
GAMMA, WINDOW = 0.02, 500

with open(RESULTS / "meta.json") as f:
    META = json.load(f)
SEEDS = META["seeds"]

with open(RESULTS / "target_scaler.pkl", "rb") as f:
    tsc = pickle.load(f)


def inv_cols(A):
    """Inverse-transform an (n, 4, nq) quantile array horizon-wise."""
    out = np.empty_like(A, dtype=np.float64)
    for qi in range(A.shape[2]):
        out[:, :, qi] = tsc.inverse_transform(A[:, :, qi])
    return out


X_tr = np.load(RESULTS / "X_train.npy")
X_va = np.load(RESULTS / "X_val.npy")
X_te = np.load(RESULTS / "X_test.npy")
y_tr = np.load(RESULTS / "y_train.npy")
y_va_s = np.load(RESULTS / "y_val.npy")
y_va = tsc.inverse_transform(y_va_s)
y_te = tsc.inverse_transform(np.load(RESULTS / "y_test.npy"))

NQ, NH = len(QUANTILES), len(HORIZONS)


def build_qlstm(seed):
    # TensorFlow is imported here, not at module level, so that a run that only
    # re-evaluates already-saved quantile predictions needs no TensorFlow.
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam

    QVEC = tf.constant(np.array(QUANTILES, dtype=np.float32))

    def pinball_loss(y_true, y_pred):
        y_pred = tf.reshape(y_pred, (-1, NH, NQ))
        y_true = tf.expand_dims(y_true, -1)                # (b, 4, 1)
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(QVEC * e, (QVEC - 1.0) * e))

    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    inp = Input(shape=X_tr.shape[1:])
    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    out = Dense(NH * NQ, activation="linear")(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(1e-3), loss=pinball_loss)
    return m


def winkler(y, lo, hi, alpha):
    w = hi - lo
    w = w + np.where(y < lo, (2 / alpha) * (lo - y), 0) + np.where(y > hi, (2 / alpha) * (y - hi), 0)
    return float(w.mean())


rows = []


def record(method, seed, level, h, y, lo, hi):
    alpha = 1 - level
    cov = float(((y >= lo) & (y <= hi)).mean())
    rows.append((method, seed, level, h, cov, float((hi - lo).mean()), winkler(y, lo, hi, alpha)))


for seed in SEEDS:
    qfile_va = RESULTS / f"seed_{seed}/cqr_q_val.npy"
    qfile_te = RESULTS / f"seed_{seed}/cqr_q_test.npy"
    if qfile_va.exists() and qfile_te.exists():
        Q_va, Q_te = np.load(qfile_va), np.load(qfile_te)
    else:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        m = build_qlstm(seed)
        m.fit(X_tr, y_tr, validation_data=(X_va, y_va_s), epochs=80, batch_size=128,
              verbose=0, callbacks=[
                  EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                  ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)])
        Q_va = m.predict(X_va, verbose=0).reshape(-1, NH, NQ)
        Q_te = m.predict(X_te, verbose=0).reshape(-1, NH, NQ)
        np.save(qfile_va, Q_va)
        np.save(qfile_te, Q_te)
    Qv, Qt = inv_cols(Q_va), inv_cols(Q_te)
    # enforce monotone quantiles
    Qv, Qt = np.sort(Qv, axis=2), np.sort(Qt, axis=2)

    for level, (lo_i, hi_i) in LEVELS.items():
        alpha = 1 - level
        for h_i, h in enumerate(HORIZONS):
            yv1, yt1 = y_va[:, h_i], y_te[:, h_i]
            lov, hiv = Qv[:, h_i, lo_i], Qv[:, h_i, hi_i]
            lot, hit = Qt[:, h_i, lo_i], Qt[:, h_i, hi_i]

            record("qr_raw", seed, level, h, yt1, lot, hit)

            # CQR: symmetric additive correction from validation scores
            sc_val = np.maximum(lov - yv1, yv1 - hiv)
            q = conformal_q(sc_val, alpha)
            record("cqr", seed, level, h, yt1, lot - q, hit + q)

            # ACI over the CQR score stream
            pool = list(sc_val[-WINDOW:])
            a_t = alpha
            lo_arr = np.zeros(len(yt1)); hi_arr = np.zeros(len(yt1))
            for t in range(len(yt1)):
                eff = min(max(1 - a_t, 1e-3), 1 - 1e-3)
                qq = float(np.quantile(pool, eff))
                lo_arr[t], hi_arr[t] = lot[t] - qq, hit[t] + qq
                err = float(not (lo_arr[t] <= yt1[t] <= hi_arr[t]))
                a_t += GAMMA * (alpha - err)
                pool.append(max(lot[t] - yt1[t], yt1[t] - hit[t]))
                if len(pool) > WINDOW:
                    pool.pop(0)
            record("aci_cqr", seed, level, h, yt1, lo_arr, hi_arr)
    print(f"seed {seed} done", flush=True)

with open(OUT / "cqr_results.csv", "w") as f:
    f.write("method,seed,level,horizon,picp,mpiw,winkler\n")
    for m, s, lv, h, p, w, wk in rows:
        f.write(f"{m},{s},{lv},{h},{p:.4f},{w:.3f},{wk:.3f}\n")

# seed-aggregated summary at 90%
agg = {}
for m, s, lv, h, p, w, wk in rows:
    agg.setdefault((m, lv, h), []).append((p, w, wk))
print("\n=== CQR family, nominal 90% (mean PICP% / MPIW across seeds) ===")
for m in ["qr_raw", "cqr", "aci_cqr"]:
    cells = []
    for h in HORIZONS:
        a = np.array(agg[(m, 0.90, h)])
        cells.append(f"{a[:,0].mean()*100:5.1f}%/{a[:,1].mean():6.2f}")
    print(f"{m:8s} " + "  ".join(cells))
