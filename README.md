# Interpolation Confounds Conformal Calibration in Agricultural Sensor Forecasting

Code and data for *"Interpolation Confounds Conformal Calibration in
Agricultural Sensor Forecasting: A Leakage-Free Multi-Horizon Evaluation
Protocol"*.

Sparse sensor records are routinely resampled onto a regular grid and
interpolated without a cap before uncertainty methods are evaluated on them. In
a controlled experiment on a gap-free record, holding the test set real and
injecting a known fraction of interpolation into the calibration set alone is
enough to distort split-conformal calibration systematically: the magnitude
scales with the injected fraction, the severity with the gap geometry, and the
direction with the fill method. This repository contains the experiments behind
that result, the two multi-horizon evaluation defects analysed alongside it
(constructing targets before the chronological split, and feeding an online
adaptive method an outcome before it is observable), and the leakage-free
protocol evaluated against them.

The leakage-free protocol is an *evaluation* protocol: it removes the
methodological defects above so that measured coverage is trustworthy. It does
not guarantee nominal empirical coverage — on the primary greenhouse record
the corrected evaluation exposes genuine long-horizon distribution shift
(48-hour persistence-base coverage of 62.5% at nominal 90% for both split
conformal and availability-aware adaptive conformal; see
`results/tables/clean_protocol_morocco.csv`).

## Repository structure

```
scripts/     preprocessing, experiments, table and figure generation
data/        raw records, checksums, provenance (see data/README.md)
results/     one directory per experiment, plus figures/ and tables/
requirements.txt
```

Support modules: `preprocess.py` (Morocco gap-aware preprocessing, imported by
several experiments), `conformal.py` (the split-conformal and adaptive-conformal
primitives, so every experiment shares one implementation) and `sites.py`
(cross-site loading).

## Installation

```bash
git clone https://github.com/Lamhour-Mohamed-Akram/interpolation-conformal-agri
cd interpolation-conformal-agri
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`tensorflow` is needed only for the deep-model case study. The core experiments
run without it.

### Exact reproducibility environment

All published numerical outputs — core experiments, tables, figures, and the
deep-model training runs — were generated with Python 3.12.13 using the single
environment pinned in `requirements-lock.txt`
(`pip install -r requirements-lock.txt`). `requirements.txt` gives permissive
bounds for casual use; note that the gradient-boosting baseline rows
(`scikit-learn`'s `HistGradientBoostingRegressor`) are sensitive to the
scikit-learn version even at a fixed random seed, so bit-exact reproduction of
every stored value requires the locked versions.

## Data setup

Three of the four records ship with the repository. The Morocco record is
distributed gzip-compressed as `data/data.csv.gz` (5.0 MB rather than 61.6 MB);
pandas reads it directly, so nothing needs unpacking and no Git LFS is used.

The Iraq record is **not** redistributed, because its source does not state
terms that clearly permit it. [data/README.md](data/README.md) gives the
download link, the exact filename, where to put it and its SHA-256. Until it is
present the two cross-site scripts print a prominent `NOT REPRODUCED` notice
naming the missing site rather than quietly reporting a partial run; the derived
Iraq results are included under `results/`.

## Reproducing the experiments

```bash
python scripts/preprocess.py             # grid, gap audit, shift diagnostic
python scripts/controlled_injection.py   # calibration-only injection (central result)
python scripts/uscrn_replication.py      # replication on an independent record
python scripts/delayed_aci.py            # immediate vs availability-aware feedback
python scripts/target_isolation.py       # target-boundary leakage
python scripts/boundary_robustness.py    # both defects across sites and splits
python scripts/same_model_shift.py       # naive vs clean, same base model
python scripts/cap_sensitivity.py        # sensitivity to the interpolation cap
python scripts/clean_protocol.py         # cross-site evaluation under the protocol
python scripts/significance_tests.py     # paired tests / CIs for every reported difference
```

Trained-forecaster replication of the injection experiment (ridge and a
five-seed LSTM base on the same gap placements; needs `tensorflow`, ~10 min
on CPU; its outputs are already provided under `results/`):

```bash
python scripts/controlled_injection_deep.py
```

Optional deep-model case study (needs `tensorflow`; its outputs are already
provided under `results/`):

```bash
python scripts/train_deep_models.py                  # 20 seeds, Morocco
#   --recompute-outputs rebuilds baselines and aggregates from saved
#   predictions without retraining (no TensorFlow required)
python scripts/train_deep_models_multisite.py        # 20 seeds, other sites
python scripts/evaluate_uq.py                        # coverage of every UQ method
python scripts/quantile_regression.py                # CQR family
python scripts/ensembles_and_decision.py             # ensembles, EnbPI, cost-loss
python scripts/ensembles_and_decision_multisite.py   #   (+ threshold x penalty sensitivity grid)
python scripts/aggregate_seed_results.py             # per-seed metric tables
```

## Tables and figures

```bash
python scripts/generate_tables.py     # -> results/tables/
python scripts/generate_figures.py    # -> results/figures/
python scripts/test_methodology.py    # synthetic tests of the protocol invariants
python scripts/verify_outputs.py      # structural + methodological checks
```

`verify_outputs.py` also re-derives the protocol's methodological invariants
from the raw records (whole-gap interpolation, chronological target isolation,
controlled-injection test isolation, exact integer injection counts, persistence
units, and the finite-sample conformal quantile) and exits non-zero if any
fails.

Both generators read only the files under `results/`; no result value is written
into the plotting or table code. Figures whose inputs are absent are skipped
with a message rather than silently omitted.

## Runtime

* Core experiments: a few minutes on a laptop CPU. `controlled_injection.py`
  dominates (5,760 cells) at roughly 3–8 minutes; the rest take seconds to a
  couple of minutes. No neural network is trained, and the paper's central
  causal claims come entirely from this set.
* Deep-model case study: 120 LSTM/MC-dropout training runs plus quantile models,
  several hours of CPU. Its derived outputs are shipped, so the tables and
  figures can be regenerated without retraining, and the per-seed point
  predictions of both model variants are shipped so
  `train_deep_models.py --recompute-outputs` can rebuild every aggregate from
  a fresh clone. The large per-seed MC-dropout sample statistics and
  quantile-model arrays are not shipped (regenerable only by retraining).

## Reproducibility notes

The deep case study uses 20 fixed seeds — 42, 123, 2024, 7, 99, 0–6 and 8–15 —
recorded in `results/deep_20seed/seed_list.json`. Every run is reported: no seed
is excluded from any aggregate, and convergence is recorded as a column rather
than used as a filter.

## Citation

M. A. Lamhour, M. Msalek, M. Kasbouya, M. Rachdi, K. Mahjoubi,
S. Ardchir, and M. Azzouazi,
"Interpolation Confounds Conformal Calibration in Agricultural Sensor Forecasting:
A Leakage-Free Multi-Horizon Evaluation Protocol,"
submitted to *IEEE Access*, 2026.

```bibtex
@misc{lamhour2026interpolation,
  title={Interpolation Confounds Conformal Calibration in Agricultural Sensor Forecasting:
         A Leakage-Free Multi-Horizon Evaluation Protocol},
  author={Lamhour, Mohamed Akram and
          Msalek, Mohamed and
          Kasbouya, Mohamed and
          Rachdi, Mohamed and
          Mahjoubi, Khadija and
          Ardchir, Soufiane and
          Azzouazi, Mohamed},
  year={2026},
  note={Submitted to IEEE Access}
}
```

Once the paper is formally published, we will manually replace this with the
final IEEE Access citation and DOI.

The prior deployment whose data and pipeline this study audits is
M. A. Lamhour et al., "Multi-Seed XAI Validation for Reliable LSTM Irrigation
Control in Greenhouse Environments," *IEEE Access*, vol. 14, pp. 120366–120387,
2026, doi:10.1109/ACCESS.2026.3720462.

## License

Code: MIT (`LICENSE`). Datasets keep their own licences, documented in
[data/README.md](data/README.md).
