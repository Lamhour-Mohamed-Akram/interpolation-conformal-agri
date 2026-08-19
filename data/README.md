# Data

Four records are used. Three are distributed here; the Iraq record must be
downloaded from its original source because its redistribution terms are not
stated explicitly.

Place every file directly in this directory under the exact filename given
below, then run `python scripts/verify_outputs.py`, which checks each file
against its SHA-256.

| Dataset | Filename in `data/` | Cadence | Target column | Distributed here |
| --- | --- | --- | --- | --- |
| Morocco greenhouse | `data.csv.gz` | irregular, resampled to 5 min | `humiditysol` | yes, gzip-compressed |
| Iraq greenhouse | `iraq_IoTProcessed_Data.csv` | ~5 min | `water_level` | **no — download, see below** |
| Johannesburg field station | `mendeley_soil_data_incl_rain_v3.csv` | hourly | `Soil_Moisture` | yes |
| USCRN Fairhope 3 NE | `uscrn_fairhope_AL_2021.txt` | hourly | column 28 (5 cm soil moisture) | yes |

## The Morocco record is gzip-compressed

`data.csv` is 61.6 MB uncompressed, so it is distributed as `data.csv.gz`
(5.0 MB, a 91.9% reduction). No Git LFS is used.

Nothing needs to be decompressed: pandas reads the compressed file directly, and
the scripts accept either form, so the pipeline is unchanged. To obtain the
plain file anyway:

```bash
gunzip -k data/data.csv.gz      # -k keeps the .gz alongside it
```

Both checksums are recorded, so the decompressed content can be confirmed to be
byte-identical to the file the experiments were run on:

```text
769c74eff348150fa83de0d78ef81c808ac93db53291439be94bd9fc60c1cda9  data.csv.gz   (as distributed)
9d5e88addb3721903c30835dfab3bea4bf41fe9c427d90795b36ad5c290298e7  data.csv      (decompressed, as used)
```

`verify_outputs.py` verifies the second hash by streaming the decompressed
contents, so it is checked even if you never unpack the file.

## Obtaining the Iraq record

It is **not redistributed here** because the source does not state terms that
clearly permit redistribution. To reproduce the Iraq rows yourself:

1. Download *IoT Agriculture 2024* (W. D. Abdullah, Tikrit University) from
   <https://www.kaggle.com/datasets/wisam1985/iot-agriculture-2024>
   (a free Kaggle account is required).
2. Take the processed file from that dataset and name it exactly
   `iraq_IoTProcessed_Data.csv`.
3. Place it in this directory, i.e. at `data/iraq_IoTProcessed_Data.csv`.
4. Confirm it is the same file used in the paper:

   ```bash
   shasum -a 256 data/iraq_IoTProcessed_Data.csv
   # expected:
   # b6485b4322662ec6beeb986bf4adf2dfccfd3c3cf2e0bbe514439d3b517e8a82
   ```

5. Re-run any cross-site script (`clean_protocol.py`, `boundary_robustness.py`).

Please observe the source's own terms. Until the file is present, the
cross-site scripts print a prominent `NOT REPRODUCED` notice naming the missing
site and repeating these instructions; they do not report a partial run as a
complete one. **The derived Iraq results are included** under
`results/multisite/iraq/` and `results/deep_20seed/iraq_lstm_20seed.csv`, so the
paper's Iraq numbers remain inspectable without the download.

## Sources and redistribution

**Morocco greenhouse** — operational deployment of the authors' own prior work
(Lamhour et al., *IEEE Access*, vol. 14, pp. 120366–120387, 2026,
doi:10.1109/ACCESS.2026.3720462). Redistributed here by its authors.

**Iraq greenhouse** — *IoT Agriculture 2024*, W. D. Abdullah, Tikrit University,
via Kaggle. Redistribution terms not explicitly stated; not redistributed.

**Johannesburg field station** — Mendeley Data, doi:10.17632/m6j79zjyd7.1
(K. Adetunji), CC BY 4.0, which permits redistribution with attribution.

**USCRN Fairhope 3 NE** — NOAA NCEI U.S. Climate Reference Network hourly02
product, doi:10.7289/V5H13007. A work of the U.S. Government, public domain.

Cite the original source of any dataset you reuse.

## Which experiments need which file

| File | Used by |
| --- | --- |
| `data.csv.gz` | `preprocess.py`, `delayed_aci.py`, `target_isolation.py`, `same_model_shift.py`, `cap_sensitivity.py`, `clean_protocol.py`, `boundary_robustness.py`, `train_deep_models.py` |
| `mendeley_soil_data_incl_rain_v3.csv` | `controlled_injection.py` (the central causal experiment), `clean_protocol.py`, `boundary_robustness.py`, `train_deep_models_multisite.py` |
| `uscrn_fairhope_AL_2021.txt` | `uscrn_replication.py` |
| `iraq_IoTProcessed_Data.csv` | `clean_protocol.py`, `boundary_robustness.py`, `train_deep_models_multisite.py` |

The controlled injection experiment, the USCRN replication and every
Morocco-based experiment therefore run immediately after cloning. Only the Iraq
rows of the two cross-site scripts require the extra download.
