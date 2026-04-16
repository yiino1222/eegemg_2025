# eegemg_2025

**Mouse EEG / EMG Analysis Pipeline (FASTER2-based)**

This repository provides a pipeline for analyzing mouse EEG and EMG data recorded in **EDF** format. The analysis consists of three sequential steps, now implemented as Python scripts (no notebooks required):

1. Preprocessing of raw EEG/EMG data (`pipeline_step1_preprocess.py`)
2. Sleep stage classification (Wake / NREM / REM) and PSD analysis (`pipeline_step2_analyze.py`)
3. Integration with experimental metadata (CSV) and visualization (`pipeline_step3_merge.py`)

> ⚠️ **Important**: The steps must be executed **strictly in the order**: **1 → 2 → 3**, whether you call the scripts individually or through `run_pipeline.py`.

---

## 🚀 Run the whole pipeline in Docker (one command)

1. Build the image

```bash
docker build -t eegemg-pipeline .
```

2. Prepare a config file (see `pipeline.config.example.json`) and place it at
   `/data/config.json` inside the container (mount it there from the host).

3. Run all steps in sequence with a single command

```bash
# Mount your data directory and config into the container
docker run --rm \
  -v /your_project:/data \
  -v /path/to/pipeline.json:/data/config.json \
  eegemg-pipeline
```

The pipeline now executes pure Python scripts (no notebook dependency):

- `pipeline_step1_preprocess.py` — preprocess raw EDF files and cache intermediate results
- `pipeline_step2_analyze.py` — perform staging, PSD calculations, and statistical summaries
- `pipeline_step3_merge.py` — merge analyzed outputs and generate figures

You can orchestrate everything with `run_pipeline.py` and a JSON config (recommended for Docker runs).
Use `--config /path/to/other.json` when you want to run with a different configuration file.

`pipeline.config.example.json` shows the expected keys:

```json
{
  "preprocess": {
    "prj_dir": "/data/raw_data/kaist",
    "result_dir_name": "result",
    "epoch_len_sec": 8,
    "sample_freq": 128,
    "overwrite": false,
    "offset_in_msec": 0
  },
  "analysis": {
    "prj_dir": "/data/raw_data/kaist",
    "output_dir_name": "analyzed",
    "faster_dir_list": null,
    "epoch_len_sec": 8,
    "result_dir_name": "result",
    "overwrite": false,
    "injection_before_hours": 6,
    "injection_after_hours": 18
  },
  "merge": {
    "analyzed_dir_list": [
      "/data/analyzed/kaist/20251120_KA001-004"
    ],
    "rename_dict": {},
    "exclude_mouse_list": [],
    "target_group": "WT",
    "comparison_mode": "drug",
    "comparison_drug": "vehicle",
    "mouse_groups_to_compare": [],
    "drug_names": ["vehicle", "drugA", "drugB"],
    "output_dir": "/data/figures/kaist",
    "epoch_len_sec": 8,
    "sample_freq": 128,
    "quant_time_windows": {
      "stage_before": [3, 5],
      "stage_after": [6, 7],
      "psd_before": [5, 5],
      "psd_after": [6, 7],
      "norm_psd_after": [6, 7]
    }
  }
}
```

If `merge.analyzed_dir_list` is empty (`[]`) when using `run_pipeline.py`, merge targets are auto-discovered from
the same step2 output-location rules used by `pipeline_step2_analyze.py` (based on detected `result` folders and
their mapped analyzed output directories), then used as merge inputs.
Merge also checks legacy analyzed subfolder patterns such as `<drug>_24h_before6h` in addition to the current
`<drug>_before{X}h_after{Y}h` format.

To compare mouse groups (e.g., WT vs KO), set:

```json
{
  "merge": {
    "comparison_mode": "mouse_group",
    "comparison_drug": "vehicle",
    "mouse_groups_to_compare": ["WT", "KO"]
  }
}
```

`quant_time_windows` lets you override the time ranges used in Step 3 for aggregation/plots (specified in hours). Each
entry is a `[start, end]` list; the defaults match the values above. Supported keys:

* `stage_before` / `stage_after`: Stage duration & bout metrics
* `psd_before` / `psd_after`: Raw PSD summaries
* `norm_psd_after`: Normalized PSD summaries

When running `pipeline_step3_merge.py` directly, the same mapping can be passed via
`--quant-time-windows '{"stage_after":[6,7], "psd_after":[6,7]}'`. The selected windows are also printed on the bar plot
axes so the quantified interval is explicit.

> **Note on bar graph windows (including `mouse_group_comparison` mode):** Quantitative bar graphs are always computed
> from the **after** window. In `merge_n_plot`, the bar graphs use `meta_stage_n_bout_df_after` (derived from
> `stage_after_window`) and `merge_norm_psd_ts_df_after` (derived from `norm_psd_after_window`); the **before** window is
> not used for bar graphs (it is used for plots like the PSD “before” panel). When you set
> `quant_time_windows.stage_after` and `quant_time_windows.norm_psd_after`, those **after** windows directly determine
> the bar graph quantification range.

---

## Overview of the Analysis Workflow

```
EDF + CSV (data)
   ↓
pipeline_step1_preprocess.py
   ↓
Preprocessed EEG/EMG (result)
   ↓
pipeline_step2_analyze.py
   ↓
Sleep stages, PSD, statistics (analyzed)
   ↓
pipeline_step3_merge.py
   ↓
Final figures (figures)
```

---

## Archived legacy utilities

The repository previously contained notebooks and GUI helpers that are **not** used by the three-step pipeline above. These
legacy assets are still available under the `archive/` directory for reference but are excluded from the active pipeline
execution path.

---

## 1. Environment Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/yiino1222/eegemg_2025
cd eegemg_2025
```

### 1.2 Python Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Important Note on Directory Naming

This pipeline **requires** the directory name `raw_data` to be present in the project path.

**Reason**: In Step 2, output paths are generated using:

```python
dir.replace("raw_data", "analyzed")
```

⚠️ Renaming `raw_data` will break the pipeline.

---

## 3. Raw Data Placement and CSV Files

### 3.1 Recommended Directory Structure

```
/your_project/
 ├─ raw_data/
 │   └─ kaist/
 │       └─ 20251120_KA001-004/
 │           └─ data/
 │               ├─ xxxx_C1.edf
 │               ├─ xxxx_C2.edf
 │               ├─ exp.info.csv
 │               ├─ drug.info.csv
 │               └─ mouse.info.csv
 └─ analyzed/              # automatically generated in Step 2
```

**Rules**:

* Recording directories must start with `2` (e.g., `20251120_...`)
* All **EDF** and **CSV** files must be placed in the `data/` directory

---

### 3.2 EDF File Naming Rules (Critical)

EDF filenames **must** include `_C1`, `_C2`, etc.

**Mapping rule**:

| EDF suffix | Channel label |
| ---------: | ------------- |
|      `_C1` | `Ch0`         |
|      `_C2` | `Ch1`         |

These `ChX` labels **must match** the `Device label` column in `mouse.info.csv`.

---

### 3.3 `exp.info.csv` (Experiment-level Metadata)

#### Required Columns

| Column name      | Description             |
| ---------------- | ----------------------- |
| Experiment label | Experiment identifier   |
| Rack label       | Recording environment   |
| Start datetime   | Recording start time    |
| End datetime     | Recording end time      |
| Sampling freq    | Sampling frequency (Hz) |

#### Example

```csv
Experiment label,Rack label,Start datetime,End datetime,Sampling freq
EEG_p-iino-1-1,EEG_A-E,2025/11/21 7:00,2025/12/5 7:00,128
```

**Notes**:

* Datetime format: `YYYY/MM/DD HH:MM`
* `Sampling freq` must match the value specified in your pipeline configuration or CLI options

---

### 3.4 `drug.info.csv` (Drug Administration Metadata)

#### Rules

* `drugX_name` and `drugX_datetime` must appear as **pairs**
* `X` can be variable length (`drug1`, `drug2`, `drug3`, ...)
* Drug names are normalized to lowercase in the pipeline
* The merge step uses `merge.drug_names` to select and order conditions for plotting

#### Example

```csv
...,drug1_datetime,drug1_name,drug2_datetime,drug2_name
...,2025/12/2 17:00,vehicle,2025/12/3 17:00,rapalog
```

You can also provide 3+ conditions, for example:

```csv
...,drug1_datetime,drug1_name,drug2_datetime,drug2_name,drug3_datetime,drug3_name,drug4_datetime,drug4_name
...,2025/12/2 17:00,vehicle,2025/12/3 17:00,druga,2025/12/4 17:00,drugb,2025/12/5 17:00,drugc
```

Quick templates (copy/paste):

```csv
# 2-condition
Experiment label,drug1_datetime,drug1_name,drug2_datetime,drug2_name
EXP001,2025/12/2 17:00,vehicle,2025/12/3 17:00,rapalog
```

```csv
# 3-condition
Experiment label,drug1_datetime,drug1_name,drug2_datetime,drug2_name,drug3_datetime,drug3_name
EXP001,2025/12/2 17:00,vehicle,2025/12/3 17:00,druga,2025/12/4 17:00,drugb
```

```csv
# 4-condition
Experiment label,drug1_datetime,drug1_name,drug2_datetime,drug2_name,drug3_datetime,drug3_name,drug4_datetime,drug4_name
EXP001,2025/12/2 17:00,vehicle,2025/12/3 17:00,druga,2025/12/4 17:00,drugb,2025/12/5 17:00,drugc
```

> Note: Step2 normalizes `drugX_name` to lowercase when parsing; keep names consistent with `merge.drug_names` for Step3.

---

### 3.5 `mouse.info.csv` (Animal Metadata)

#### Required Columns

| Column name  | Description        |
| ------------ | ------------------ |
| Device label | `Ch0`, `Ch1`, ...  |
| Mouse group  | Experimental group |
| Mouse ID     | Animal ID          |
| DOB          | Date of birth      |
| Stats report | `Yes` / `No`       |

#### Example

```csv
Device label,Mouse group,Mouse ID,DOB,Stats report
Ch0,synK-shMegf10,KA001,2022/7/25,Yes
```

⚠️ **Device label must exactly match** the channel labels derived from EDF filenames.

---

## 4. Analysis Procedure (Script-based)

### Option A: Run the entire pipeline

```bash
python run_pipeline.py --config /path/to/pipeline.json
```

### Option B: Run each step manually

**Step 1: Preprocessing**

```bash
python pipeline_step1_preprocess.py --prj_dir /your_project/raw_data/kaist \
  --result_dir_name result --epoch_len_sec 8 --sample_freq 128 --overwrite false
```

Outputs are written under `raw_data/.../result/pkl/` as `ChX_EEG.pkl` and `ChX_EMG.pkl`.

**Step 2: Sleep Stage Classification and PSD Analysis**

```bash
python pipeline_step2_analyze.py --prj_dir /your_project/raw_data/kaist \
  --output_dir_name analyzed --epoch_len_sec 8 --result_dir_name result
```

Outputs are written under drug-specific subfolders such as
`analyzed/.../vehicle_before6h_after18h/`, `analyzed/.../druga_before6h_after18h/`, etc.
When `drug.info.csv` is missing, Step2 still creates a drug-named subfolder (auto-detected
from metadata/path, fallback: `drug1_before6h_after18h`) to avoid overwriting root-level
analysis outputs.

By default, pipeline step 2 extracts a window from **6 hours before** to **18 hours after**
each injection (`injection_before_hours=6`, `injection_after_hours=18`).
Override these values in `config.json` under the `analysis` section.

> (Recommended) Visually inspect EEG/EMG traces and hypnograms before proceeding:
> ```bash
> python EEG_EMG_stage_viewer.py
> ```

**Step 3: Data Integration and Visualization**

```bash
python pipeline_step3_merge.py --analyzed_dir_list /your_project/analyzed/kaist/20251120_KA001-004 \
  --output_dir /your_project/figures/kaist \
  --comparison-mode drug
```

This step generates **hypnograms**, **PSD plots**, and **summary figures**.

By default, plots are written under `output_dir/<target_group>/` to avoid overwriting
results when multiple mouse groups are analyzed.

* Use `--comparison-mode drug` (default) to compare one or more drug conditions within a single mouse group (legacy behavior controlled by `--target-group`).
* Set the drug list with `--drug-names vehicle drugA drugB` (or via `merge.drug_names` in config).
* Use `--comparison-mode mouse_group` with `--comparison-drug vehicle` (or `rapalog`) and
  `--mouse-groups-to-compare WT KO` to generate WT vs KO plots. Outputs go under
  `output_dir/WT_vs_KO/` when `mouse_groups_to_compare` is provided.
* Use `--comparison-mode mouse_group` to compare two mouse groups within a single drug; set the reference drug with `--comparison-drug` and optionally limit groups via `--mouse-groups-to-compare`.

### Quick Reference (drug condition setup)

* 2-condition (legacy): `drug_names: ["vehicle", "rapalog"]`
* 3-condition: `drug_names: ["vehicle", "drugA", "drugB"]`
* 4-condition: `drug_names: ["vehicle", "drugA", "drugB", "drugC"]`
* Keep names consistent between:
  * `drug.info.csv` (`drugX_name`)
  * analyzed subdirectory names generated in step2 (`<drug>_before..._after...`)
  * merge configuration (`merge.drug_names`)

### Minimal Workflow Summary

1. Place EDF + `exp.info.csv` + `drug.info.csv` + `mouse.info.csv` in `data/`
2. Run `pipeline_step1_preprocess.py`
3. Run `pipeline_step2_analyze.py`
4. Visually inspect sleep stages (optional but recommended)
5. Run `pipeline_step3_merge.py`

---

## 6. Notes

* Most errors are caused by incorrect **paths**, **filenames**, or **CSV formatting**
* Always confirm channel mapping (`_C1 → Ch0`) and sampling frequency consistency
* This pipeline is based on **FASTER2** and customized for **experimental intervention analysis**
