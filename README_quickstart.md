# eegemg_2025 – Quick Start

**Mouse EEG / EMG Analysis Pipeline (FASTER2-based)**
*For new lab members / 1-page summary*

---

## What this pipeline does

This pipeline analyzes **mouse EEG / EMG EDF files** to:

* Preprocess EEG/EMG signals
* Classify sleep stages (**Wake / NREM / REM**)
* Compute PSD and generate summary figures

Analysis is **CSV-metadata–driven** and optimized for **drug/intervention experiments**.

⚠️ **Run notebooks strictly in order**: **1 → 2 → 3**

---

## Minimal Workflow

1. Put **EDF + CSV files** in `raw_data/.../data/`
2. Run `1_preprocess_edf.ipynb`
3. Run `2_analyze_stage_n_PSD.ipynb`
4. Visually inspect sleep stages
5. Run `3_merge_and_plot_data.ipynb`

---

## Required Directory Structure

```
/your_project/
 ├─ raw_data/
 │   └─ kaist/
 │       └─ 2025XXXX_mouseID/
 │           └─ data/
 │               ├─ xxxx_C1.edf
 │               ├─ xxxx_C2.edf
 │               ├─ exp.info.csv
 │               ├─ drug.info.csv
 │               └─ mouse.info.csv
 └─ analyzed/   # auto-generated
```

⚠️ Directory name **must be `raw_data`** (hard-coded replacement logic).

---

## EDF File Rules (Critical)

* Filenames must contain `_C1`, `_C2`, ...
* Channel mapping:

| EDF suffix | Channel |
| ---------: | ------- |
|      `_C1` | `Ch0`   |
|      `_C2` | `Ch1`   |

These **must match** `Device label` in `mouse.info.csv`.

---

## Required CSV Files

### `exp.info.csv`

* Experiment label
* Rack label
* Start datetime / End datetime (`YYYY/MM/DD HH:MM`)
* Sampling freq (Hz)

### `drug.info.csv`

* `drugX_datetime` + `drugX_name` pairs
* Must include `vehicle` and `rapalog` (lowercase)

### `mouse.info.csv`

* Device label (`Ch0`, `Ch1`, ...)
* Mouse group, Mouse ID, DOB
* Stats report (`Yes` / `No`)

---

## Notebook Overview

### Step 1 – Preprocessing

`1_preprocess_edf.ipynb`

Set at notebook end:

```python
prj_dir = "/your_project/raw_data/kaist"
epoch_len_sec = 8
sample_freq = 128
```

Output: EEG/EMG `.pkl` files

---

### Step 2 – Sleep Stage & PSD

`2_analyze_stage_n_PSD.ipynb`

* 24 h window starting **6 h before drug administration**
* Output written to `analyzed/`

Visual check (strongly recommended):

```bash
python EEG_EMG_stage_viewer.py
```

---

### Step 3 – Merge & Plot

`3_merge_and_plot_data.ipynb`

* Generates hypnograms, PSD plots, summary figures

---

## Common Pitfalls

* Wrong directory name (`raw_data` required)
* EDF filename ↔ channel mismatch
* CSV formatting errors
* Sampling frequency mismatch

---

## Notes

* Based on **FASTER2**
* Customized for **pharmacological / intervention experiments**
* Always inspect sleep staging before final plots
