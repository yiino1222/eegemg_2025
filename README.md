# eegemg_2025

**Mouse EEG / EMG Analysis Pipeline (FASTER2-based)**

This repository provides a pipeline for analyzing mouse EEG and EMG data recorded in **EDF** format. The analysis consists of three sequential steps:

1. Preprocessing of raw EEG/EMG data
2. Sleep stage classification (Wake / NREM / REM) and PSD analysis
3. Integration with experimental metadata (CSV) and visualization

> ⚠️ **Important**: The steps must be executed **strictly in the order**: **1 → 2 → 3**.

---

## Overview of the Analysis Workflow

```
EDF + CSV (data)
   ↓
1_preprocess_edf.ipynb
   ↓
Preprocessed EEG/EMG (result)
   ↓
2_analyze_stage_n_PSD.ipynb
   ↓
Sleep stages, PSD, statistics (analyzed)
   ↓
3_merge_and_plot_data.ipynb
   ↓
Final figures (figures)
```

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
pip install notebook
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
* `Sampling freq` must match the value specified in the notebooks

---

### 3.4 `drug.info.csv` (Drug Administration Metadata)

#### Rules

* `drugX_name` and `drugX_datetime` must appear as **pairs**
* `drugX_name` must include `vehicle` and `rapalog`
* Matching is **case-sensitive** (use lowercase)

#### Example

```csv
...,drug1_datetime,drug1_name,drug2_datetime,drug2_name
...,2025/12/2 17:00,vehicle,2025/12/3 17:00,rapalog
```

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

## 4. Analysis Procedure

### Step 1: Preprocessing

**Notebook**: `1_preprocess_edf.ipynb`

Set parameters at the end of the notebook:

```python
prj_dir = "/your_project/raw_data/kaist"
result_dir_name = "result"
epoch_len_sec = 8
sample_freq = 128
```

**Output**:

```
raw_data/.../result/pkl/Ch0_EEG.pkl
raw_data/.../result/pkl/Ch0_EMG.pkl
```

---

### Step 2: Sleep Stage Classification and PSD Analysis

**Notebook**: `2_analyze_stage_n_PSD.ipynb`

#### Analysis Windows

* **Vehicle**: 24 h window starting **6 h before** administration
* **Rapalog**: 24 h window starting **6 h before** administration

**Output**:

```
analyzed/.../vehicle_24h_before6h/
analyzed/.../rapalog_24h_before6h/
```

#### (Recommended) Visual Inspection of Sleep Stages

```bash
python EEG_EMG_stage_viewer.py
```

> Always visually inspect EEG/EMG traces and hypnograms before proceeding.

---

### Step 3: Data Integration and Visualization

**Notebook**: `3_merge_and_plot_data.ipynb`

#### Example Configuration

```python
analyzed_dir_list = [
    "/your_project/analyzed/kaist/20251120_KA001-004"
]
```

This step generates **hypnograms**, **PSD plots**, and **summary figures**.

---

## 5. Minimal Workflow Summary

1. Place EDF + `exp.info.csv` + `drug.info.csv` + `mouse.info.csv` in `data/`
2. Run `1_preprocess_edf.ipynb`
3. Run `2_analyze_stage_n_PSD.ipynb`
4. Visually inspect sleep stages
5. Run `3_merge_and_plot_data.ipynb`

---

## 6. Notes

* Most errors are caused by incorrect **paths**, **filenames**, or **CSV formatting**
* Always confirm channel mapping (`_C1 → Ch0`) and sampling frequency consistency
* This pipeline is based on **FASTER2** and customized for **experimental intervention analysis**
