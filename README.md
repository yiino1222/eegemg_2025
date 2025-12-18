eegemg_2025
Mouse EEG / EMG Analysis Pipeline (FASTER2-based)

This repository provides a pipeline for analyzing mouse EEG and EMG data recorded in EDF format.
The analysis consists of three sequential steps:

Preprocessing of raw EEG/EMG data

Sleep stage classification (Wake / NREM / REM) and PSD analysis

Integration with experimental metadata (CSV) and visualization

⚠️ The steps must be executed strictly in the order: 1 → 2 → 3.

Overview of the Analysis Workflow
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

1. Environment Setup
1.1 Clone the Repository
git clone https://github.com/yiino1222/eegemg_2025
cd eegemg_2025

1.2 Python Environment (Recommended)
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install notebook

Important Note on Directory Naming

This pipeline requires the directory name raw_data to be present in the project path.

Reason:
In Step 2, output paths are generated using:

dir.replace("raw_data", "analyzed")


Renaming raw_data will break the pipeline.

2. Raw Data Placement and CSV Files
2.1 Recommended Directory Structure
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
 └─ analyzed/   (automatically generated in Step 2)


Recording directories must start with 2 (e.g., 20251120_...)

All EDF and CSV files must be placed in the data/ directory

2.2 EDF File Naming Rules (Critical)

EDF filenames must include _C1, _C2, etc.

Mapping rule:

_C1 → Ch0

_C2 → Ch1

These ChX labels must match the Device label column in mouse.info.csv

2.3 exp.info.csv (Experiment-level Metadata)

Required columns

Column name	Description
Experiment label	Experiment identifier
Rack label	Recording environment
Start datetime	Recording start time
End datetime	Recording end time
Sampling freq	Sampling frequency (Hz)

Example

Experiment label,Rack label,Start datetime,End datetime,Sampling freq
EEG_p-iino-1-1,EEG_A-E,2025/11/21 7:00,2025/12/5 7:00,128


Datetime format: YYYY/MM/DD HH:MM

Sampling freq must match the value specified in the notebooks

2.4 drug.info.csv (Drug Administration Metadata)

Rules

drugX_name and drugX_datetime must appear as pairs

drugX_name must include vehicle and rapalog

Matching is case-sensitive (use lowercase)

Example

...,drug1_datetime,drug1_name,drug2_datetime,drug2_name
...,2025/12/2 17:00,vehicle,2025/12/3 17:00,rapalog

2.5 mouse.info.csv (Animal Metadata)

Required columns

Column name	Description
Device label	Ch0, Ch1, ...
Mouse group	Experimental group
Mouse ID	Animal ID
DOB	Date of birth
Stats report	Yes / No

Example

Device label,Mouse group,Mouse ID,DOB,Stats report
Ch0,synK-shMegf10,KA001,2022/7/25,Yes


⚠️ Device label must exactly match the channel labels derived from EDF filenames.

3. Analysis Procedure
Step 1: Preprocessing

1_preprocess_edf.ipynb

Set parameters at the end of the notebook:

prj_dir = "/your_project/raw_data/kaist"
result_dir_name = "result"
epoch_len_sec = 8
sample_freq = 128


Output

raw_data/.../result/pkl/Ch0_EEG.pkl
raw_data/.../result/pkl/Ch0_EMG.pkl

Step 2: Sleep Stage Classification and PSD Analysis

2_analyze_stage_n_PSD.ipynb

Analysis windows:

Vehicle: 24 h window starting 6 h before administration

Rapalog: 24 h window starting 6 h before administration

Output

analyzed/.../vehicle_24h_before6h/
analyzed/.../rapalog_24h_before6h/

(Recommended) Visual Inspection of Sleep Stages
python EEG_EMG_stage_viewer.py


Always visually inspect EEG/EMG traces and hypnograms before proceeding.

Step 3: Data Integration and Visualization

3_merge_and_plot_data.ipynb

Example configuration:

analyzed_dir_list = [
  "/your_project/analyzed/kaist/20251120_KA001-004"
]


This step generates hypnograms, PSD plots, and summary figures.

Minimal Workflow Summary
1. Place EDF + exp.info.csv + drug.info.csv + mouse.info.csv in data/
2. Run 1_preprocess_edf.ipynb
3. Run 2_analyze_stage_n_PSD.ipynb
4. Visually inspect stages
5. Run 3_merge_and_plot_data.ipynb

Notes

Most errors are caused by incorrect paths, filenames, or CSV formatting.

Always confirm channel mapping (_C1 → Ch0) and sampling frequency consistency.

This pipeline is based on FASTER2 and customized for experimental intervention analysis.
