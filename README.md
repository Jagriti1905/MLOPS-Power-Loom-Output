# MP026 – Power Loom Output Prediction

## Project Overview
This project is part of the MLOps Mini Project (SCSE3040).

The objective is to build an ML-based system that predicts the metres of fabric woven per shift in power looms using operational and environmental information.

The project is based on data from the Lucknow Weavers' Cluster.

## Problem Details
- **Group ID:** G136
- **Problem ID:** MP026
- **Project:** Power Loom Output Prediction
- **Target Variable:** `metres_per_shift`
- **Task Type:** Regression

## Team Members

| Name | Student ID |
|---|---|
| Sonu Satrakumar | S24CSEU0555 |
| Jagriti Shukla | S24CSEU0684 |
| Sidharth Kumar | S24CSEU0546 |

## Repository Structure

```text
MLOPS-Power-Loom-Output/
├── data/
│   └── raw/
│       ├── train.csv
│       ├── later.csv
│       └── holdout_features.csv
├── reports/
│   └── findings.yaml
├── src/
│   └── check_data.py
├── analysis_output.txt
├── .dvc/
├── .gitignore
├── GROUP.md
├── PROBLEM.md
└── README.md
```

## Dataset Description

The project contains three datasets:

- **train.csv:** Historical training data.
- **later.csv:** Later production data used for monitoring and model evaluation.
- **holdout_features.csv:** Data used for final predictions.

The dataset includes features such as loom age, thread count, power cuts, weaver experience, humidity, shift hours, fabric type and shed.

The target variable is `metres_per_shift`.

## Data Version Control (DVC)

DVC is used to track and version the dataset files.

Google Drive is configured as the DVC remote storage.

Commands used:

```bash
python -m dvc init
python -m dvc add data/raw/train.csv
python -m dvc add data/raw/later.csv
python -m dvc add data/raw/holdout_features.csv
python -m dvc push
python -m dvc status
```

## Stage 1 – Data Quality Analysis

Seven data quality issues were investigated in `train.csv`.

| Data Issue | Finding |
|---|---|
| Leaking Column | Present – `metres_checked_at_folding` |
| Unit Mixup | Present – `power_cuts_min` in Shed 1 |
| Sentinel Value | Not detected |
| Duplicate Rows | 39 duplicate rows |
| Stuck Sensor | Present – `shift_hours` |
| Target Typos | Not detected |
| Spelling Variants | Not detected |

### Important Findings

1. `metres_checked_at_folding` is highly correlated with the target and is unavailable in the holdout dataset.
2. `power_cuts_min` values in Shed 1 are recorded in seconds instead of minutes. Conversion factor: 60.
3. The dataset contains 39 duplicate rows.
4. `shift_hours` remained constant at 10.8 from 1 March 2026 to 20 March 2026.

Detailed findings are stored in `reports/findings.yaml`.

## Tools and Technologies

- Python
- Pandas
- Git and GitHub
- DVC
- Google Drive

## Project Progress

- [x] GitHub repository setup
- [x] Dataset tracking using DVC
- [x] Google Drive remote configuration
- [x] Data quality analysis
- [x] Stage 1 findings recorded in YAML

More project stages will be documented as development progresses.
