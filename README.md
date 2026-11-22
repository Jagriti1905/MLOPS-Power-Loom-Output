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



## Stage 2 – MLflow Experiment Tracking and Model Registry

Stage 2 adds experiment tracking, model comparison, and model registration using MLflow.

### MLflow Configuration

- **Tracking URI:** `sqlite:///mlflow.db`
- **Experiment name:** `mp026-power-loom`
- **Registered model:** `PowerLoomOutput`
- **Model alias:** `champion`

The MLflow tracking database and run artifacts are local project files. They should not be committed to GitHub; `mlflow.db` and `mlruns/` are ignored through `.gitignore`.

### Training Experiments

The training pipeline evaluates multiple regression models and logs their parameters and evaluation metrics to MLflow.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest (200 trees) | 3.3846 | 4.1150 | 0.8734 |
| Gradient Boosting | 2.7627 | 3.3258 | 0.9173 |

**MAE** is used as the primary model-selection metric. A lower MAE indicates a smaller average prediction error.

### Best Model and Registration

- **Selected model:** Gradient Boosting
- **Best MAE:** 2.7627
- **Registered model:** `PowerLoomOutput`
- **Registered version:** `1`
- **Champion alias:** `champion`

The selected model is logged with MLflow and registered in the MLflow Model Registry. The `champion` alias identifies the current preferred model version.

### Run and View MLflow

From the project root, activate the project environment and run the training script:

```powershell
python src/train_model.py
```

Start the MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000` in a browser.

In the MLflow UI:
1. Open the `mp026-power-loom` experiment.
2. Select **Training runs** to compare runs and their metrics.
3. Open a run to inspect its parameters, metrics, and artifacts.
4. Open **Model registry** to view `PowerLoomOutput`, version `1`, and its `champion` alias.

### Stage 2 Progress

- [x] MLflow tracking configured
- [x] Multiple model/configuration experiments executed
- [x] Experiment parameters and metrics logged
- [x] Models compared using MAE
- [x] Best model selected
- [x] Best model registered as `PowerLoomOutput`
- [x] `champion` alias assigned

## Tools and Technologies

- Python
- Pandas
- Git and GitHub
- DVC
- Google Drive
- Scikit-learn
- MLflow

## Project Progress

- [x] GitHub repository setup
- [x] Dataset tracking using DVC
- [x] Google Drive remote configuration
- [x] Data quality analysis
- [x] Stage 1 findings recorded in YAML
- [x] Stage 2 MLflow experiment tracking and model registry

Stage 3 will add the model-serving API, Docker, CI/CD, and Kubernetes components.
