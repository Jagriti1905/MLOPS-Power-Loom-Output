# MP026 – Power Loom Output Prediction

## Project Overview

This project is part of the MLOps Mini Project (SCSE3040).

The objective is to build an ML-based system that predicts the metres of fabric woven per shift in power looms using operational and environmental information.

The project is based on data from the Lucknow Weavers' Cluster and implements an end-to-end MLOps pipeline covering data versioning, data cleaning, model training, experiment tracking, model serving, containerization, continuous integration, and Kubernetes deployment.

## Problem Details

- **Group ID:** G136
- **Problem ID:** MP026
- **Project:** Power Loom Output Prediction
- **Target Variable:** `metres_per_shift`
- **Task Type:** Regression

## Team Members

| Name            | Student ID  |
| --------------- | ----------- |
| Sonu Satrakumar | S24CSEU0555 |
| Jagriti Shukla  | S24CSEU0684 |
| Sidharth Kumar  | S24CSEU0546 |
| Jatin Sheoran   | S24CSEU0673 |

## Repository Structure

```text
MLOPS-Power-Loom-Output/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .dvc/
├── app/
│   └── main.py
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── later.csv
│   │   └── holdout_features.csv
│   └── processed/
│       └── train_cleaned.csv
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── models/
│   └── model.joblib
├── notebooks/
├── reports/
│   └── findings.yaml
├── src/
│   ├── mp026/
│   │   ├── __init__.py
│   │   └── clean.py
│   ├── check_data.py
│   ├── clean_data.py
│   └── train_model.py
├── tests/
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── .dvcignore
├── Dockerfile
├── mlflow.db
├── params.yaml
├── requirements.txt
├── requirements-docker.txt
├── analysis_output.txt
├── GROUP.md
├── PROBLEM.md
├── STUDENT_HANDBOOK.md
└── README.md
```

Dataset files are managed through DVC, with corresponding tracking files maintained in the repository.

## Dataset Description

The project contains three datasets:

- **train.csv:** Historical training data used for model development.
- **later.csv:** Later production data used for monitoring, drift detection, and model evaluation.
- **holdout_features.csv:** Unlabelled data used for final predictions.

### Input Features

| Feature          | Description                                |
| ---------------- | ------------------------------------------ |
| `loom_age_yr`    | Age of the power loom in years             |
| `thread_count`   | Thread count of the fabric                 |
| `power_cuts_min` | Duration of power interruptions in minutes |
| `weaver_exp_yr`  | Experience of the weaver in years          |
| `humidity_pct`   | Environmental humidity percentage          |
| `shift_hours`    | Duration of the working shift              |
| `fabric`         | Type of fabric being produced              |
| `shed`           | Production shed                            |

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
python -m dvc add data/processed/train_cleaned.csv
python -m dvc push
python -m dvc status
```

To retrieve tracked datasets:

```bash
python -m dvc pull
```

## Stage 1 – Data Quality Analysis

Seven data quality issues were investigated in `train.csv`.

| Data Issue        | Finding                               |
| ----------------- | ------------------------------------- |
| Leaking Column    | Present – `metres_checked_at_folding` |
| Unit Mixup        | Present – `power_cuts_min` in Shed 1  |
| Sentinel Value    | Not detected                          |
| Duplicate Rows    | 39 duplicate rows                     |
| Stuck Sensor      | Present – `shift_hours`               |
| Target Typos      | Not detected                          |
| Spelling Variants | Not detected                          |

### Important Findings

1. `metres_checked_at_folding` is highly correlated with the target and is unavailable in the holdout dataset.
2. `power_cuts_min` values in Shed 1 are recorded in seconds instead of minutes. Conversion factor: 60.
3. The dataset contains 39 duplicate rows.
4. `shift_hours` remained constant at 10.8 from 1 March 2026 to 20 March 2026.

Detailed findings are stored in `reports/findings.yaml`.

### Stage 1 Progress

- [x] GitHub repository setup
- [x] Project directory structure
- [x] DVC installation and configuration
- [x] Google Drive remote configuration
- [x] Dataset version control
- [x] Data quality analysis
- [x] Identification of data quality issues
- [x] Findings recorded in YAML
- [x] DVC remote verification

## Stage 2 – Data Cleaning, Model Training and MLflow

Stage 2 implements data preprocessing, regression model training, experiment tracking, model comparison, model registration, and automated testing.

### Data Cleaning

The following preprocessing operations are performed:

- Removal of the target leakage column.
- Removal of 39 duplicate records.
- Correction of the power cuts unit mismatch in Shed 1.
- Identification and treatment of stuck sensor readings.
- Handling of missing numerical and categorical values through the preprocessing pipeline.
- Standardization of numerical features.
- One-hot encoding of categorical features.

### Data Cleaning Results

| Parameter                                    | Value |
| -------------------------------------------- | ----: |
| Original dataset rows                        | 1,039 |
| Duplicate rows removed                       |    39 |
| Cleaned dataset rows                         | 1,000 |
| Stuck sensor records after duplicate removal |    96 |
| Original dataset columns                     |    12 |
| Cleaned dataset columns                      |    11 |

The cleaned dataset is stored at `data/processed/train_cleaned.csv`.

Run the cleaning pipeline:

```bash
python src/clean_data.py
```

### MLflow Configuration

- **Tracking URI:** `sqlite:///mlflow.db`
- **Original experiment name:** `mp026-power-loom`
- **Local recovery experiment:** `mp026-power-loom-local`
- **Registered model:** `PowerLoomOutput`
- **Model alias:** `champion`

MLflow is used to track training runs, hyperparameters, evaluation metrics, dataset hashes, and model artifacts.

The MLflow tracking database `mlflow.db` and exported deployment model `models/model.joblib` are maintained in the GitHub repository. Run artifacts are stored locally, and their accessibility must be ensured when transferring the project between systems.

### Training Experiments

The training pipeline evaluates multiple regression models, including:

- Random Forest Regressor with 100 trees
- Random Forest Regressor with 200 trees
- Gradient Boosting Regressor

The following results were recorded during the earlier training experiments:

| Model                     |    MAE |   RMSE |     R² |
| ------------------------- | -----: | -----: | -----: |
| Random Forest (200 trees) | 3.3846 | 4.1150 | 0.8734 |
| Gradient Boosting         | 2.7627 | 3.3258 | 0.9173 |

**MAE** is used as the primary model-selection metric. A lower MAE indicates a smaller average prediction error.

### Model Registration

Gradient Boosting achieved the lowest MAE in the earlier recorded experiments.

- **Best historical MAE:** 2.7627
- **Registered model:** `PowerLoomOutput`
- **Previously verified champion version:** 2
- **Champion alias:** `champion`
- **Deployment model:** `models/model.joblib`

The champion alias identifies the preferred registered model. The champion model was successfully loaded, and the deployment model synchronization was subsequently performed.

### Configuration Management

The `params.yaml` file stores project configuration, including:

- Dataset paths
- Target variable
- Data cleaning parameters
- Training and testing split
- Random seed
- Model hyperparameters
- MLflow configuration

### Run and View MLflow

Run the training script:

```bash
python src/train_model.py
```

Start the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

The MLflow UI allows users to compare training runs, inspect model parameters and metrics, view model artifacts, and access the model registry.

### Automated Testing

Pytest is used to validate the data cleaning pipeline and FastAPI endpoints.

Run the tests:

```bash
python -m pytest tests/ -q
```

Verified test result:

```text
7 passed in 2.67s
```

### Stage 2 Progress

- [x] Data cleaning pipeline
- [x] Feature preprocessing
- [x] Configuration management using params.yaml
- [x] Multiple regression models trained
- [x] Model evaluation using MAE, RMSE and R²
- [x] MLflow experiment tracking
- [x] Dataset SHA-256 hash logging
- [x] Model registry configuration
- [x] Champion alias assignment
- [x] Joblib model export
- [x] Champion model loading
- [x] Champion model synchronization
- [x] Automated testing – 7 tests passed
- [x] MLflow database and deployment model committed to GitHub
- [ ] Cross-system MLflow artifact accessibility verification

## Stage 3 – Model Serving, Docker, CI/CD and Kubernetes

Stage 3 implements the deployment infrastructure required to serve the trained machine learning model through a REST API, package the application into a Docker container, automate testing using GitHub Actions, and deploy the application using Kubernetes.

### 3.1 FastAPI Model Serving

FastAPI is used to expose the trained regression model as a REST API.

The application loads the serialized model from `models/model.joblib` and generates predictions using operational and environmental input features.

#### API Endpoints

| Method | Endpoint   | Description                           |
| ------ | ---------- | ------------------------------------- |
| GET    | `/health`  | Checks application health             |
| POST   | `/predict` | Predicts fabric output per shift      |
| GET    | `/docs`    | Interactive Swagger API documentation |

Start the FastAPI application:

```bash
python -m uvicorn app.main:app --reload
```

Access Swagger documentation:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Sample Prediction Request

```json
{
  "loom_age_yr": 12,
  "thread_count": 80,
  "power_cuts_min": 30,
  "weaver_exp_yr": 8,
  "humidity_pct": 65,
  "shift_hours": 8,
  "fabric": "Cotton",
  "shed": "Shed 1"
}
```

### Verified Prediction Response

```json
{
  "prediction": 47.0266
}
```

- **HTTP Status Code:** 200 OK
- **Predicted output:** 47.0266 metres per shift

This result was obtained during Swagger API testing before the final champion synchronization.

### Input Validation

FastAPI and Pydantic are used to validate incoming prediction requests and reject invalid input values.

### 3.2 Docker Containerization

Docker is used to package the FastAPI application, trained model, and runtime dependencies into a portable container.

#### Docker Configuration

| Parameter        | Value                    |
| ---------------- | ------------------------ |
| Base image       | python:3.14-slim         |
| Docker image     | mp026-api:1.0            |
| Container name   | mp026-container          |
| Application port | 8000                     |
| Model path       | /app/models/model.joblib |

The project includes:

- `Dockerfile` for building the application image.
- `.dockerignore` for excluding unnecessary files.
- `requirements-docker.txt` for container-compatible Python dependencies.

A separate Docker requirements file is maintained to avoid dependency conflicts between Windows and Linux environments.

Build the Docker image:

```bash
docker build -t mp026-api:1.0 .
```

Run the Docker container:

```bash
docker run -d --name mp026-container -p 8000:8000 mp026-api:1.0
```

Check running containers:

```bash
docker ps
```

Test the health endpoint:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

View container logs:

```bash
docker logs mp026-container
```

### 3.3 GitHub Actions CI/CD

GitHub Actions is configured to automate project testing whenever code is pushed to the repository or a pull request is created.

The workflow configuration is stored at:

```text
.github/workflows/ci.yml
```

The CI workflow performs the following operations:

1. Checks out the GitHub repository.
2. Configures the Python environment.
3. Installs the required dependencies.
4. Executes automated pytest tests.
5. Reports the workflow execution status.

The workflow helps identify integration errors and ensures that code changes are tested automatically.

### 3.4 Kubernetes Deployment

Kubernetes is used to orchestrate the containerized FastAPI application.

The deployment configuration is maintained in the `k8s/` directory.

#### Kubernetes Configuration

| Parameter            | Value         |
| -------------------- | ------------- |
| Deployment name      | mp026-api     |
| Docker image         | mp026-api:1.0 |
| Application replicas | 3             |
| Service name         | mp026-svc     |
| Service type         | ClusterIP     |
| Container port       | 8000          |
| Service port         | 80            |
| Health endpoint      | /health       |

The Kubernetes deployment is configured with three application replicas to support availability and load distribution.

#### Deployment Commands

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Check deployment status:

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

Check rollout status:

```bash
kubectl rollout status deployment/mp026-api
```

#### Kubernetes Health Probes

The deployment includes readiness and liveness probes.

| Probe     | Initial Delay |     Period | Endpoint |
| --------- | ------------: | ---------: | -------- |
| Readiness |     5 seconds | 10 seconds | /health  |
| Liveness  |    30 seconds | 15 seconds | /health  |

The readiness probe determines whether a container is ready to receive requests. The liveness probe determines whether the container is functioning correctly and allows Kubernetes to restart unhealthy containers.

#### Accessing the Kubernetes Application

```bash
kubectl port-forward service/mp026-svc 8080:80
```

Access the health endpoint:

```bash
curl http://localhost:8080/health
```

Access Swagger documentation:

[http://localhost:8080/docs](http://localhost:8080/docs)

### 3.5 API Logging

Application logging is included to support API debugging and operational monitoring.

Docker logs can be accessed using:

```bash
docker logs mp026-container
```

Kubernetes logs can be accessed using:

```bash
kubectl logs -l app=mp026-api --tail=30
```

These logs support the inspection of application startup, API requests, prediction activity, and application errors. Structured prediction log fields require final verification.

### Stage 3 Progress

- [x] FastAPI model-serving application
- [x] Health endpoint implementation
- [x] Prediction endpoint implementation
- [x] Input validation
- [x] Swagger API testing
- [x] Successful prediction with HTTP 200
- [x] Dockerfile creation
- [x] Dockerignore configuration
- [x] Docker-compatible dependency configuration
- [x] Docker image build
- [x] Docker container deployment
- [x] GitHub Actions workflow configuration
- [x] Kubernetes cluster configuration
- [x] Kubernetes deployment configuration
- [x] Three application replicas configured
- [x] Kubernetes service configuration
- [x] Liveness and readiness probes
- [x] API logging implementation
- [x] Deployment-related GitHub commits
- [ ] Final GitHub Actions green status verification
- [ ] Final verification of all three running replicas
- [ ] Structured prediction log verification

## Stage 4 – Model Monitoring and Retraining

Stage 4 will extend the existing MLOps pipeline with model monitoring, drift detection, retraining, and final holdout predictions.

Planned activities:

- [ ] Data drift detection using Population Stability Index (PSI)
- [ ] Model performance evaluation on the later dataset
- [ ] MAE comparison across production sheds
- [ ] Identification of the most drifted feature
- [ ] Model retraining using updated production data
- [ ] Comparison of existing and retrained models
- [ ] Champion model promotion decision
- [ ] Holdout dataset predictions
- [ ] Completion of the monitoring section in findings.yaml

## MLOps Pipeline Architecture

```text
Raw Dataset
     |
     v
Data Version Control (DVC)
     |
     v
Data Quality Analysis
     |
     v
Data Cleaning and Preprocessing
     |
     v
Machine Learning Model Training
     |
     v
MLflow Experiment Tracking
     |
     v
Model Registry and Champion Selection
     |
     v
Joblib Model Export
     |
     v
FastAPI Model Serving
     |
     v
Docker Containerization
     |
     v
Kubernetes Deployment
     |
     v
Model Monitoring and Retraining
       (Stage 4 - Pending)
```

GitHub Actions provides automated testing across the development workflow.

## Tools and Technologies

| Technology     | Purpose                                |
| -------------- | -------------------------------------- |
| Python         | Core programming language              |
| Pandas         | Data analysis and preprocessing        |
| NumPy          | Numerical operations                   |
| Scikit-learn   | Machine learning model development     |
| Git            | Source code version control            |
| GitHub         | Repository hosting and collaboration   |
| DVC            | Dataset version control                |
| Google Drive   | Remote dataset storage                 |
| MLflow         | Experiment tracking and model registry |
| Joblib         | Model serialization                    |
| PyYAML         | Configuration management               |
| FastAPI        | REST API development                   |
| Pydantic       | Input validation                       |
| Uvicorn        | ASGI application server                |
| Pytest         | Automated testing                      |
| Docker         | Application containerization           |
| GitHub Actions | Continuous integration                 |
| Kubernetes     | Container orchestration                |
| Swagger UI     | Interactive API documentation          |

## Overall Project Progress

| Stage   | Description                                     | Status                                               |
| ------- | ----------------------------------------------- | ---------------------------------------------------- |
| Stage 1 | Repository setup, DVC and data quality analysis | Completed                                            |
| Stage 2 | Data cleaning, model training and MLflow        | Implemented – final portability verification pending |
| Stage 3 | FastAPI, Docker, CI/CD and Kubernetes           | Implemented – final operational verification pending |
| Stage 4 | Monitoring, drift detection and retraining      | Pending                                              |

## Conclusion

The project implements an end-to-end MLOps workflow for predicting power loom production output. The system integrates dataset version control, automated data preprocessing, machine learning experiments, model registration, API-based prediction, containerization, continuous integration, and Kubernetes deployment.

The first three stages have been implemented, with final operational verification remaining. Stage 4 will extend the system with continuous model monitoring, drift detection, retraining, and holdout predictions to support long-term model reliability.
