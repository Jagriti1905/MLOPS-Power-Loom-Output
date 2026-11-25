
import hashlib
import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlflow.tracking import MlflowClient

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
with open(BASE_DIR / "params.yaml", "r") as f:
    params = yaml.safe_load(f)
DATA_FILE = BASE_DIR / params["data"]["processed_path"]
RAW_DATA_FILE = BASE_DIR / params["data"]["raw_path"]
data_sha256 = hashlib.sha256(RAW_DATA_FILE.read_bytes()).hexdigest()

# MLflow configuration
mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
mlflow.set_experiment(params["mlflow"]["experiment_name"])

MODEL_NAME = params["mlflow"]["registered_model_name"]

# Run existing cleaning pipeline
from clean_data import clean_data

print("Running data cleaning...")
clean_data()

# Load cleaned dataset
df = pd.read_csv(DATA_FILE)

# Features and target
numeric_features = [
    "loom_age_yr",
    "thread_count",
    "power_cuts_min",
    "weaver_exp_yr",
    "humidity_pct",
    "shift_hours"
]

categorical_features = ["fabric", "shed"]
target = params["data"]["target"]

X = df[numeric_features + categorical_features]
y = df[target]

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["training"]["test_size"],
    random_state=params["training"]["random_state"]
)


# Candidate models
candidates = {
    "RandomForest_100": RandomForestRegressor(
        n_estimators=100, max_depth=8, random_state=42
    ),
    "RandomForest_200": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
}

best_mae = float("inf")
best_run_id = None

# Train and track experiments
for model_name, estimator in candidates.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", estimator)
    ])

    with mlflow.start_run(run_name=model_name) as run:

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model", model_name)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(
    pipeline,
    artifact_path="model",
    serialization_format="cloudpickle"
)

        print(f"\n{model_name}")
        print("MAE:", mae)
        print("RMSE:", rmse)
        print("R2:", r2)

        if mae < best_mae:
            best_mae = mae
            best_run_id = run.info.run_id

# Register best model
model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

# Assign champion alias
client = MlflowClient()

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=registered_model.version
)

# Load champion and verify predictions
champion_uri = f"models:/{MODEL_NAME}@champion"
champion_model = mlflow.pyfunc.load_model(champion_uri)

final_predictions = champion_model.predict(X_test)

print("\nTraining completed successfully!")
print("Best MAE:", best_mae)
print("Best Run ID:", best_run_id)
print("Registered Model:", MODEL_NAME)
print("Model Version:", registered_model.version)
print("Champion predictions:", len(final_predictions))
