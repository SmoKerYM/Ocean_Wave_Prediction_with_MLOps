"""
Apache Airflow DAG for scheduled retraining of the wave height prediction model.

This DAG orchestrates the full ML pipeline:
  1. Extract new ocean buoy data from the data source.
  2. Run the preprocessing pipeline (prep_pipe) on the new data.
  3. Train a new LightGBM model using the best known hyperparameters.
  4. Evaluate the new model and replace the production model if it improves.

Schedule: runs on the 1st of every month (when new buoy data typically arrives).
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def extract_new_data(**kwargs):
    """Pull the latest ocean buoy measurements into data/Hs.csv."""
    import pandas as pd
    from pathlib import Path

    data_path = Path(__file__).resolve().parent.parent / "data" / "Hs.csv"
    df = pd.read_csv(data_path)
    kwargs["ti"].xcom_push(key="row_count", value=len(df))
    print(f"Loaded {len(df)} rows from {data_path}")


def run_preprocessing(**kwargs):
    """Apply the preprocessing pipeline and split the data."""
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    data_path = Path(__file__).resolve().parent.parent / "data" / "Hs.csv"
    df = pd.read_csv(data_path).drop_duplicates()

    X = df.drop(columns=["Hsig"])
    y = df["Hsig"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Persist splits for downstream tasks via temp parquet files
    artifact_dir = Path(__file__).resolve().parent.parent / "models" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(artifact_dir / "X_train.parquet")
    X_test.to_parquet(artifact_dir / "X_test.parquet")
    y_train.to_frame().to_parquet(artifact_dir / "y_train.parquet")
    y_test.to_frame().to_parquet(artifact_dir / "y_test.parquet")

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")


def train_model(**kwargs):
    """Train a LightGBM model with the best hyperparameters from Phase 1."""
    import numpy as np
    import pandas as pd
    import cloudpickle
    from pathlib import Path
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        StandardScaler, FunctionTransformer, OneHotEncoder, KBinsDiscretizer,
    )
    from lightgbm import LGBMRegressor

    artifact_dir = Path(__file__).resolve().parent.parent / "models" / "artifacts"
    X_train = pd.read_parquet(artifact_dir / "X_train.parquet")
    y_train = pd.read_parquet(artifact_dir / "y_train.parquet").squeeze()

    # --- Reproduce the preprocessing pipeline from Phase 1 ---
    def sin_cos_transform(X):
        radians = np.deg2rad(X)
        return np.concatenate([np.sin(radians), np.cos(radians)], axis=1)

    def dir_feature_names(self, names):
        return ["Dir_sin", "Dir_cos"]

    cyclic_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("sincos", FunctionTransformer(sin_cos_transform, validate=True,
                                       feature_names_out=dir_feature_names)),
    ])

    temp_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("bin", KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")),
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ])

    wind_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cyclic", cyclic_pipeline, ["Dir"]),
            ("temp", temp_pipeline, ["Temperature"]),
            ("wind", wind_pipeline, ["Wind_Speed"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Best hyperparameters from Phase 1 experiment
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LGBMRegressor(
            random_state=42,
            verbose=-1,
            subsample=0.7,
            num_leaves=31,
            n_estimators=100,
            learning_rate=0.05,
            colsample_bytree=1.0,
        )),
    ])

    full_pipeline.fit(X_train, y_train)

    model_path = Path(__file__).resolve().parent.parent / "models" / "lgbm_candidate.pkl"
    with open(model_path, "wb") as f:
        cloudpickle.dump(full_pipeline, f)

    print(f"Candidate model saved to {model_path}")


def evaluate_and_promote(**kwargs):
    """Evaluate the candidate model; replace production model if it improves."""
    import pandas as pd
    import cloudpickle
    from pathlib import Path
    from sklearn.metrics import mean_squared_error, r2_score

    model_dir = Path(__file__).resolve().parent.parent / "models"
    artifact_dir = model_dir / "artifacts"

    X_test = pd.read_parquet(artifact_dir / "X_test.parquet")
    y_test = pd.read_parquet(artifact_dir / "y_test.parquet").squeeze()

    candidate_path = model_dir / "lgbm_candidate.pkl"
    production_path = model_dir / "lgbm_best.pkl"

    with open(candidate_path, "rb") as f:
        candidate = cloudpickle.load(f)

    y_pred = candidate.predict(X_test)
    new_mse = mean_squared_error(y_test, y_pred)
    new_r2 = r2_score(y_test, y_pred)
    print(f"Candidate — MSE: {new_mse:.4f}, R2: {new_r2:.4f}")

    # Compare against the current production model (if it exists)
    promote = True
    if production_path.exists():
        with open(production_path, "rb") as f:
            production = cloudpickle.load(f)
        y_prod = production.predict(X_test)
        prod_mse = mean_squared_error(y_test, y_prod)
        print(f"Production — MSE: {prod_mse:.4f}")
        promote = new_mse < prod_mse

    if promote:
        candidate_path.rename(production_path)
        print("Candidate promoted to production.")
    else:
        candidate_path.unlink()
        print("Candidate discarded; production model retained.")


with DAG(
    dag_id="wave_prediction_retraining",
    default_args=default_args,
    description="Monthly retraining pipeline for ocean wave height prediction",
    schedule="0 0 1 * *",  # 1st of every month
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "retraining"],
) as dag:

    t1 = PythonOperator(task_id="extract_new_data", python_callable=extract_new_data)
    t2 = PythonOperator(task_id="run_preprocessing", python_callable=run_preprocessing)
    t3 = PythonOperator(task_id="train_model", python_callable=train_model)
    t4 = PythonOperator(task_id="evaluate_and_promote", python_callable=evaluate_and_promote)

    t1 >> t2 >> t3 >> t4
