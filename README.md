# Ocean Wave Height Prediction with MLOps

Production-grade ML system for predicting significant wave heights (Hsig) from oceanic and environmental conditions. Originally an academic project for the Global Ocean Exploration Initiative (GOEI), rebuilt with a full MLOps architecture: experiment tracking, model serving, containerization, and orchestrated retraining.

## Architecture Overview

```
                +-----------------+
                |   Raw Buoy Data |
                |   (data/Hs.csv) |
                +--------+--------+
                         |
          +--------------v--------------+
          |  Phase 1: Research & EDA    |
          |  (notebooks/)               |
          |  - Feature engineering      |
          |  - Model comparison         |
          |  - MLflow experiment logs   |
          +--------------+--------------+
                         |
              +----------v----------+
              |  Serialized Model   |
              |  (models/*.pkl)     |
              +----------+----------+
                         |
         +---------------v---------------+
         |  Phase 2: Serving (FastAPI)   |
         |  POST /predict                |
         |  GET  /health                 |
         +---------------+---------------+
                         |
              +----------v----------+
              |  Phase 3: Docker    |
              |  Containerized API  |
              +----------+----------+
                         |
            +------------v------------+
            |  Phase 4: Airflow DAG   |
            |  Monthly retraining     |
            +-------------------------+
```

## Project Structure

```
ocean-wave-prediction/
├── dags/
│   └── retrain_pipeline.py        # Airflow DAG for scheduled retraining
├── data/
│   └── Hs.csv                     # Ocean dataset (68k+ records)
├── models/
│   └── lgbm_best.pkl              # Production model (LightGBM pipeline)
├── notebooks/
│   ├── mlruns/                    # MLflow run artifacts and metrics
│   ├── mlflow.db                  # MLflow experiment tracking database
│   └── phase_1_experiment.ipynb   # EDA, feature engineering, model selection
├── src/
│   ├── app.py                     # FastAPI prediction service
│   └── schemas.py                 # Pydantic request/response models
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## Model Summary

| Model | CV MSE | CV R2 | Inference (20k samples) | Size |
|---|---|---|---|---|
| Linear Regression (baseline) | 0.045 | 0.608 | - | - |
| Random Forest | 0.022 | 0.808 | 565 ms | 52.25 MB |
| XGBoost | 0.022 | 0.808 | 36 ms | 2.09 MB |
| **LightGBM (selected)** | **0.022** | **0.808** | **22 ms** | **0.30 MB** |

All three ensemble models converged on the same accuracy ceiling (R2=0.808), indicating the signal limit of the current feature set. LightGBM was selected for production due to **24x faster inference** and **180x smaller model size** compared to Random Forest.

### Feature Engineering Pipeline

| Feature | Transformation |
|---|---|
| `Dir` (wave direction) | Cyclic encoding to `sin`/`cos` components |
| `Temperature` | KBinsDiscretizer (3 quantile bins) + one-hot encoding |
| `Wind_Speed` | Median imputation + log1p + StandardScaler |
| `U10`, `Xp`, `Yp`, `Depth` | Dropped (99% missing or constant) |
| `Season` | Dropped (no discriminative power) |
| `Wave_Steepness` | Dropped (data leakage: derived from target) |
| `X-Windv`, `Y-Windv` | Dropped (redundant with Wind_Speed + Dir) |

## Quick Start

### Run locally

```bash
pip install -r requirements.txt
uvicorn src.app:app --reload
```

### Run with Docker

```bash
docker build -t wave-predictor .
docker run -p 8000:8000 wave-predictor
```

### Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Temperature": 14.0, "Wind_Speed": 1.78, "Dir": 353.2}'
```

Response:
```json
{"Hsig": 0.03444}
```

## MLOps Components

### Experiment Tracking (MLflow)
All model training runs are logged via MLflow with parameters, metrics, and artifacts. To view the experiment results:

```bash
cd notebooks/
mlflow ui
```

Then in the MLflow UI (http://localhost:5000):
1. Select **Model training** → **Experiments** from the left sidebar
2. Click the experiment named **"Maritime Wave Height Prediction"** in the main view to inspect logged parameters, metrics, and run comparisons

### Model Serving (FastAPI)
A lightweight REST API with Pydantic validation. The full sklearn pipeline (preprocessing + LightGBM) is loaded at startup, so raw feature values can be sent directly.

### Containerization (Docker)
The API is packaged in a slim Python 3.13 container for reproducible deployment.

### Orchestration (Airflow)
The `dags/retrain_pipeline.py` DAG defines a monthly retraining workflow:
1. **Extract** new buoy data
2. **Preprocess** with the same pipeline from Phase 1
3. **Train** a new LightGBM model
4. **Evaluate** against the current production model and promote only if improved

### Scaling Considerations
If GOEI scales to terabytes of global buoy data, the preprocessing pipeline (`prep_pipe`) can be translated to PySpark and executed on Databricks clusters, with the Airflow DAG orchestrating Spark jobs instead of local Python tasks.
