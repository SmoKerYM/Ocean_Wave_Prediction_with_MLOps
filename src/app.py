"""FastAPI application for serving wave height predictions."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cloudpickle
from fastapi import FastAPI, HTTPException
import pandas as pd

from src.schemas import WavePredictionRequest, WavePredictionResponse, HealthResponse

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "lgbm_best.pkl"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    if not MODEL_PATH.exists():
        logger.error("Model file not found at %s", MODEL_PATH)
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = cloudpickle.load(f)
    logger.info("Model loaded from %s", MODEL_PATH)
    yield


app = FastAPI(
    title="Ocean Wave Height Prediction API",
    description="Predict significant wave height (Hsig) from oceanic conditions.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.post("/predict", response_model=WavePredictionResponse)
def predict(request: WavePredictionRequest):
    """Predict significant wave height from input features.

    The loaded pipeline handles all preprocessing (cyclic encoding,
    temperature binning, wind speed scaling) internally.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build a DataFrame matching the raw columns the pipeline expects.
    # Columns dropped by the pipeline (Xp, Yp, Depth, U10, Season, etc.)
    # are filled with dummy values since ColumnTransformer drops them anyway.
    input_df = pd.DataFrame([{
        "Xp": 0.0,
        "Yp": 0.0,
        "Dir": request.Dir,
        "Depth": 0.0,
        "X-Windv": 0.0,
        "Y-Windv": 0.0,
        "U10": 0.0,
        "Season": "Summer",
        "Temperature": request.Temperature,
        "Wind_Speed": request.Wind_Speed,
        "Wave_Steepness": 0.0,
        "Wind_Dir_Category": "West",
    }])

    prediction = model.predict(input_df)[0]
    return WavePredictionResponse(Hsig=round(float(prediction), 5))
