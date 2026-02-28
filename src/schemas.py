"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class WavePredictionRequest(BaseModel):
    """Input schema for wave height prediction."""

    Temperature: float = Field(..., description="Water temperature in degrees Celsius")
    Wind_Speed: float = Field(..., ge=0, description="Wind speed in m/s")
    Dir: float = Field(..., ge=0, lt=360, description="Wave direction in degrees (0-360)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Temperature": 14.0,
                    "Wind_Speed": 1.78,
                    "Dir": 353.2,
                }
            ]
        }
    }


class WavePredictionResponse(BaseModel):
    """Output schema for wave height prediction."""

    Hsig: float = Field(..., description="Predicted significant wave height in meters")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
