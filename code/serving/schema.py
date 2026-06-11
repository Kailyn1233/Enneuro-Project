from typing import List, Optional, Any
from pydantic import BaseModel

class PredictRequest(BaseModel):
    inputs: List[List[float]]
    version: Optional[str] = None

class PredictResponse(BaseModel):
    predictions: List[float]
    model_version: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_seconds: float

class ErrorResponse(BaseModel):
    error_code: int
    error_message: str
    timestamp: str

class PingResponse(BaseModel):
    status: str
    message: str = "pong"
