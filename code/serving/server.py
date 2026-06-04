import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
from .predictor import Predictor
from .logger import get_logger
from .config import Config
from .schema import (
    PredictRequest, PredictResponse, 
    HealthResponse, ErrorResponse, PingResponse
)

logger = get_logger(__name__)

class ServingServer:
    def __init__(self, host: str = None, port: int = None):
        self.host = host or Config.HOST
        self.port = port or Config.PORT
        self.app = FastAPI(title="EnNeuro Serving", version="1.0.0")
        self.predictor = Predictor()
        self.start_time = time.time()
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.get("/ping", response_model=PingResponse)
        async def ping():
            logger.info("Received ping request")
            return PingResponse(status="success")
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            uptime = time.time() - self.start_time
            status = "healthy" if self.predictor.is_ready() else "unhealthy"
            logger.info(f"Health check: {status}, uptime: {uptime:.2f}s")
            return HealthResponse(
                status=status,
                model_version=self.predictor.get_model_version(),
                uptime_seconds=uptime
            )
        
        @self.app.post("/predict", response_model=PredictResponse)
        async def predict(request: PredictRequest):
            start_time = time.time()
            try:
                logger.debug(f"Received predict request with {len(request.inputs)} inputs")
                predictions = self.predictor.predict(request.inputs)
                latency = (time.time() - start_time) * 1000
                logger.info(f"Prediction completed in {latency:.2f}ms")
                return PredictResponse(
                    predictions=predictions,
                    model_version=self.predictor.get_model_version(),
                    latency_ms=latency
                )
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error_code=500,
                    error_message=str(exc),
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                ).dict()
            )
    
    def run(self):
        import uvicorn
        logger.info(f"Starting server on {self.host}:{self.port}")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level=Config.LOG_LEVEL.lower()
        )

if __name__ == "__main__":
    server = ServingServer()
    server.run()
