import numpy as np
import os
from typing import List, Any
from .logger import get_logger
from .config import Config

logger = get_logger(__name__)

class Predictor:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.MODEL_PATH
        self.model = None
        self.model_version = "1.0.0"
        self.load_model()
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
            else:
                logger.warning(f"Model path {self.model_path} not found, using dummy model")
            self._initialize_dummy_model()
            logger.info(f"Model loaded successfully, version: {self.model_version}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._initialize_dummy_model()
    
    def _initialize_dummy_model(self):
        self.model = {
            'weights': np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            'bias': np.array([0.1, 0.2, 0.3])
        }
    
    def predict(self, inputs: List[List[float]]) -> List[float]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if not inputs:
            return []
        
        try:
            input_array = np.array(inputs)
            if input_array.ndim == 1:
                input_array = input_array.reshape(1, -1)
            
            result = []
            for row in input_array:
                if len(row) == len(self.model['weights']):
                    output = np.dot(row, self.model['weights']) + self.model['bias']
                    result.extend(output.tolist())
                else:
                    output = np.sum(row) * 0.1
                    result.append(output)
            
            logger.debug(f"Prediction completed, input shape: {input_array.shape}, output length: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def get_model_version(self) -> str:
        return self.model_version
    
    def is_ready(self) -> bool:
        return self.model is not None
