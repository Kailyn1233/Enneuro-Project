import os

class Config:
    HOST = os.environ.get('SERVING_HOST', 'localhost')
    PORT = int(os.environ.get('SERVING_PORT', 8080))
    MODEL_PATH = os.environ.get('MODEL_PATH', './models')
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    METRICS_ENABLED = os.environ.get('METRICS_ENABLED', 'true').lower() == 'true'
    
    @classmethod
    def from_env(cls):
        return cls()
