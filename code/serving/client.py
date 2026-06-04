import requests
import json
from typing import List, Dict, Any, Optional
from .logger import get_logger
from .config import Config

logger = get_logger(__name__)

class ServingClient:
    def __init__(self, host: str = None, port: int = None):
        self.host = host or Config.HOST
        self.port = port or Config.PORT
        self.base_url = f"http://{self.host}:{self.port}"
        logger.info(f"Initialized client with base URL: {self.base_url}")
    
    def ping(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/ping", timeout=5)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Ping response: {result}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Ping failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def health(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Health response: {result}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def predict(self, inputs: List[List[float]], version: Optional[str] = None) -> Dict[str, Any]:
        try:
            payload = {"inputs": inputs}
            if version:
                payload["version"] = version
            
            logger.debug(f"Sending predict request with {len(inputs)} inputs")
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Predict response received, latency: {result.get('latency_ms', 'N/A')}ms")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Predict failed: {e}")
            return {"status": "error", "message": str(e)}

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Serving Client")
    parser.add_argument("--host", type=str, default=Config.HOST, help="Server host")
    parser.add_argument("--port", type=int, default=Config.PORT, help="Server port")
    parser.add_argument("--action", type=str, required=True, choices=["ping", "health", "predict"], help="Action to perform")
    parser.add_argument("--input", type=str, help="Input data for predict (JSON array)")
    
    args = parser.parse_args()
    
    client = ServingClient(host=args.host, port=args.port)
    
    if args.action == "ping":
        result = client.ping()
        print(json.dumps(result, indent=2))
    elif args.action == "health":
        result = client.health()
        print(json.dumps(result, indent=2))
    elif args.action == "predict":
        if not args.input:
            print("Error: --input is required for predict action")
            return
        
        try:
            inputs = json.loads(args.input)
            result = client.predict(inputs)
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError as e:
            print(f"Error parsing input: {e}")

if __name__ == "__main__":
    main()
