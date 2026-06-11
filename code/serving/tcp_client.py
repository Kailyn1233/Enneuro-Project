import socket
import json
from typing import List, Dict, Any, Optional
from .logger import get_logger
from .config import Config

logger = get_logger(__name__)

class TCPClient:
    def __init__(self, host: str = None, port: int = None):
        self.host = host or Config.HOST
        self.port = port or (Config.PORT + 1)
        logger.info(f"TCP client initialized for {self.host}:{self.port}")
    
    def _send_request(self, action: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(5)
                sock.connect((self.host, self.port))
                
                request = {'action': action}
                if data:
                    request.update(data)
                
                request_data = json.dumps(request) + '\n'
                sock.sendall(request_data.encode('utf-8'))
                
                response = b''
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    response += chunk
                    if b'\n' in response:
                        break
                
                result = json.loads(response.decode('utf-8'))
                logger.debug(f"TCP response received: {result}")
                return result
        except Exception as e:
            logger.error(f"TCP request failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def ping(self) -> Dict[str, Any]:
        return self._send_request('ping')
    
    def health(self) -> Dict[str, Any]:
        return self._send_request('health')
    
    def predict(self, inputs: List[List[float]], version: Optional[str] = None) -> Dict[str, Any]:
        data = {'inputs': inputs}
        if version:
            data['version'] = version
        return self._send_request('predict', data)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TCP Serving Client")
    parser.add_argument("--host", type=str, default=Config.HOST, help="Server host")
    parser.add_argument("--port", type=int, default=Config.PORT + 1, help="Server port")
    parser.add_argument("--action", type=str, required=True, choices=["ping", "health", "predict"], help="Action to perform")
    parser.add_argument("--input", type=str, help="Input data for predict (JSON array)")
    
    args = parser.parse_args()
    
    client = TCPClient(host=args.host, port=args.port)
    
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
