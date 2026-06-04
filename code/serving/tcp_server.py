import socket
import threading
import json
import time
from typing import Dict, Any
from .predictor import Predictor
from .logger import get_logger
from .config import Config

logger = get_logger(__name__)

class TCPServer:
    def __init__(self, host: str = None, port: int = None):
        self.host = host or Config.HOST
        self.port = port or (Config.PORT + 1)
        self.socket = None
        self.predictor = Predictor()
        self.running = False
    
    def handle_client(self, client_socket: socket.socket):
        try:
            data = b''
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b'\n' in data:
                    break
            
            if not data:
                return
            
            request = json.loads(data.decode('utf-8'))
            action = request.get('action')
            
            response: Dict[str, Any] = {}
            
            if action == 'ping':
                response = {'status': 'success', 'message': 'pong'}
            elif action == 'health':
                uptime = time.time() - self.start_time
                status = 'healthy' if self.predictor.is_ready() else 'unhealthy'
                response = {
                    'status': status,
                    'model_version': self.predictor.get_model_version(),
                    'uptime_seconds': uptime
                }
            elif action == 'predict':
                inputs = request.get('inputs', [])
                start_time = time.time()
                predictions = self.predictor.predict(inputs)
                latency = (time.time() - start_time) * 1000
                response = {
                    'predictions': predictions,
                    'model_version': self.predictor.get_model_version(),
                    'latency_ms': latency
                }
            else:
                response = {'status': 'error', 'message': f'Unknown action: {action}'}
            
            response_data = json.dumps(response) + '\n'
            client_socket.sendall(response_data.encode('utf-8'))
            logger.debug(f"TCP response sent: {response}")
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            client_socket.close()
    
    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(100)
        self.running = True
        self.start_time = time.time()
        logger.info(f"TCP server started on {self.host}:{self.port}")
        
        while self.running:
            try:
                client_socket, addr = self.socket.accept()
                logger.debug(f"New client connected: {addr}")
                thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                thread.daemon = True
                thread.start()
            except Exception as e:
                if self.running:
                    logger.error(f"Server error: {e}")
    
    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()
        logger.info("TCP server stopped")

if __name__ == "__main__":
    server = TCPServer()
    try:
        server.run()
    except KeyboardInterrupt:
        server.stop()
