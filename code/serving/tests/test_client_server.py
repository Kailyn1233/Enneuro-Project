import unittest
import sys
import os
import threading
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.server import ServingServer
from serving.client import ServingClient

class TestClientServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ServingServer(host="localhost", port=8081)
        cls.server_thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.server_thread.start()
        time.sleep(2)
        cls.client = ServingClient(host="localhost", port=8081)
    
    def test_ping(self):
        result = self.client.ping()
        self.assertEqual(result.get("status"), "success")
    
    def test_health(self):
        result = self.client.health()
        self.assertIn("status", result)
        self.assertIn("model_version", result)
    
    def test_predict(self):
        inputs = [[1.0, 2.0, 3.0]]
        result = self.client.predict(inputs)
        self.assertIn("predictions", result)
        self.assertIn("model_version", result)
        self.assertIn("latency_ms", result)

if __name__ == '__main__':
    unittest.main()
