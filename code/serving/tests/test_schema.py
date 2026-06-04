import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.schema import PredictRequest, PredictResponse, HealthResponse, PingResponse, ErrorResponse

class TestSchema(unittest.TestCase):
    def test_predict_request(self):
        request = PredictRequest(inputs=[[1.0, 2.0, 3.0]], version="1.0")
        self.assertEqual(request.inputs, [[1.0, 2.0, 3.0]])
        self.assertEqual(request.version, "1.0")
    
    def test_predict_request_no_version(self):
        request = PredictRequest(inputs=[[1.0, 2.0, 3.0]])
        self.assertIsNone(request.version)
    
    def test_predict_response(self):
        response = PredictResponse(predictions=[0.5, 0.8], model_version="1.0", latency_ms=10.5)
        self.assertEqual(response.predictions, [0.5, 0.8])
        self.assertEqual(response.model_version, "1.0")
        self.assertEqual(response.latency_ms, 10.5)
    
    def test_health_response(self):
        response = HealthResponse(status="healthy", model_version="1.0", uptime_seconds=120.5)
        self.assertEqual(response.status, "healthy")
        self.assertEqual(response.model_version, "1.0")
        self.assertEqual(response.uptime_seconds, 120.5)
    
    def test_ping_response(self):
        response = PingResponse(status="success")
        self.assertEqual(response.status, "success")
        self.assertEqual(response.message, "pong")
    
    def test_error_response(self):
        response = ErrorResponse(error_code=500, error_message="Internal error", timestamp="2024-01-01 12:00:00")
        self.assertEqual(response.error_code, 500)
        self.assertEqual(response.error_message, "Internal error")

if __name__ == '__main__':
    unittest.main()
