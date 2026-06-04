import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.predictor import Predictor

class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = Predictor()
    
    def test_predictor_initialization(self):
        self.assertIsNotNone(self.predictor.model)
        self.assertEqual(self.predictor.get_model_version(), "1.0.0")
        self.assertTrue(self.predictor.is_ready())
    
    def test_predict_single_input(self):
        inputs = [[1.0, 2.0, 3.0]]
        result = self.predictor.predict(inputs)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
    
    def test_predict_multiple_inputs(self):
        inputs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = self.predictor.predict(inputs)
        self.assertIsInstance(result, list)
    
    def test_empty_input(self):
        inputs = []
        result = self.predictor.predict(inputs)
        self.assertEqual(result, [])
    
    def test_model_version(self):
        version = self.predictor.get_model_version()
        self.assertIsInstance(version, str)

if __name__ == '__main__':
    unittest.main()
