import time
import threading
from typing import Dict, List, Any
from collections import defaultdict
from .logger import get_logger

logger = get_logger(__name__)

class MetricsCollector:
    def __init__(self):
        self.counter = defaultdict(int)
        self.histogram = defaultdict(list)
        self.gauge = {}
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def increment_counter(self, name: str, value: int = 1):
        with self.lock:
            self.counter[name] += value
            logger.debug(f"Counter {name} incremented by {value}")
    
    def record_histogram(self, name: str, value: float):
        with self.lock:
            self.histogram[name].append(value)
            if len(self.histogram[name]) > 10000:
                self.histogram[name] = self.histogram[name][-5000:]
            logger.debug(f"Histogram {name} recorded: {value}")
    
    def set_gauge(self, name: str, value: float):
        with self.lock:
            self.gauge[name] = value
            logger.debug(f"Gauge {name} set to {value}")
    
    def get_counter(self, name: str) -> int:
        with self.lock:
            return self.counter.get(name, 0)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        with self.lock:
            values = self.histogram.get(name, [])
            if not values:
                return {}
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'p50': self._percentile(values, 50),
                'p90': self._percentile(values, 90),
                'p99': self._percentile(values, 99)
            }
    
    def get_gauge(self, name: str) -> float:
        with self.lock:
            return self.gauge.get(name, 0.0)
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        index = int(len(sorted_vals) * percentile / 100)
        return sorted_vals[min(index, len(sorted_vals) - 1)]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        result = {
            'counters': dict(self.counter),
            'gauges': dict(self.gauge),
            'histograms': {}
        }
        for name in self.histogram:
            result['histograms'][name] = self.get_histogram_stats(name)
        return result
    
    def reset(self):
        with self.lock:
            self.counter.clear()
            self.histogram.clear()
            self.gauge.clear()
            logger.info("Metrics collector reset")

metrics = MetricsCollector()

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        start_time = time.time()
        await self.app(scope, receive, send)
        duration = (time.time() - start_time) * 1000
        metrics.record_histogram('request_latency_ms', duration)
        metrics.increment_counter('total_requests')
