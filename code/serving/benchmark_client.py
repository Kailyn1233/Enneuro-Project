import time
import threading
import statistics
from typing import List, Dict, Any
from .client import ServingClient
from .logger import get_logger

logger = get_logger(__name__)

class BenchmarkClient:
    def __init__(self, host: str, port: int):
        self.client = ServingClient(host=host, port=port)
        self.results = []
        self.errors = 0
        self.lock = threading.Lock()
    
    def _worker(self, inputs: List[List[float]], num_requests: int):
        for _ in range(num_requests):
            start_time = time.time()
            result = self.client.predict(inputs)
            latency = (time.time() - start_time) * 1000
            
            with self.lock:
                if 'predictions' in result:
                    self.results.append(latency)
                else:
                    self.errors += 1
    
    def run_benchmark(
        self,
        inputs: List[List[float]],
        num_threads: int = 10,
        requests_per_thread: int = 100
    ) -> Dict[str, Any]:
        start_time = time.time()
        threads = []
        
        for _ in range(num_threads):
            thread = threading.Thread(
                target=self._worker,
                args=(inputs, requests_per_thread)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        total_requests = num_threads * requests_per_thread
        success_rate = ((total_requests - self.errors) / total_requests) * 100
        
        if self.results:
            stats = {
                'min_latency_ms': min(self.results),
                'max_latency_ms': max(self.results),
                'avg_latency_ms': statistics.mean(self.results),
                'p50_latency_ms': self._percentile(self.results, 50),
                'p90_latency_ms': self._percentile(self.results, 90),
                'p99_latency_ms': self._percentile(self.results, 99),
                'throughput': total_requests / total_time
            }
        else:
            stats = {}
        
        return {
            'total_requests': total_requests,
            'successful_requests': total_requests - self.errors,
            'failed_requests': self.errors,
            'success_rate': success_rate,
            'total_time_seconds': total_time,
            **stats
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        index = int(len(sorted_vals) * percentile / 100)
        return sorted_vals[min(index, len(sorted_vals) - 1)]

def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Benchmark Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--threads", type=int, default=10, help="Number of concurrent threads")
    parser.add_argument("--requests", type=int, default=100, help="Requests per thread")
    parser.add_argument("--input-size", type=int, default=3, help="Input vector size")
    
    args = parser.parse_args()
    
    inputs = [[float(i) for i in range(args.input_size)]]
    
    benchmark = BenchmarkClient(host=args.host, port=args.port)
    result = benchmark.run_benchmark(
        inputs=inputs,
        num_threads=args.threads,
        requests_per_thread=args.requests
    )
    
    print("Benchmark Results:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
