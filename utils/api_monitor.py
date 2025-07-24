"""
📁 api_monitor.py
🎯 Purpose: Track API call latency and store metrics
"""

import time
from functools import wraps
from typing import Callable, Any, Dict, List
from datetime import datetime

class APIMonitor:
    def __init__(self):
        self.api_metrics = {
            'STT': {'count': 0, 'total_time': 0, 'last_latency': 0},
            'TTS': {'count': 0, 'total_time': 0, 'last_latency': 0},
            'LLM': {'count': 0, 'total_time': 0, 'last_latency': 0},
            'RAG': {'count': 0, 'total_time': 0, 'last_latency': 0},
            'LLM+LPU': {'count': 0, 'total_time': 0, 'last_latency': 0},
        }
        self.total_latency = 0
        self.history: List[Dict] = []
        
    def track(self, service_name: str):
        """Decorator factory to track specific service"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                elapsed_ms = (end_time - start_time) * 1000
                self._update_metrics(service_name, elapsed_ms)
                self._print_latency(service_name, elapsed_ms)
                
                return result
            return wrapper
        return decorator
    
    def _update_metrics(self, service_name: str, latency_ms: float):
        """Update metrics for a service"""
        if service_name in self.api_metrics:
            self.api_metrics[service_name]['count'] += 1
            self.api_metrics[service_name]['total_time'] += latency_ms
            self.api_metrics[service_name]['last_latency'] = latency_ms
            self.total_latency += latency_ms
            
            # Store in history
            self.history.append({
                'timestamp': datetime.now(),
                'service': service_name,
                'latency': latency_ms
            })
    
    def _print_latency(self, service_name: str, latency_ms: float):
        """Print latency to console with color coding"""
        color_code = ""
        if latency_ms < 500:
            color_code = "\033[92m"  # Green
        elif latency_ms < 1000:
            color_code = "\033[93m"  # Yellow
        else:
            color_code = "\033[91m"  # Red
        
        reset_code = "\033[0m"
        print(f"{color_code}[{service_name} Latency] {latency_ms:.2f}ms{reset_code}")
    
    def get_metrics(self) -> Dict:
        """Get current metrics including total latency"""
        return {
            'services': self.api_metrics,
            'total_latency': self.total_latency,
            'average_total_per_interaction': self.total_latency / max(1, len(self.history))
        }

# Singleton instance
monitor = APIMonitor()