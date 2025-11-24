"""
Performance Monitor
Provides performance monitoring and metrics collection capabilities
"""

import time
import logging
from typing import Dict, Any, List
from collections import defaultdict, deque
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Monitor application performance and collect metrics
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.request_times = deque(maxlen=max_history)
        self.endpoint_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'errors': 0,
            'min_time': float('inf'),
            'max_time': 0.0
        })
        self.start_time = time.time()
        
    def record_request(self, endpoint: str, duration: float, success: bool = True):
        """
        Record a request execution
        
        Args:
            endpoint: Endpoint name
            duration: Request duration in seconds
            success: Whether request was successful
        """
        try:
            # Record overall timing
            self.request_times.append({
                'endpoint': endpoint,
                'duration': duration,
                'success': success,
                'timestamp': time.time()
            })
            
            # Update endpoint stats
            stats = self.endpoint_stats[endpoint]
            stats['count'] += 1
            stats['total_time'] += duration
            stats['min_time'] = min(stats['min_time'], duration)
            stats['max_time'] = max(stats['max_time'], duration)
            
            if not success:
                stats['errors'] += 1
                
        except Exception as e:
            logger.error(f"Error recording request metrics: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance statistics
        """
        try:
            # Calculate overall stats
            total_requests = len(self.request_times)
            successful_requests = sum(1 for r in self.request_times if r['success'])
            failed_requests = total_requests - successful_requests
            
            if total_requests > 0:
                avg_response_time = sum(r['duration'] for r in self.request_times) / total_requests
                min_response_time = min(r['duration'] for r in self.request_times)
                max_response_time = max(r['duration'] for r in self.request_times)
            else:
                avg_response_time = 0
                min_response_time = 0
                max_response_time = 0
            
            # Calculate uptime
            uptime = time.time() - self.start_time
            
            # Format endpoint stats
            endpoint_stats = {}
            for endpoint, stats in self.endpoint_stats.items():
                if stats['count'] > 0:
                    endpoint_stats[endpoint] = {
                        'count': stats['count'],
                        'avg_time': stats['total_time'] / stats['count'],
                        'min_time': stats['min_time'],
                        'max_time': stats['max_time'],
                        'error_rate': stats['errors'] / stats['count'] if stats['count'] > 0 else 0,
                        'success_rate': (stats['count'] - stats['errors']) / stats['count'] if stats['count'] > 0 else 0
                    }
            
            return {
                'status': 'active',
                'uptime_seconds': uptime,
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
                'avg_response_time': avg_response_time,
                'min_response_time': min_response_time,
                'max_response_time': max_response_time,
                'endpoint_stats': endpoint_stats
            }
            
        except Exception as e:
            logger.error(f"Error generating performance stats: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

# Global instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """
    Get the global performance monitor instance
    
    Returns:
        PerformanceMonitor instance
    """
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor