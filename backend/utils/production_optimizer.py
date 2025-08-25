"""
Phase 4: Production Optimization System
Implements memory management, caching, and performance tuning for institutional-grade performance
"""

import gc
import os
import sys
import time
import psutil
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from collections import defaultdict, deque
import logging
import json
import pickle
from dataclasses import dataclass, asdict
import weakref

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    cpu_usage: float
    memory_usage: float
    memory_available: float
    cache_hit_rate: float
    request_count: int
    avg_response_time: float
    error_rate: float
    timestamp: str


class MemoryManager:
    """Advanced memory management for production optimization"""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.memory_threshold = 0.8  # 80% threshold
        self.cleanup_callbacks = []
        self.memory_monitor_thread = None
        self.monitoring_active = False
        
        logger.info(f"✅ Memory Manager initialized (limit: {max_memory_mb}MB)")
    
    def start_monitoring(self):
        """Start memory monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.memory_monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
            self.memory_monitor_thread.start()
            logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.memory_monitor_thread:
            self.memory_monitor_thread.join(timeout=1)
    
    def _monitor_memory(self):
        """Monitor memory usage and trigger cleanup when needed"""
        while self.monitoring_active:
            try:
                current_memory = self.get_memory_usage()
                memory_percent = current_memory / self.max_memory_mb
                
                if memory_percent > self.memory_threshold:
                    logger.warning(f"High memory usage: {current_memory:.1f}MB ({memory_percent:.1%})")
                    self.cleanup_memory()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(60)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.0
    
    def register_cleanup_callback(self, callback: Callable):
        """Register callback for memory cleanup"""
        self.cleanup_callbacks.append(callback)
    
    def cleanup_memory(self):
        """Perform memory cleanup"""
        try:
            initial_memory = self.get_memory_usage()
            
            # Run registered cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in cleanup callback: {e}")
            
            # Force garbage collection
            gc.collect()
            
            final_memory = self.get_memory_usage()
            freed_memory = initial_memory - final_memory
            
            logger.info(f"Memory cleanup completed: {freed_memory:.1f}MB freed")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")


class IntelligentCache:
    """Advanced caching system with TTL, LRU, and intelligent eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.access_counts = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()
        
        logger.info(f"✅ Intelligent Cache initialized (size: {max_size}, TTL: {default_ttl}s)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            current_time = time.time()
            
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            # Check TTL
            creation_time = self.creation_times.get(key, 0)
            if current_time - creation_time > self.default_ttl:
                self._remove_key(key)
                self.miss_count += 1
                return None
            
            # Update access tracking
            self.access_times[key] = current_time
            self.access_counts[key] += 1
            self.hit_count += 1
            
            return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set item in cache"""
        with self._lock:
            current_time = time.time()
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_least_valuable()
            
            self.cache[key] = value
            self.creation_times[key] = current_time
            self.access_times[key] = current_time
            self.access_counts[key] = 1
    
    def _evict_least_valuable(self):
        """Evict least valuable items using intelligent scoring"""
        if not self.cache:
            return
        
        current_time = time.time()
        scores = {}
        
        for key in self.cache:
            creation_time = self.creation_times.get(key, current_time)
            last_access = self.access_times.get(key, creation_time)
            access_count = self.access_counts.get(key, 1)
            
            # Calculate value score (higher = more valuable)
            age = current_time - creation_time
            recency = current_time - last_access
            frequency = access_count
            
            # Combine factors (lower score = less valuable)
            score = frequency / (1 + recency) / (1 + age)
            scores[key] = score
        
        # Remove least valuable item
        least_valuable = min(scores, key=scores.get)
        self._remove_key(least_valuable)
    
    def _remove_key(self, key: str):
        """Remove key from all tracking structures"""
        self.cache.pop(key, None)
        self.creation_times.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'utilization': len(self.cache) / self.max_size
        }
    
    def clear(self):
        """Clear entire cache"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()
            self.access_counts.clear()


class PerformanceProfiler:
    """Performance profiling and optimization recommendations"""
    
    def __init__(self):
        self.execution_times = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.memory_usage = deque(maxlen=100)
        
    def profile_function(self, func_name: str = None):
        """Decorator for profiling function execution"""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_current_memory()
                
                try:
                    result = func(*args, **kwargs)
                    self.call_counts[name] += 1
                    return result
                except Exception as e:
                    self.error_counts[name] += 1
                    raise
                finally:
                    end_time = time.time()
                    end_memory = self._get_current_memory()
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    self.execution_times[name].append(execution_time)
                    self.memory_usage.append({
                        'function': name,
                        'memory_delta': memory_delta,
                        'timestamp': time.time()
                    })
                    
                    # Keep only recent timing data
                    if len(self.execution_times[name]) > 100:
                        self.execution_times[name] = self.execution_times[name][-100:]
            
            return wrapper
        return decorator
    
    def _get_current_memory(self) -> float:
        """Get current memory usage"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        report = {
            'function_stats': {},
            'slowest_functions': [],
            'most_called_functions': [],
            'error_prone_functions': [],
            'memory_intensive_functions': []
        }
        
        # Analyze function performance
        for func_name, times in self.execution_times.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                call_count = self.call_counts[func_name]
                error_count = self.error_counts[func_name]
                error_rate = error_count / call_count if call_count > 0 else 0
                
                report['function_stats'][func_name] = {
                    'avg_time': avg_time,
                    'max_time': max_time,
                    'call_count': call_count,
                    'error_count': error_count,
                    'error_rate': error_rate
                }
        
        # Identify performance issues
        sorted_by_time = sorted(
            report['function_stats'].items(),
            key=lambda x: x[1]['avg_time'],
            reverse=True
        )
        report['slowest_functions'] = sorted_by_time[:5]
        
        sorted_by_calls = sorted(
            report['function_stats'].items(),
            key=lambda x: x[1]['call_count'],
            reverse=True
        )
        report['most_called_functions'] = sorted_by_calls[:5]
        
        # Analyze memory usage
        memory_by_function = defaultdict(list)
        for entry in self.memory_usage:
            memory_by_function[entry['function']].append(entry['memory_delta'])
        
        for func_name, deltas in memory_by_function.items():
            avg_delta = sum(deltas) / len(deltas)
            if avg_delta > 1.0:  # More than 1MB average
                report['memory_intensive_functions'].append((func_name, avg_delta))
        
        return report


class ProductionOptimizer:
    """Main production optimization system"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.cache = IntelligentCache()
        self.profiler = PerformanceProfiler()
        
        # Performance monitoring
        self.metrics_history = deque(maxlen=1000)
        self.optimization_suggestions = []
        
        # Configuration
        self.optimization_enabled = True
        self.auto_cleanup_enabled = True
        
        # Register memory cleanup
        self.memory_manager.register_cleanup_callback(self._cleanup_caches)
        
        logger.info("✅ Production Optimizer initialized")
    
    def start_optimization(self):
        """Start production optimization"""
        try:
            self.memory_manager.start_monitoring()
            self._start_performance_monitoring()
            logger.info("Production optimization started")
        except Exception as e:
            logger.error(f"Error starting optimization: {e}")
    
    def stop_optimization(self):
        """Stop production optimization"""
        try:
            self.memory_manager.stop_monitoring()
            logger.info("Production optimization stopped")
        except Exception as e:
            logger.error(f"Error stopping optimization: {e}")
    
    def _start_performance_monitoring(self):
        """Start performance metrics collection"""
        def monitor_performance():
            while self.optimization_enabled:
                try:
                    metrics = self._collect_performance_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Generate optimization suggestions
                    self._analyze_performance(metrics)
                    
                    time.sleep(60)  # Collect metrics every minute
                    
                except Exception as e:
                    logger.error(f"Error in performance monitoring: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
        monitor_thread.start()
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.used / 1024 / 1024  # MB
            memory_available = memory.available / 1024 / 1024  # MB
            
            # Cache metrics
            cache_stats = self.cache.get_stats()
            cache_hit_rate = cache_stats['hit_rate']
            
            # Application metrics (simplified)
            request_count = cache_stats['hit_count'] + cache_stats['miss_count']
            avg_response_time = self._calculate_avg_response_time()
            error_rate = self._calculate_error_rate()
            
            return PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                memory_available=memory_available,
                cache_hit_rate=cache_hit_rate,
                request_count=request_count,
                avg_response_time=avg_response_time,
                error_rate=error_rate,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, datetime.now().isoformat())
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        try:
            all_times = []
            for func_times in self.profiler.execution_times.values():
                all_times.extend(func_times[-10:])  # Recent 10 calls
            
            return sum(all_times) / len(all_times) if all_times else 0.0
        except:
            return 0.0
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate"""
        try:
            total_calls = sum(self.profiler.call_counts.values())
            total_errors = sum(self.profiler.error_counts.values())
            
            return total_errors / total_calls if total_calls > 0 else 0.0
        except:
            return 0.0
    
    def _analyze_performance(self, metrics: PerformanceMetrics):
        """Analyze performance and generate suggestions"""
        try:
            suggestions = []
            
            # Memory analysis
            if metrics.memory_usage > 400:  # More than 400MB
                suggestions.append("High memory usage detected - consider reducing cache size")
            
            # CPU analysis
            if metrics.cpu_usage > 80:
                suggestions.append("High CPU usage - consider optimizing slow functions")
            
            # Cache analysis
            if metrics.cache_hit_rate < 0.7:
                suggestions.append("Low cache hit rate - review caching strategy")
            
            # Response time analysis
            if metrics.avg_response_time > 1.0:  # More than 1 second
                suggestions.append("Slow response times - profile and optimize bottlenecks")
            
            # Error rate analysis
            if metrics.error_rate > 0.05:  # More than 5% errors
                suggestions.append("High error rate - investigate error-prone functions")
            
            # Update suggestions
            self.optimization_suggestions = suggestions
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
    
    def _cleanup_caches(self):
        """Cleanup caches to free memory"""
        try:
            # Clear old cache entries
            self.cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Cache cleanup completed")
            
        except Exception as e:
            logger.error(f"Error cleaning up caches: {e}")
    
    def get_optimization_report(self) -> Dict:
        """Get comprehensive optimization report"""
        try:
            recent_metrics = list(self.metrics_history)[-10:] if self.metrics_history else []
            
            report = {
                'system_status': {
                    'memory_usage_mb': self.memory_manager.get_memory_usage(),
                    'memory_limit_mb': self.memory_manager.max_memory_mb,
                    'cache_stats': self.cache.get_stats(),
                    'optimization_enabled': self.optimization_enabled
                },
                'performance_metrics': {
                    'current': asdict(recent_metrics[-1]) if recent_metrics else {},
                    'average_last_10': self._calculate_average_metrics(recent_metrics)
                },
                'performance_analysis': self.profiler.get_performance_report(),
                'optimization_suggestions': self.optimization_suggestions,
                'timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating optimization report: {e}")
            return {}
    
    def _calculate_average_metrics(self, metrics_list: List[PerformanceMetrics]) -> Dict:
        """Calculate average of performance metrics"""
        if not metrics_list:
            return {}
        
        try:
            avg_metrics = {
                'cpu_usage': sum(m.cpu_usage for m in metrics_list) / len(metrics_list),
                'memory_usage': sum(m.memory_usage for m in metrics_list) / len(metrics_list),
                'cache_hit_rate': sum(m.cache_hit_rate for m in metrics_list) / len(metrics_list),
                'avg_response_time': sum(m.avg_response_time for m in metrics_list) / len(metrics_list),
                'error_rate': sum(m.error_rate for m in metrics_list) / len(metrics_list)
            }
            
            return avg_metrics
            
        except Exception as e:
            logger.error(f"Error calculating average metrics: {e}")
            return {}
    
    def optimize_now(self):
        """Trigger immediate optimization"""
        try:
            logger.info("Starting immediate optimization...")
            
            # Memory cleanup
            self.memory_manager.cleanup_memory()
            
            # Cache optimization
            if self.cache.get_stats()['hit_rate'] < 0.5:
                self.cache.clear()  # Reset cache if hit rate is poor
            
            # Garbage collection
            gc.collect()
            
            logger.info("Immediate optimization completed")
            
        except Exception as e:
            logger.error(f"Error during immediate optimization: {e}")
    
    def cached_function(self, ttl: int = 300):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
                
                # Try to get from cache
                result = self.cache.get(key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.cache.set(key, result, ttl)
                
                return result
            
            return wrapper
        return decorator


# Global instance
_production_optimizer = None

def get_production_optimizer() -> ProductionOptimizer:
    """Get global production optimizer instance"""
    global _production_optimizer
    if _production_optimizer is None:
        _production_optimizer = ProductionOptimizer()
    return _production_optimizer


# Convenience decorators
def optimize_performance(func):
    """Decorator to add performance optimization to functions"""
    optimizer = get_production_optimizer()
    profiled_func = optimizer.profiler.profile_function()(func)
    return profiled_func


def cache_result(ttl: int = 300):
    """Decorator to cache function results"""
    optimizer = get_production_optimizer()
    return optimizer.cached_function(ttl)