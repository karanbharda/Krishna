"""
Code Quality: Performance monitoring utilities
"""

import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Performance monitoring and metrics collection
    
    Tracks request counts, response times, error rates, and system uptime
    with automatic memory management to prevent bloat.
    """
    
    def __init__(self) -> None:
        """Initialize performance monitor"""
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.start_time = datetime.now()
    
    def record_request(self, response_time: float, success: bool = True) -> None:
        """
        Record request metrics with error handling
        
        Args:
            response_time: Request processing time in seconds
            success: Whether the request was successful
        """
        try:
            if not isinstance(response_time, (int, float)) or response_time < 0:
                logger.warning(f"Invalid response time: {response_time}")
                return
            
            self.request_count += 1
            self.response_times.append(response_time)
            if not success:
                self.error_count += 1

            # Priority 2: Enhanced memory management with aggressive cleanup
            max_response_times = 1000
            if len(self.response_times) > max_response_times:
                # Keep only the most recent entries and clear extra buffer
                self.response_times = self.response_times[-max_response_times:]

            # Additional cleanup for extreme scenarios
            if len(self.response_times) > max_response_times * 1.2:  # 20% buffer exceeded
                logger.warning("Performance monitor memory cleanup triggered")
                self.response_times = self.response_times[-int(max_response_times * 0.8):]  # Keep 80%
        except Exception as e:
            logger.error(f"Error recording request metrics: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics with error handling
        
        Returns:
            Dictionary containing performance metrics
        """
        try:
            if not self.response_times:
                return {"status": "no_data"}
            
            avg_response_time = sum(self.response_times) / len(self.response_times)
            uptime = (datetime.now() - self.start_time).total_seconds()
            error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
            
            return {
                "total_requests": self.request_count,
                "error_count": self.error_count,
                "error_rate_percent": round(error_rate, 2),
                "avg_response_time_ms": round(avg_response_time * 1000, 2),
                "uptime_seconds": round(uptime, 2),
                "requests_per_minute": round(self.request_count / (uptime / 60), 2) if uptime > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return {
                "status": "error",
                "error": str(e),
                "total_requests": getattr(self, 'request_count', 0),
                "error_count": getattr(self, 'error_count', 0)
            }
    
    def reset_stats(self) -> None:
        """Reset all performance statistics"""
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.start_time = datetime.now()
        logger.info("Performance statistics reset")
