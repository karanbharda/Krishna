"""
Enhanced Error Handling and Recovery System
Provides intelligent retry logic, circuit breaker pattern, and automatic recovery
"""

import asyncio
import time
import logging
from typing import Callable, Any, Dict, Optional
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from .exceptions import (
    TradingBotError, NetworkError, DataServiceError, 
    AuthenticationError, TradingExecutionError
)

logger = logging.getLogger(__name__)

class CircuitBreakerOpenError(TradingBotError):
    """Raised when circuit breaker is open"""
    pass

class EnhancedErrorHandler:
    """Production-grade error handling with circuit breaker pattern"""
    
    def __init__(self):
        self.circuit_breaker_state = {}
        self.failure_counts = {}
        self.last_failure_time = {}
        self.success_counts = {}
        
        # Circuit breaker configuration
        self.failure_threshold = 5
        self.recovery_timeout = 300  # 5 minutes
        self.half_open_max_calls = 3
        
    def _is_circuit_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open for a service"""
        if service_name not in self.circuit_breaker_state:
            return False
            
        failures = self.failure_counts.get(service_name, 0)
        last_failure = self.last_failure_time.get(service_name, 0)
        
        # Check if we should reset the circuit breaker
        if failures >= self.failure_threshold:
            if time.time() - last_failure > self.recovery_timeout:
                logger.info(f"Attempting to reset circuit breaker for {service_name}")
                self.circuit_breaker_state[service_name] = 'half-open'
                self.success_counts[service_name] = 0
                return False
            return True
            
        return False
    
    def _record_success(self, service_name: str):
        """Record successful operation"""
        if service_name in self.circuit_breaker_state:
            if self.circuit_breaker_state[service_name] == 'half-open':
                self.success_counts[service_name] = self.success_counts.get(service_name, 0) + 1
                if self.success_counts[service_name] >= self.half_open_max_calls:
                    logger.info(f"Circuit breaker closed for {service_name}")
                    del self.circuit_breaker_state[service_name]
                    self.failure_counts[service_name] = 0
            else:
                # Reset failure count on success
                self.failure_counts[service_name] = 0
    
    def _record_failure(self, service_name: str):
        """Record failed operation"""
        self.failure_counts[service_name] = self.failure_counts.get(service_name, 0) + 1
        self.last_failure_time[service_name] = time.time()
        
        if self.failure_counts[service_name] >= self.failure_threshold:
            self.circuit_breaker_state[service_name] = 'open'
            logger.error(f"Circuit breaker opened for {service_name} after {self.failure_counts[service_name]} failures")
    
    def circuit_breaker(self, service_name: str):
        """Circuit breaker decorator"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Check if circuit is open
                if self._is_circuit_open(service_name):
                    if self.circuit_breaker_state.get(service_name) != 'half-open':
                        raise CircuitBreakerOpenError(f"Service {service_name} is unavailable")
                
                try:
                    result = await func(*args, **kwargs)
                    self._record_success(service_name)
                    return result
                except Exception as e:
                    self._record_failure(service_name)
                    raise
            return wrapper
        return decorator
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout, NetworkError))
    )
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with intelligent retry logic"""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except requests.ConnectionError as e:
            logger.warning(f"Connection error in {func.__name__}: {e}")
            await self._handle_connection_error(e)
            raise NetworkError(f"Connection failed: {str(e)}")
        except requests.Timeout as e:
            logger.warning(f"Timeout error in {func.__name__}: {e}")
            await self._handle_timeout_error(e)
            raise NetworkError(f"Request timeout: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            await self._handle_generic_error(e)
            raise
    
    async def _handle_connection_error(self, error):
        """Handle network connectivity issues"""
        logger.warning("Network error detected, implementing backoff...")
        await asyncio.sleep(2)
    
    async def _handle_timeout_error(self, error):
        """Handle timeout errors"""
        logger.warning("Timeout error, implementing backoff...")
        await asyncio.sleep(1)
    
    async def _handle_generic_error(self, error):
        """Handle generic errors"""
        logger.error(f"Generic error handler: {error}")
        await asyncio.sleep(0.5)
    
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get status information for a service"""
        return {
            'service_name': service_name,
            'circuit_state': self.circuit_breaker_state.get(service_name, 'closed'),
            'failure_count': self.failure_counts.get(service_name, 0),
            'success_count': self.success_counts.get(service_name, 0),
            'last_failure_time': self.last_failure_time.get(service_name, 0)
        }
    
    def get_all_service_status(self) -> Dict[str, Dict]:
        """Get status for all monitored services"""
        services = set(list(self.circuit_breaker_state.keys()) + 
                      list(self.failure_counts.keys()))
        return {service: self.get_service_status(service) for service in services}

# Global error handler instance
_error_handler = None

def get_error_handler() -> EnhancedErrorHandler:
    """Get singleton instance of EnhancedErrorHandler"""
    global _error_handler
    if _error_handler is None:
        _error_handler = EnhancedErrorHandler()
    return _error_handler

# Convenience decorators
def with_circuit_breaker(service_name: str):
    """Decorator for circuit breaker protection"""
    return get_error_handler().circuit_breaker(service_name)

async def execute_with_retry(func: Callable, *args, **kwargs) -> Any:
    """Convenience function for retry execution"""
    return await get_error_handler().execute_with_retry(func, *args, **kwargs)

def handle_api_error(func):
    """Decorator for API error handling"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {str(e)}")
            elif e.response.status_code == 429:
                logger.warning("Rate limit hit, backing off...")
                await asyncio.sleep(60)  # Wait 1 minute for rate limit reset
                raise NetworkError(f"Rate limit exceeded: {str(e)}")
            elif e.response.status_code >= 500:
                raise DataServiceError(f"Server error: {str(e)}")
            else:
                raise TradingExecutionError(f"API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    return wrapper