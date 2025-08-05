"""
Backend Utilities - 100% Production Ready
========================================

Modular utility components for enterprise-grade trading bot backend.
Provides standardized error handling, input validation, and performance monitoring.

Components:
- exceptions: Standardized exception hierarchy
- validators: Input validation and configuration validation
- monitoring: Performance monitoring and metrics collection

All components are fully type-annotated, tested, and production-ready.
"""

from .validators import ConfigValidator, validate_chat_input
from .exceptions import (
    TradingBotError,
    ConfigurationError,
    DataServiceError,
    TradingExecutionError,
    ValidationError,
    NetworkError,
    AuthenticationError
)
from .monitoring import PerformanceMonitor

__all__ = [
    'ConfigValidator',
    'validate_chat_input',
    'TradingBotError',
    'ConfigurationError',
    'DataServiceError',
    'TradingExecutionError',
    'ValidationError',
    'NetworkError',
    'AuthenticationError',
    'PerformanceMonitor'
]
