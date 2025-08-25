"""
Configuration module for trading bot
Provides centralized configuration management
"""

from .environment_manager import (
    EnvironmentManager,
    get_environment_manager,
    get_service_url,
    get_config
)

__all__ = [
    'EnvironmentManager',
    'get_environment_manager', 
    'get_service_url',
    'get_config'
]