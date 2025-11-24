"""
Centralized Configuration Management System
Replaces hardcoded URLs with environment variables for flexible deployment
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for external services"""
    groq_api_key: str
    groq_base_url: str
    mcp_server_host: str
    mcp_server_port: int
    data_service_url: str
    websocket_port: int
    redis_url: str
    database_url: str
    fyers_base_url: str
    dhan_base_url: str


class EnvironmentManager:
    """Centralized environment configuration management"""

    def __init__(self):
        self.config = self._load_configuration()
        self._validate_configuration()

    def _load_configuration(self) -> ServiceConfig:
        """Load all configuration from environment variables with sensible defaults"""
        return ServiceConfig(
            # AI/ML Services
            groq_api_key=os.getenv('GROQ_API_KEY', ''),
            groq_base_url=os.getenv(
                'GROQ_BASE_URL', 'https://api.groq.com/openai/v1'),

            # MCP Server
            mcp_server_host=os.getenv('MCP_SERVER_HOST', 'localhost'),
            mcp_server_port=int(os.getenv('MCP_SERVER_PORT', '8002')),

            # Data Services
            data_service_url=os.getenv(
                'DATA_SERVICE_URL', 'http://127.0.0.1:8002'),

            # WebSocket
            websocket_port=int(os.getenv('WEBSOCKET_PORT', '8765')),

            # Storage
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379'),
            database_url=os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db'),

            # Trading APIs
            fyers_base_url=os.getenv('FYERS_BASE_URL', 'https://api.fyers.in'),
            dhan_base_url=os.getenv('DHAN_BASE_URL', 'https://api.dhan.co')
        )

    def _validate_configuration(self):
        """Validate configuration and log warnings for missing critical settings"""
        warnings = []

        # Check for missing Groq API key
        if not self.config.groq_api_key:
            warnings.append("GROQ_API_KEY is not set")

        if 'localhost' in self.config.mcp_server_host and os.getenv('ENVIRONMENT') == 'production':
            warnings.append(
                "MCP_SERVER_HOST uses localhost in production environment")

        # Log warnings
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")

        logger.info("Configuration validation completed")

    def get_service_url(self, service_name: str) -> str:
        """Get URL for a specific service"""
        service_urls = {
            'groq': self.config.groq_base_url,
            'mcp': f"http://{self.config.mcp_server_host}:{self.config.mcp_server_port}",
            'data_service': self.config.data_service_url,
            'fyers': self.config.fyers_base_url,
            'dhan': self.config.dhan_base_url,
            'websocket': f"ws://localhost:{self.config.websocket_port}",
            'redis': self.config.redis_url,
            'database': self.config.database_url
        }

        return service_urls.get(service_name, '')

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            'groq_api_key': self.config.groq_api_key,
            'groq_base_url': self.config.groq_base_url,
            'mcp_server_host': self.config.mcp_server_host,
            'mcp_server_port': self.config.mcp_server_port,
            'data_service_url': self.config.data_service_url,
            'websocket_port': self.config.websocket_port,
            'redis_url': self.config.redis_url,
            'database_url': self.config.database_url,
            'fyers_base_url': self.config.fyers_base_url,
            'dhan_base_url': self.config.dhan_base_url
        }

    def is_production_environment(self) -> bool:
        """Check if running in production environment"""
        return os.getenv('ENVIRONMENT', 'development').lower() == 'production'

    def get_log_level(self) -> str:
        """Get configured log level"""
        return os.getenv('LOG_LEVEL', 'INFO').upper()


# Global instance
_env_manager = None


def get_environment_manager() -> EnvironmentManager:
    """Get singleton instance of EnvironmentManager"""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvironmentManager()
    return _env_manager


def get_service_url(service_name: str) -> str:
    """Convenience function to get service URL"""
    return get_environment_manager().get_service_url(service_name)


def get_config() -> Dict[str, Any]:
    """Convenience function to get all configuration"""
    return get_environment_manager().get_all_config()
