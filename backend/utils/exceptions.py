"""
Code Quality: Standardized exception hierarchy
"""


class TradingBotError(Exception):
    """
    Base exception for trading bot errors
    
    All trading bot specific exceptions should inherit from this class
    to provide a consistent error handling interface.
    """
    pass


class ConfigurationError(TradingBotError):
    """
    Configuration-related errors
    
    Raised when there are issues with configuration validation,
    missing required settings, or invalid parameter values.
    """
    pass


class DataServiceError(TradingBotError):
    """
    Data service-related errors
    
    Raised when there are issues with data retrieval, API failures,
    or data quality problems.
    """
    pass


class TradingExecutionError(TradingBotError):
    """
    Trading execution errors
    
    Raised when there are issues with trade execution, order placement,
    or portfolio management operations.
    """
    pass


class ValidationError(TradingBotError):
    """
    Input validation errors
    
    Raised when user input or API parameters fail validation checks.
    """
    pass


class NetworkError(TradingBotError):
    """
    Network and connectivity errors
    
    Raised when there are network connectivity issues, timeouts,
    or service unavailability.
    """
    pass


class AuthenticationError(TradingBotError):
    """
    Authentication and authorization errors
    
    Raised when API credentials are invalid, expired, or insufficient
    permissions for requested operations.
    """
    pass
