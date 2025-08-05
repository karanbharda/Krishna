"""
Code Quality: Input validation utilities
"""

import logging
import re
from typing import Dict, Any, List

from .exceptions import ValidationError, ConfigurationError

logger = logging.getLogger(__name__)

# Constants
CHAT_MESSAGE_MAX_LENGTH = 1000


class ConfigValidator:
    """
    Configuration validation utility
    
    Provides comprehensive validation for trading bot configuration
    with detailed error reporting and warnings.
    """
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize configuration
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ConfigurationError: If validation fails
        """
        errors = []
        warnings = []
        
        # Validate starting balance
        starting_balance = config.get("starting_balance", 0)
        if starting_balance <= 0:
            errors.append(f"starting_balance must be positive (got: {starting_balance})")

        # Validate risk parameters
        stop_loss = config.get("stop_loss_pct", 0)
        if not 0 < stop_loss <= 1:
            errors.append(f"stop_loss_pct must be between 0 and 1 (got: {stop_loss})")

        max_capital = config.get("max_capital_per_trade", 0)
        if not 0 < max_capital <= 1:
            errors.append(f"max_capital_per_trade must be between 0 and 1 (got: {max_capital})")
        
        # Validate sleep interval
        sleep_interval = config.get("sleep_interval", 30)
        if sleep_interval < 10:
            warnings.append("sleep_interval less than 10 seconds may cause rate limiting")
        
        # Validate tickers
        tickers = config.get("tickers", [])
        if not isinstance(tickers, list):
            errors.append("tickers must be a list")
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
        
        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning}")
        
        return config


def validate_chat_input(message: str) -> str:
    """
    Validate and sanitize chat input
    
    Args:
        message: User input message to validate
        
    Returns:
        Sanitized message string
        
    Raises:
        ValidationError: If validation fails
    """
    logger.debug(f"Validating chat input: length={len(message) if message else 0}")
    
    if not message or not isinstance(message, str):
        logger.warning("Chat input validation failed: empty or non-string message")
        raise ValidationError("Message must be a non-empty string")
    
    # Security: Sanitize input to prevent injection attacks
    message = message.strip()
    if len(message) > CHAT_MESSAGE_MAX_LENGTH:
        logger.warning(f"Chat input validation failed: message too long ({len(message)} chars)")
        raise ValidationError(f"Message too long (max {CHAT_MESSAGE_MAX_LENGTH} characters)")
    
    # Remove potentially dangerous characters
    original_length = len(message)
    message = re.sub(r'[<>"\']', '', message)
    if len(message) != original_length:
        logger.info(f"Sanitized chat input: removed {original_length - len(message)} dangerous characters")
    
    logger.debug("Chat input validation successful")
    return message
