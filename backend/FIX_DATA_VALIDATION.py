"""
Fix for Trading System Data Validation Issues

This script addresses the problems identified in the trading system:
1. Rate limiting causing data fetching failures
2. Missing data leading to false buy signals
3. Improper handling of incomplete analysis results
"""

import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_analysis_data(analysis_data: Dict[str, Any]) -> bool:
    """
    Validate that we have sufficient data to make trading decisions.
    
    Args:
        analysis_data: Dictionary containing all analysis results
        
    Returns:
        bool: True if data is sufficient, False otherwise
    """
    # Check if we have basic price data
    if not analysis_data.get('current_price') or analysis_data.get('current_price') <= 0:
        logger.warning("Missing or invalid current price data")
        return False
    
    # Check if we have technical analysis data
    technical_analysis = analysis_data.get('technical_analysis', {})
    if not technical_analysis:
        logger.warning("Missing technical analysis data")
        return False
    
    # Check if we have at least some key technical indicators
    required_technical_indicators = ['rsi', 'sma_20', 'sma_50']
    missing_indicators = [indicator for indicator in required_technical_indicators 
                         if technical_analysis.get(indicator) is None]
    
    if len(missing_indicators) == len(required_technical_indicators):
        logger.warning("Missing key technical indicators")
        return False
    
    # Check if we have sentiment analysis data
    sentiment_analysis = analysis_data.get('sentiment_analysis', {})
    if not sentiment_analysis:
        logger.warning("Missing sentiment analysis data")
        return False
    
    # Check if we have ML analysis data
    ml_analysis = analysis_data.get('ml_analysis', {})
    if not ml_analysis:
        logger.warning("Missing ML analysis data")
        return False
    
    # Check if we have fundamental data
    fundamental_data = analysis_data.get('fundamental_data', {})
    if not fundamental_data:
        logger.warning("Missing fundamental data")
        return False
    
    logger.info("All required data present for analysis")
    return True

def create_safe_default_analysis(ticker: str) -> Dict[str, Any]:
    """
    Create a safe default analysis when data is missing.
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dict containing safe default values
    """
    return {
        'symbol': ticker,
        'current_price': 0,
        'technical_analysis': {
            'rsi': 50,
            'sma_20': 0,
            'sma_50': 0,
            'macd': 0,
            'signal': 0
        },
        'sentiment_analysis': {
            'overall_sentiment': 0,
            'confidence': 0
        },
        'ml_analysis': {
            'prediction_direction': 0,
            'confidence': 0.5,
            'success': False
        },
        'fundamental_data': {
            'pe_ratio': 0,
            'pb_ratio': 0,
            'dividend_yield': 0
        },
        'data_quality': 'insufficient'
    }

def should_skip_analysis_due_to_errors(ticker: str, error_messages: list) -> bool:
    """
    Determine if we should skip analysis for a stock due to errors.
    
    Args:
        ticker: Stock symbol
        error_messages: List of error messages encountered
        
    Returns:
        bool: True if we should skip analysis, False otherwise
    """
    # Skip if we have rate limiting errors
    rate_limit_errors = [msg for msg in error_messages if 'rate limit' in msg.lower()]
    if rate_limit_errors:
        logger.warning(f"Skipping {ticker} due to rate limiting: {len(rate_limit_errors)} errors")
        return True
    
    # Skip if we have "no price data found" errors
    no_data_errors = [msg for msg in error_messages if 'no price data found' in msg.lower()]
    if no_data_errors:
        logger.warning(f"Skipping {ticker} due to missing price data")
        return True
    
    # Skip if we have critical data source failures
    critical_failures = [msg for msg in error_messages if 'critical' in msg.lower()]
    if len(critical_failures) >= 2:  # Multiple critical failures
        logger.warning(f"Skipping {ticker} due to multiple critical failures")
        return True
    
    return False

def enhance_error_handling_in_stock_analysis():
    """
    Enhance error handling in the stock analysis process.
    This would be implemented in the testindia.py file.
    """
    enhancement_notes = """
    To fix the issues in the trading system:
    
    1. In testindia.py analyze_stock method, add proper error handling:
       - Check for rate limiting before proceeding
       - Return safe defaults when data is missing
       - Skip analysis when critical data is unavailable
    
    2. In professional_buy_logic.py, enhance data validation:
       - Add comprehensive data quality checks
       - Return HOLD decisions when data is insufficient
       - Log detailed reasons for skipped analyses
    
    3. In the main trading loop, implement rate limiting:
       - Add delays between API calls
       - Implement exponential backoff for retries
       - Cache results to reduce API calls
    """
    
    logger.info("Enhancement notes for trading system:")
    logger.info(enhancement_notes)

def fix_ml_model_health_check():
    """
    Fix the ML model health check to properly handle missing data.
    """
    fix_notes = """
    To fix ML model health issues:
    
    1. In ml_interface.py, add data validation:
       - Check if sufficient training data is available
       - Return appropriate status when models can't be trained
       - Implement fallback mechanisms for critical models
    
    2. In ensemble_optimizer.py, handle empty model predictions:
       - Return neutral predictions when models fail
       - Log detailed error information
       - Prevent critical health status when data is temporarily unavailable
    """
    
    logger.info("Fix notes for ML model health:")
    logger.info(fix_notes)

if __name__ == "__main__":
    logger.info("Trading System Data Validation Fix")
    logger.info("================================")
    
    # Demonstrate the validation function
    test_data = {
        'current_price': 100.0,
        'technical_analysis': {'rsi': 60, 'sma_20': 95, 'sma_50': 90},
        'sentiment_analysis': {'overall_sentiment': 0.2, 'confidence': 0.7},
        'ml_analysis': {'prediction_direction': 0.05, 'confidence': 0.8, 'success': True},
        'fundamental_data': {'pe_ratio': 15, 'pb_ratio': 2.0, 'dividend_yield': 0.02}
    }
    
    is_valid = validate_analysis_data(test_data)
    logger.info(f"Test data validation result: {'Valid' if is_valid else 'Invalid'}")
    
    # Test with missing data
    invalid_data = {
        'current_price': 0,  # Invalid price
        'technical_analysis': {},  # Missing technical data
        'sentiment_analysis': {'overall_sentiment': 0},  # Missing confidence
        'ml_analysis': {'success': False},  # Failed ML analysis
        'fundamental_data': {}  # Missing fundamental data
    }
    
    is_invalid = validate_analysis_data(invalid_data)
    logger.info(f"Invalid data validation result: {'Valid' if is_invalid else 'Invalid'}")
    
    # Show enhancement recommendations
    enhance_error_handling_in_stock_analysis()
    fix_ml_model_health_check()
    
    logger.info("================================")
    logger.info("Fix implementation should be added to the respective modules")