"""
Test script to verify the changes to professional buy logic
"""

import logging
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.professional_buy_logic import ProfessionalBuyLogic, StockMetrics, MarketContext, MarketTrend
from core.professional_buy_config import ProfessionalBuyConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_strong_buy_signal():
    """Test a strong buy signal that should pass"""
    logger.info("=== Testing Strong Buy Signal ===")
    
    config = ProfessionalBuyConfig.get_config()
    buy_logic = ProfessionalBuyLogic(config)
    
    # Create strong stock metrics
    stock_metrics = StockMetrics(
        current_price=100.0,
        entry_price=100.0,
        quantity=100,
        volatility=0.02,
        atr=2.0,
        rsi=30,  # Oversold
        macd=0.5,
        macd_signal=0.2,
        sma_20=95.0,
        sma_50=90.0,
        sma_200=85.0,
        support_level=98.0,
        resistance_level=105.0,
        volume_ratio=1.5,
        price_to_book=1.2,
        price_to_earnings=15.0,
        earnings_growth=0.1,
        return_on_equity=0.15,
        free_cash_flow_yield=0.08,
        debt_to_equity=0.3,
        dividend_yield=0.02,
        payout_ratio=0.3,
        earnings_quality=0.8,
        insider_ownership=0.1,
        sector_pe=18.0
    )
    
    # Create bullish market context
    market_context = MarketContext(
        trend=MarketTrend.UPTREND,
        trend_strength=0.7,
        volatility_regime="normal",
        market_stress=0.2,
        sector_performance=0.05,
        volume_profile=0.6
    )
    
    # Strong technical analysis
    technical_analysis = {
        "rsi": 30,
        "macd": 0.5,
        "macd_signal": 0.2,
        "sma_20": 95.0,
        "sma_50": 90.0,
        "sma_200": 85.0,
        "support_level": 98.0,
        "resistance_level": 105.0,
        "volume_ratio": 1.5
    }
    
    # Strong sentiment analysis
    sentiment_analysis = {
        "overall_sentiment": 0.6,
        "sentiment_momentum": 0.4,
        "source_diversity": 6
    }
    
    # Strong ML analysis
    ml_analysis = {
        "prediction_direction": 0.08,
        "confidence": 0.85,
        "success": True
    }
    
    # Evaluate buy decision
    decision = buy_logic.evaluate_buy_decision(
        ticker="TEST.NS",
        stock_metrics=stock_metrics,
        market_context=market_context,
        technical_analysis=technical_analysis,
        sentiment_analysis=sentiment_analysis,
        ml_analysis=ml_analysis,
        portfolio_context={}
    )
    
    logger.info(f"Should Buy: {decision.should_buy}")
    logger.info(f"Confidence: {decision.confidence:.3f}")
    logger.info(f"Buy Percentage: {decision.buy_percentage:.3f}")
    logger.info(f"Reasoning: {decision.reasoning}")
    
    return decision.should_buy

def test_weak_buy_signal():
    """Test a weak buy signal that should be rejected"""
    logger.info("=== Testing Weak Buy Signal ===")
    
    config = ProfessionalBuyConfig.get_config()
    buy_logic = ProfessionalBuyLogic(config)
    
    # Create weak stock metrics
    stock_metrics = StockMetrics(
        current_price=100.0,
        entry_price=100.0,
        quantity=100,
        volatility=0.02,
        atr=2.0,
        rsi=50,  # Neutral
        macd=0.1,
        macd_signal=0.05,
        sma_20=99.0,
        sma_50=99.5,
        sma_200=100.0,
        support_level=98.0,
        resistance_level=102.0,
        volume_ratio=0.8,
        price_to_book=1.5,
        price_to_earnings=20.0,
        earnings_growth=0.05,
        return_on_equity=0.1,
        free_cash_flow_yield=0.03,
        debt_to_equity=0.5,
        dividend_yield=0.01,
        payout_ratio=0.4,
        earnings_quality=0.6,
        insider_ownership=0.05,
        sector_pe=20.0
    )
    
    # Create sideways market context
    market_context = MarketContext(
        trend=MarketTrend.SIDEWAYS,
        trend_strength=0.3,
        volatility_regime="normal",
        market_stress=0.5,
        sector_performance=0.0,
        volume_profile=0.5
    )
    
    # Weak technical analysis
    technical_analysis = {
        "rsi": 50,
        "macd": 0.1,
        "macd_signal": 0.05,
        "sma_20": 99.0,
        "sma_50": 99.5,
        "sma_200": 100.0,
        "support_level": 98.0,
        "resistance_level": 102.0,
        "volume_ratio": 0.8
    }
    
    # Weak sentiment analysis
    sentiment_analysis = {
        "overall_sentiment": 0.2,
        "sentiment_momentum": 0.1,
        "source_diversity": 2
    }
    
    # Weak ML analysis
    ml_analysis = {
        "prediction_direction": 0.02,
        "confidence": 0.6,
        "success": True
    }
    
    # Evaluate buy decision
    decision = buy_logic.evaluate_buy_decision(
        ticker="TEST.NS",
        stock_metrics=stock_metrics,
        market_context=market_context,
        technical_analysis=technical_analysis,
        sentiment_analysis=sentiment_analysis,
        ml_analysis=ml_analysis,
        portfolio_context={}
    )
    
    logger.info(f"Should Buy: {decision.should_buy}")
    logger.info(f"Confidence: {decision.confidence:.3f}")
    logger.info(f"Buy Percentage: {decision.buy_percentage:.3f}")
    logger.info(f"Reasoning: {decision.reasoning}")
    
    return decision.should_buy

if __name__ == "__main__":
    logger.info("Testing Professional Buy Logic Changes")
    
    # Test strong signal
    strong_buy = test_strong_buy_signal()
    logger.info(f"Strong buy signal result: {'BUY' if strong_buy else 'HOLD'}")
    
    # Test weak signal
    weak_buy = test_weak_buy_signal()
    logger.info(f"Weak buy signal result: {'BUY' if weak_buy else 'HOLD'}")
    
    logger.info("Test completed")