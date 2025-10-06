"""
Test script to verify the buy logic fixes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.core.professional_buy_logic import ProfessionalBuyLogic, StockMetrics
from backend.core.professional_sell_logic import MarketTrend, MarketContext

def test_buy_logic():
    # Test configuration with stricter thresholds
    config = {
        "min_buy_signals": 3,
        "max_buy_signals": 5,
        "min_weighted_buy_score": 0.15,
        "min_buy_confidence": 0.60,
        "signal_sensitivity_multiplier": 0.9,
        "early_entry_buffer_pct": 0.01,
        "aggressive_entry_threshold": 0.90,
        "ml_signal_weight_boost": 0.05,
        "ml_confidence_multiplier": 1.10,
    }
    
    # Initialize buy logic
    buy_logic = ProfessionalBuyLogic(config)
    
    # Create test stock metrics
    stock_metrics = StockMetrics(
        current_price=100.0,
        entry_price=100.0,
        quantity=0,
        volatility=0.02,
        atr=1.5,
        rsi=45.0,
        macd=0.5,
        macd_signal=0.3,
        sma_20=99.0,
        sma_50=98.0,
        sma_200=95.0,
        support_level=95.0,
        resistance_level=105.0,
        volume_ratio=1.2,
        price_to_book=1.5,
        price_to_earnings=12.0,
        earnings_growth=0.08,
        return_on_equity=0.15,
        free_cash_flow_yield=0.08,
        debt_to_equity=0.3,
        dividend_yield=0.02,
        payout_ratio=0.3,
        earnings_quality=0.8,
        insider_ownership=0.15,
        sector_pe=18.0
    )
    
    # Create test market context
    market_context = MarketContext(
        trend=MarketTrend.UPTREND,
        trend_strength=0.6,
        volatility_regime="normal",
        market_stress=0.2,
        sector_performance=0.05,
        volume_profile=0.6
    )
    
    # Create test technical analysis data with weak signals
    technical_analysis = {
        "rsi": 45.0,
        "rsi_14": 45.0,
        "macd": 0.5,
        "macd_signal": 0.3,
        "sma_20": 99.0,
        "sma_50": 98.0,
        "support_level": 95.0,
        "resistance_level": 105.0,
        "mfi": 40.0,
        "stoch_k": 45.0,
        "bb_position": 0.3,
        "williams_r": -50.0,
        "volume_roc": 20.0,
        "order_book_imbalance": 0.1
    }
    
    # Create test sentiment analysis data with weak signals
    sentiment_analysis = {
        "overall_sentiment": 0.1,
        "sentiment_momentum": 0.05,
        "news_sentiment": {"positive": 10, "negative": 8, "neutral": 15},
        "social_sentiment": {"positive": 5, "negative": 4, "neutral": 10},
        "options_flow_sentiment": 0.1,
        "call_put_ratio": 1.05,
        "insider_activity_score": 0.2,
        "recent_insider_buys": 0,
        "pc_ratio": 1.0,
        "pc_trend": 0.02,
        "short_interest_pct": 0.03,
        "short_trend": -0.01
    }
    
    # Create test ML analysis data with weak signals
    ml_analysis = {
        "prediction_direction": 0.01,
        "confidence": 0.3,
        "model_accuracy": 0.6,
        "ensemble_prediction": 0.005,
        "ensemble_models_count": 2,
        "rl_recommendation": "HOLD",
        "rl_confidence": 0.4,
        "rl_sharpe_ratio": 0.3,
        "feature_importance_score": 0.4,
        "key_bullish_features": ["feature1"],
        "cross_validation_score": 0.5,
        "backtest_performance": 0.4,
        "recent_accuracy": 0.5,
        "prediction_consistency": 0.4,
        "current_price": 100.0,
        "predicted_price": 100.5,
        "prediction_confidence": 0.3
    }
    
    # Test with weak signals - should NOT generate buy signal
    print("=== TEST 1: Weak Signals (Should NOT Buy) ===")
    buy_decision = buy_logic.evaluate_buy_decision(
        ticker="TEST.STOCK",
        stock_metrics=stock_metrics,
        market_context=market_context,
        technical_analysis=technical_analysis,
        sentiment_analysis=sentiment_analysis,
        ml_analysis=ml_analysis,
        portfolio_context={}
    )
    
    print(f"Should Buy: {buy_decision.should_buy}")
    print(f"Confidence: {buy_decision.confidence:.3f}")
    print(f"Reasoning: {buy_decision.reasoning}")
    
    # Create test technical analysis data with strong signals
    strong_technical_analysis = {
        "rsi": 20.0,
        "rsi_14": 20.0,
        "macd": 2.0,
        "macd_signal": 0.5,
        "sma_20": 102.0,
        "sma_50": 100.0,
        "support_level": 99.0,
        "resistance_level": 105.0,
        "mfi": 5.0,
        "stoch_k": 5.0,
        "bb_position": 0.01,
        "williams_r": -95.0,
        "volume_roc": 150.0,
        "order_book_imbalance": 0.6
    }
    
    # Create test sentiment analysis data with strong signals
    strong_sentiment_analysis = {
        "overall_sentiment": 0.6,
        "sentiment_momentum": 0.3,
        "news_sentiment": {"positive": 30, "negative": 5, "neutral": 10},
        "social_sentiment": {"positive": 20, "negative": 3, "neutral": 5},
        "options_flow_sentiment": 0.6,
        "call_put_ratio": 1.6,
        "insider_activity_score": 0.8,
        "recent_insider_buys": 5,
        "pc_ratio": 1.6,
        "pc_trend": 0.15,
        "short_interest_pct": 0.12,
        "short_trend": -0.06
    }
    
    # Create test ML analysis data with strong signals
    strong_ml_analysis = {
        "prediction_direction": 0.10,
        "confidence": 0.85,
        "model_accuracy": 0.90,
        "ensemble_prediction": 0.08,
        "ensemble_models_count": 5,
        "rl_recommendation": "STRONG_BUY",
        "rl_confidence": 0.90,
        "rl_sharpe_ratio": 0.8,
        "feature_importance_score": 0.85,
        "key_bullish_features": ["feature1", "feature2", "feature3", "feature4"],
        "cross_validation_score": 0.85,
        "backtest_performance": 0.80,
        "recent_accuracy": 0.85,
        "prediction_consistency": 0.82,
        "current_price": 100.0,
        "predicted_price": 110.0,
        "prediction_confidence": 0.85
    }
    
    # Test with strong signals - should generate buy signal
    print("\n=== TEST 2: Strong Signals (Should Buy) ===")
    buy_decision = buy_logic.evaluate_buy_decision(
        ticker="TEST.STOCK",
        stock_metrics=stock_metrics,
        market_context=market_context,
        technical_analysis=strong_technical_analysis,
        sentiment_analysis=strong_sentiment_analysis,
        ml_analysis=strong_ml_analysis,
        portfolio_context={}
    )
    
    print(f"Should Buy: {buy_decision.should_buy}")
    print(f"Confidence: {buy_decision.confidence:.3f}")
    print(f"Buy Percentage: {buy_decision.buy_percentage:.3f}")
    print(f"Reasoning: {buy_decision.reasoning}")
    
    # Test with extremely large ML values (the original issue)
    extreme_ml_analysis = {
        "prediction_direction": 1006505.469,  # Extremely large value
        "confidence": 1098005.966,  # Extremely large value
        "model_accuracy": 0.85,
        "ensemble_prediction": 1006505.469,  # Extremely large value
        "ensemble_models_count": 4,
        "rl_recommendation": "STRONG_BUY",
        "rl_confidence": 0.85,
        "rl_sharpe_ratio": 0.7,
        "feature_importance_score": 0.8,
        "key_bullish_features": ["feature1", "feature2", "feature3"],
        "cross_validation_score": 0.8,
        "backtest_performance": 0.75,
        "recent_accuracy": 0.82,
        "prediction_consistency": 0.78,
        "current_price": 100.0,
        "predicted_price": 108.0,
        "prediction_confidence": 1098005.966  # Extremely large value
    }
    
    print("\n=== TEST 3: Extreme ML Values (Should Handle Properly) ===")
    buy_decision = buy_logic.evaluate_buy_decision(
        ticker="TEST.STOCK",
        stock_metrics=stock_metrics,
        market_context=market_context,
        technical_analysis=strong_technical_analysis,
        sentiment_analysis=strong_sentiment_analysis,
        ml_analysis=extreme_ml_analysis,
        portfolio_context={}
    )
    
    print(f"Should Buy: {buy_decision.should_buy}")
    print(f"Confidence: {buy_decision.confidence:.3f}")
    print(f"Buy Percentage: {buy_decision.buy_percentage:.3f}")
    print(f"Reasoning: {buy_decision.reasoning}")
    
    # Test with no signals at all
    print("\n=== TEST 4: No Signals (Should NOT Buy) ===")
    no_signal_technical = {
        "rsi": 50.0,
        "rsi_14": 50.0,
        "macd": 0.0,
        "macd_signal": 0.0,
        "sma_20": 100.0,
        "sma_50": 100.0,
        "support_level": 95.0,
        "resistance_level": 105.0,
        "mfi": 50.0,
        "stoch_k": 50.0,
        "bb_position": 0.5,
        "williams_r": -50.0,
        "volume_roc": 0.0,
        "order_book_imbalance": 0.0
    }
    
    no_signal_sentiment = {
        "overall_sentiment": 0.0,
        "sentiment_momentum": 0.0,
        "news_sentiment": {"positive": 10, "negative": 10, "neutral": 10},
        "social_sentiment": {"positive": 5, "negative": 5, "neutral": 10},
        "options_flow_sentiment": 0.0,
        "call_put_ratio": 1.0,
        "insider_activity_score": 0.0,
        "recent_insider_buys": 0,
        "pc_ratio": 1.0,
        "pc_trend": 0.0,
        "short_interest_pct": 0.05,
        "short_trend": 0.0
    }
    
    no_signal_ml = {
        "prediction_direction": 0.0,
        "confidence": 0.5,
        "model_accuracy": 0.5,
        "ensemble_prediction": 0.0,
        "ensemble_models_count": 1,
        "rl_recommendation": "HOLD",
        "rl_confidence": 0.5,
        "rl_sharpe_ratio": 0.5,
        "feature_importance_score": 0.5,
        "key_bullish_features": [],
        "cross_validation_score": 0.5,
        "backtest_performance": 0.5,
        "recent_accuracy": 0.5,
        "prediction_consistency": 0.5,
        "current_price": 100.0,
        "predicted_price": 100.0,
        "prediction_confidence": 0.5
    }
    
    buy_decision = buy_logic.evaluate_buy_decision(
        ticker="TEST.STOCK",
        stock_metrics=stock_metrics,
        market_context=market_context,
        technical_analysis=no_signal_technical,
        sentiment_analysis=no_signal_sentiment,
        ml_analysis=no_signal_ml,
        portfolio_context={}
    )
    
    print(f"Should Buy: {buy_decision.should_buy}")
    print(f"Confidence: {buy_decision.confidence:.3f}")
    print(f"Reasoning: {buy_decision.reasoning}")

if __name__ == "__main__":
    test_buy_logic()