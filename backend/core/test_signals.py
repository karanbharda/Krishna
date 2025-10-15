"""
Test script to verify the enhanced signal generation logic
"""
import sys
import os
import logging

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.professional_buy_logic import ProfessionalBuyLogic, BuySignal, StockMetrics
from core.professional_sell_logic import ProfessionalSellLogic, SellSignal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ml_signals():
    """Test ML signal generation with enhanced thresholds"""
    print("Testing ML Signal Generation...")
    
    # Create a ProfessionalBuyLogic instance
    config = {
        "min_buy_signals": 3,
        "max_buy_signals": 5,
        "min_buy_confidence": 0.60,
        "min_weighted_buy_score": 0.15,
        "signal_sensitivity_multiplier": 0.7,
        "early_entry_buffer_pct": 0.01,
        "aggressive_entry_threshold": 0.90,
        "ml_signal_weight_boost": 0.05,
        "ml_confidence_multiplier": 1.10
    }
    
    buy_logic = ProfessionalBuyLogic(config)
    
    # Test ML analysis data with values that should trigger signals with enhanced thresholds
    ml_analysis = {
        "success": True,
        "prediction_direction": 0.008,  # Slightly above new threshold of 0.005
        "confidence": 0.7,
        "model_accuracy": 0.8,
        "ensemble_prediction": 0.015,  # Slightly above new threshold of 0.01
        "ensemble_models_count": 2,
        "rl_recommendation": "BUY",
        "rl_confidence": 0.4,  # Slightly above new threshold of 0.3
        "feature_importance_score": 0.6,  # Slightly above new threshold of 0.5
        "key_bullish_features": ["feature1", "feature2"],
        "cross_validation_score": 0.6,  # Slightly above new threshold of 0.55
        "backtest_performance": 0.5,  # Slightly above new threshold of 0.45
        "recent_accuracy": 0.65,  # Slightly above new threshold of 0.60
        "prediction_consistency": 0.55,  # Slightly above new threshold of 0.50
        "predicted_price": 105,
        "current_price": 100,
        "prediction_confidence": 0.6,
        "confidence_interval": 0.04,  # Below threshold for boost
        "prediction_validation_score": 0.85  # Above threshold for boost
    }
    
    # Generate ML signals
    ml_signals = buy_logic._generate_ml_signals(ml_analysis, category_weight=0.20)
    
    print(f"Generated {len(ml_signals)} ML signals:")
    for signal in ml_signals:
        print(f"  - {signal.name}: strength={signal.strength:.3f}, confidence={signal.confidence:.3f}")
    
    # Verify that signals were generated
    assert len(ml_signals) > 0, "No ML signals were generated"
    print("✅ ML signal generation test passed")
    
def test_value_signals():
    """Test Value signal generation with enhanced thresholds"""
    print("\nTesting Value Signal Generation...")
    
    # Create a ProfessionalBuyLogic instance
    config = {
        "min_buy_signals": 3,
        "max_buy_signals": 5,
        "min_buy_confidence": 0.60,
        "min_weighted_buy_score": 0.15,
        "signal_sensitivity_multiplier": 0.7,
        "early_entry_buffer_pct": 0.01,
        "aggressive_entry_threshold": 0.90,
        "ml_signal_weight_boost": 0.05,
        "ml_confidence_multiplier": 1.10
    }
    
    buy_logic = ProfessionalBuyLogic(config)
    
    # Create stock metrics with values that should trigger signals with enhanced thresholds
    stock = StockMetrics(
        current_price=100.0,
        entry_price=95.0,
        quantity=100,
        volatility=0.02,
        atr=1.5,
        rsi=45.0,
        macd=0.5,
        macd_signal=0.3,
        sma_20=98.0,
        sma_50=95.0,
        sma_200=90.0,
        support_level=94.0,
        resistance_level=106.0,
        volume_ratio=1.2,
        price_to_book=1.6,  # Slightly above new threshold of 1.5
        price_to_earnings=16.0,  # Slightly above new threshold of 15
        earnings_growth=0.09,  # Slightly above new threshold of 0.08
        return_on_equity=0.13,  # Slightly above new threshold of 0.12
        free_cash_flow_yield=0.07,  # Slightly above new threshold of 0.06
        debt_to_equity=0.18,  # Slightly below new threshold of 0.20
        dividend_yield=0.02,  # Slightly above new threshold of 0.015
        payout_ratio=0.55,  # Slightly below new threshold of 0.6
        earnings_quality=0.7,  # Slightly above new threshold of 0.65
        insider_ownership=0.09  # Slightly above new threshold of 0.08
    )
    
    # Generate value signals
    value_signals = buy_logic._generate_value_signals(stock, category_weight=0.20)
    
    print(f"Generated {len(value_signals)} value signals:")
    for signal in value_signals:
        print(f"  - {signal.name}: strength={signal.strength:.3f}, confidence={signal.confidence:.3f}")
    
    # Verify that signals were generated
    assert len(value_signals) > 0, "No value signals were generated"
    print("✅ Value signal generation test passed")

def test_sell_signals():
    """Test Sell signal generation with enhanced thresholds"""
    print("\nTesting Sell Signal Generation...")
    
    # Create a ProfessionalSellLogic instance
    config = {
        "min_sell_signals": 2,
        "min_sell_confidence": 0.60,
        "min_weighted_sell_score": 0.15
    }
    
    sell_logic = ProfessionalSellLogic(config)
    
    # Test ML analysis data with values that should trigger signals with enhanced thresholds
    ml_analysis = {
        "prediction_direction": -0.015,  # Slightly below new threshold of -0.01
        "confidence": 0.7,
        "rl_recommendation": "SELL",
        "rl_confidence": 0.6
    }
    
    # Generate ML signals
    ml_signals = sell_logic._generate_ml_signals(ml_analysis)
    
    print(f"Generated {len(ml_signals)} sell signals:")
    for signal in ml_signals:
        print(f"  - {signal.name}: strength={signal.strength:.3f}, confidence={signal.confidence:.3f}")
    
    # Verify that signals were generated
    assert len(ml_signals) > 0, "No sell signals were generated"
    print("✅ Sell signal generation test passed")

if __name__ == "__main__":
    print("Running Signal Generation Tests...")
    test_ml_signals()
    test_value_signals()
    test_sell_signals()
    print("\n🎉 All tests passed!")