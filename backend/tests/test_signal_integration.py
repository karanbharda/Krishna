#!/usr/bin/env python3
"""
Comprehensive test for signal integration in both analysis and decision making
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.advanced_feature_engineer import AdvancedFeatureEngineer
from core.professional_buy_logic import ProfessionalBuyLogic, StockMetrics, MarketTrend
from core.professional_sell_logic import ProfessionalSellLogic, PositionMetrics, MarketContext

def create_comprehensive_test_data():
    """Create comprehensive test data with strong signals for testing"""
    # Create 200 days of sample data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    # Generate price data with clear technical patterns
    np.random.seed(42)
    prices = [100]  # Start at $100
    
    # Create a clear uptrend with some volatility
    for i in range(1, len(dates)):
        # Add trend component
        trend = 0.001  # 0.1% daily trend
        # Add some volatility
        noise = np.random.normal(0, 0.02)
        # Add momentum
        if i > 50 and i < 100:
            trend += 0.003  # Strong uptrend
        elif i > 100 and i < 150:
            trend -= 0.002  # Correction
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(new_price)
    
    # Create OHLC data with clear patterns
    open_prices = [p * (1 + np.random.normal(0, 0.002)) for p in prices]
    high = [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices]
    low = [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices]
    volume = [np.random.randint(1000000, 5000000) for _ in range(200)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': prices,
        'Volume': volume
    })
    
    return df

def test_feature_engineering_comprehensive():
    """Test comprehensive feature engineering with all indicators"""
    print("=== Testing Comprehensive Feature Engineering ===")
    
    # Create sample data
    df = create_comprehensive_test_data()
    print(f"Created sample data with {len(df)} rows")
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer()
    
    # Engineer features
    features_df = feature_engineer.engineer_all_features(df)
    
    print(f"Engineered {len(features_df.columns)} features")
    
    # Check for key technical indicators
    key_indicators = [
        'rsi_14', 'stoch_k', 'stoch_d', 'willr_14', 'cci_14', 'roc_10', 'roc_20',
        'mfi_14', 'cmo_14', 'ultosc', 'stochrsi_k', 'stochrsi_d', 'ppo',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'atr_14', 'natr', 'std_dev_20', 'var_20',
        'macd', 'macd_signal', 'macd_hist', 'adx', 'plus_di', 'minus_di',
        'aroon_up', 'aroon_down', 'aroon_osc', 'trix', 'dx', 'mdi', 'pdi',
        'obv', 'ad', 'adosc', 'vroc_10', 'vol_sma_20', 'vol_ratio',
        'pvt', 'vwap', 'force_index', 'vol_osc', 'cmf',
        'sma_20', 'sma_50', 'sma_200', 'ema_20', 'wma_20', 'dema_20', 'tema_20',
        'ht_sine', 'ht_leadsine', 'ht_dcperiod', 'ht_dcphase', 'ht_phasor_inphase', 'ht_phasor_quad',
        'linearreg', 'correl', 'beta', 'tsf_14', 'linearreg_intercept', 'linearreg_angle', 'zscore_20',
        'price_position', 'gap_up', 'gap_down', 'intraday_range', 'body_size',
        'upper_shadow', 'lower_shadow', 'price_accel', 'close_to_sma20',
        'volatility_ratio', 'resistance_level', 'support_level'
    ]
    
    found_indicators = []
    missing_indicators = []
    
    for indicator in key_indicators:
        if indicator in features_df.columns:
            found_indicators.append(indicator)
        else:
            missing_indicators.append(indicator)
    
    print(f"Found {len(found_indicators)} key indicators out of {len(key_indicators)}")
    print(f"Missing {len(missing_indicators)} indicators: {missing_indicators[:10]}{'...' if len(missing_indicators) > 10 else ''}")
    
    return features_df

def test_buy_logic_comprehensive(features_df):
    """Test comprehensive buy logic with all signal categories"""
    print("\n=== Testing Comprehensive Buy Logic ===")
    
    # Create sample configuration
    config = {
        "min_buy_signals": 3,
        "min_buy_confidence": 0.60,
        "min_weighted_buy_score": 0.15,
        "signal_sensitivity_multiplier": 0.9,
        "aggressive_entry_threshold": 0.90
    }
    
    # Initialize buy logic
    buy_logic = ProfessionalBuyLogic(config)
    
    # Get the latest data point
    latest_idx = -1
    current_price = features_df['Close'].iloc[latest_idx]
    
    # Create sample stock metrics with strong buy signals
    stock_metrics = StockMetrics(
        current_price=current_price,
        entry_price=current_price * 0.95,  # 5% below current price to create unrealized gain
        quantity=100,
        volatility=0.02,
        atr=features_df['atr_14'].iloc[latest_idx] if 'atr_14' in features_df.columns else current_price * 0.01,
        rsi=30,  # Oversold condition
        macd=0.5,  # Positive MACD
        macd_signal=0.2,  # Positive MACD signal
        sma_20=features_df['sma_20'].iloc[latest_idx] if 'sma_20' in features_df.columns else current_price * 0.98,
        sma_50=features_df['sma_50'].iloc[latest_idx] if 'sma_50' in features_df.columns else current_price * 0.95,
        sma_200=features_df['sma_200'].iloc[latest_idx] if 'sma_200' in features_df.columns else current_price * 0.90,
        support_level=features_df['support_level'].iloc[latest_idx] if 'support_level' in features_df.columns else current_price * 0.95,
        resistance_level=features_df['resistance_level'].iloc[latest_idx] if 'resistance_level' in features_df.columns else current_price * 1.05,
        volume_ratio=1.5,  # Above average volume
        price_to_book=1.5,  # Reasonable P/B
        price_to_earnings=12,  # Reasonable P/E
        earnings_growth=0.10,  # 10% earnings growth
        return_on_equity=0.15,  # 15% ROE
        free_cash_flow_yield=0.08,  # 8% FCF yield
        debt_to_equity=0.3,  # Low debt
        dividend_yield=0.02,  # 2% dividend yield
        payout_ratio=0.3,  # 30% payout ratio
        earnings_quality=0.8,  # High earnings quality
        insider_ownership=0.15,  # 15% insider ownership
        sector_pe=15.0  # Sector P/E
    )
    
    # Create comprehensive technical analysis data
    technical_analysis = {
        # Basic indicators
        'rsi': 30,  # Oversold
        'rsi_14': 30,  # Oversold
        'rsi_5': 25,  # Even more oversold
        'stoch_k': 15,  # Oversold
        'stoch_d': 20,  # Oversold
        'willr_14': -85,  # Oversold
        'cci_14': -120,  # Oversold
        'roc_10': 2.0,  # Positive momentum
        'roc_20': 1.5,  # Positive momentum
        'mfi_14': 20,  # Oversold
        'cmo_14': -60,  # Oversold
        'ultosc': 35,  # Oversold
        'stochrsi_k': 10,  # Oversold
        'stochrsi_d': 15,  # Oversold
        'ppo': 0.5,  # Positive
        
        # Bollinger Bands
        'bb_position': 0.1,  # Near lower band
        'bb_width': 0.08,  # Normal width
        
        # Volatility
        'atr_14': features_df['atr_14'].iloc[latest_idx] if 'atr_14' in features_df.columns else current_price * 0.01,
        
        # Moving Averages
        'sma_20': features_df['sma_20'].iloc[latest_idx] if 'sma_20' in features_df.columns else current_price * 0.98,
        'sma_50': features_df['sma_50'].iloc[latest_idx] if 'sma_50' in features_df.columns else current_price * 0.95,
        'sma_200': features_df['sma_200'].iloc[latest_idx] if 'sma_200' in features_df.columns else current_price * 0.90,
        
        # Support/Resistance
        'support_level': features_df['support_level'].iloc[latest_idx] if 'support_level' in features_df.columns else current_price * 0.95,
        'resistance_level': features_df['resistance_level'].iloc[latest_idx] if 'resistance_level' in features_df.columns else current_price * 1.05,
        
        # Volume
        'vol_ratio': 1.5,  # Above average volume
        'volume_roc': 150,  # Strong volume increase
        'price_roc': 2.0,  # Positive price momentum
        
        # Trend indicators
        'macd': 0.5,  # Positive
        'macd_signal': 0.2,  # Positive
        'macd_histogram': 0.3,  # Positive
        'macd_histogram_prev': 0.1,  # Increasing
        'adx': 30,  # Strong trend
        'plus_di': 25,  # Positive directional movement
        'minus_di': 15,  # Weak negative directional movement
        'aroon_up': 80,  # Strong uptrend
        'aroon_down': 20,  # Weak downtrend
        'aroon_osc': 60,  # Bullish
        'trix': 0.001,  # Positive
        
        # Additional indicators
        'order_book_imbalance': 0.7,  # Bullish order flow
        'bid_ask_spread': 0.01,  # Tight spread
    }
    
    # Create sample market context with bullish conditions
    market_context = MarketContext(
        trend=MarketTrend.UPTREND,
        trend_strength=0.8,
        volatility_regime="normal",
        market_stress=0.2,
        sector_performance=0.03,
        volume_profile=1.2
    )
    
    # Create sample sentiment analysis with positive sentiment
    sentiment_analysis = {
        "overall_sentiment": 0.3,  # Positive sentiment
        "news_sentiment": {"positive": 0.7, "negative": 0.1, "neutral": 0.2},
        "social_sentiment": {"positive": 0.6, "negative": 0.2, "neutral": 0.2},
        "sentiment_momentum": 0.15,  # Improving sentiment
        "source_diversity": 5,  # Multiple sources
        "news_sentiment_trend": 0.1,  # Positive trend
        "social_volume": 1000,  # High volume
        "social_impact": 0.8,  # High impact
        "options_flow_sentiment": 0.5,  # Bullish options flow
        "call_put_ratio": 1.5,  # More calls than puts
        "flow_confirmed": True,  # Confirmed flow
        "insider_activity_score": 0.7,  # Active insider buying
        "recent_insider_buys": 5,  # Recent buys
        "insider_timing_score": 0.8,  # Good timing
        "insider_transaction_size": 0.7,  # Large transactions
        "pc_ratio": 1.6,  # High put/call ratio (fear indicator)
        "pc_trend": 0.15,  # Increasing fear (contrarian)
        "pc_volume": 6000,  # High volume
        "short_interest_pct": 0.05,  # 5% short interest
        "short_trend": -0.05,  # Declining short interest
        "short_momentum": -0.15  # Strong declining momentum
    }
    
    # Create sample ML analysis with bullish prediction
    ml_analysis = {
        "prediction_direction": 0.03,  # 3% positive prediction
        "confidence": 0.8,  # High confidence
        "model_accuracy": 0.85,  # High accuracy
        "ensemble_prediction": 0.025,  # Ensemble agreement
        "ensemble_models_count": 5,  # 5 models
        "model_diversity_score": 0.8,  # High diversity
        "ensemble_consistency": 0.9,  # High consistency
        "rl_recommendation": "BUY",  # RL recommends buy
        "rl_confidence": 0.75,  # RL confidence
        "rl_sharpe_ratio": 0.9,  # Good Sharpe ratio
        "rl_performance_history": [0.01, 0.02, 0.015, 0.025, 0.03],  # Good history
        "feature_importance_score": 0.85,  # High feature importance
        "key_bullish_features": ["rsi", "macd", "volume"],  # Key features aligned
        "feature_stability_score": 0.9,  # Stable features
        "predictive_power_score": 0.8,  # Good predictive power
        "cross_validation_score": 0.8,  # Good validation
        "backtest_performance": 0.75,  # Good backtest
        "robustness_score": 0.85,  # Robust model
        "generalization_score": 0.8,  # Good generalization
        "recent_accuracy": 0.85,  # Recent accuracy
        "prediction_consistency": 0.8,  # Consistent predictions
        "performance_trend": 0.1,  # Improving performance
        "consistency_trend": 0.05,  # Improving consistency
        "current_price": current_price,
        "predicted_price": current_price * 1.03,  # 3% price prediction
        "prediction_confidence": 0.8,  # Prediction confidence
        "confidence_interval": 0.04,  # 4% confidence interval
        "prediction_validation_score": 0.85  # Good validation
    }
    
    # Create sample portfolio context
    portfolio_context = {
        "available_cash": 10000,
        "position_size_limit": 0.1,
        "risk_tolerance": "moderate"
    }
    
    # Evaluate buy decision
    buy_decision = buy_logic.evaluate_buy_decision(
        ticker="TEST",
        stock_metrics=stock_metrics,
        market_context=market_context,
        technical_analysis=technical_analysis,
        sentiment_analysis=sentiment_analysis,
        ml_analysis=ml_analysis,
        portfolio_context=portfolio_context
    )
    
    print(f"Buy Decision: {buy_decision.should_buy}")
    print(f"Confidence: {buy_decision.confidence:.3f}")
    print(f"Reasoning: {buy_decision.reasoning}")
    print(f"Triggered Signals: {len([s for s in buy_decision.signals_triggered if s.triggered])}")
    
    # Analyze triggered signals by category
    categories = {}
    for signal in buy_decision.signals_triggered:
        if signal.triggered:
            if signal.category not in categories:
                categories[signal.category] = []
            categories[signal.category].append(signal)
    
    print("\nTriggered Signals by Category:")
    for category, signals in categories.items():
        print(f"  {category}: {len(signals)} signals")
        for signal in signals:
            print(f"    - {signal.name}: strength={signal.strength:.3f}, confidence={signal.confidence:.3f}")
    
    return buy_decision

def test_sell_logic_comprehensive(features_df):
    """Test comprehensive sell logic with all signal categories"""
    print("\n=== Testing Comprehensive Sell Logic ===")
    
    # Create sample configuration
    config = {
        "min_sell_signals": 2,
        "min_sell_confidence": 0.45,
        "min_weighted_sell_score": 0.06
    }
    
    # Initialize sell logic
    sell_logic = ProfessionalSellLogic(config)
    
    # Get the latest data point
    latest_idx = -1
    current_price = features_df['Close'].iloc[latest_idx]
    
    # Create sample position metrics with strong sell signals
    position_metrics = PositionMetrics(
        entry_price=current_price * 1.05,  # Entered at higher price (loss position)
        current_price=current_price,
        quantity=100,
        unrealized_pnl=(current_price - current_price * 1.05) * 100,  # Loss
        unrealized_pnl_pct=(current_price / (current_price * 1.05)) - 1,  # Loss percentage
        days_held=15,
        highest_price_since_entry=current_price * 1.08,  # Made 8% gain at peak
        lowest_price_since_entry=current_price * 0.92,  # Dropped 8% from peak
        volatility=0.03
    )
    
    # Create comprehensive technical analysis data for sell signals
    technical_analysis = {
        # Basic indicators
        'rsi': 80,  # Overbought
        'rsi_14': 80,  # Overbought
        'stoch_k': 95,  # Overbought
        'stoch_d': 90,  # Overbought
        'willr_14': -5,  # Overbought
        'cci_14': 150,  # Overbought
        'roc_10': -3.0,  # Negative momentum
        'roc_20': -2.0,  # Negative momentum
        'mfi_14': 85,  # Overbought
        'cmo_14': 70,  # Overbought
        'ultosc': 75,  # Overbought
        'stochrsi_k': 90,  # Overbought
        'stochrsi_d': 85,  # Overbought
        'ppo': -0.3,  # Negative
        
        # Bollinger Bands
        'bb_position': 0.9,  # Near upper band
        
        # Moving Averages
        'sma_20': features_df['sma_20'].iloc[latest_idx] if 'sma_20' in features_df.columns else current_price * 1.02,
        'sma_50': features_df['sma_50'].iloc[latest_idx] if 'sma_50' in features_df.columns else current_price * 1.01,
        
        # Support/Resistance
        'support_level': features_df['support_level'].iloc[latest_idx] if 'support_level' in features_df.columns else current_price * 0.95,
        
        # Trend indicators
        'macd': -0.2,  # Negative
        'macd_signal': 0.1,  # Positive (bearish crossover)
        'adx': 35,  # Strong trend
        'plus_di': 15,  # Weak positive directional movement
        'minus_di': 30,  # Strong negative directional movement
        'aroon_up': 20,  # Weak uptrend
        'aroon_down': 80,  # Strong downtrend
        'aroon_osc': -60,  # Bearish
        'trix': -0.002,  # Negative
    }
    
    # Create sample market context with bearish conditions
    market_context = MarketContext(
        trend=MarketTrend.DOWNTREND,
        trend_strength=0.7,
        volatility_regime="high",
        market_stress=0.7,
        sector_performance=-0.02,
        volume_profile=1.1
    )
    
    # Create sample sentiment analysis with negative sentiment
    sentiment_analysis = {
        "overall_sentiment": -0.2,  # Negative sentiment
        "news_sentiment": {"positive": 0.1, "negative": 0.7, "neutral": 0.2},
        "social_sentiment": {"positive": 0.2, "negative": 0.6, "neutral": 0.2}
    }
    
    # Create sample ML analysis with bearish prediction
    ml_analysis = {
        "prediction_direction": -0.02,  # 2% negative prediction
        "confidence": 0.75,  # High confidence
        "model_accuracy": 0.82  # High accuracy
    }
    
    # Evaluate sell decision
    sell_decision = sell_logic.evaluate_sell_decision(
        ticker="TEST",
        position_metrics=position_metrics,
        market_context=market_context,
        technical_analysis=technical_analysis,
        sentiment_analysis=sentiment_analysis,
        ml_analysis=ml_analysis
    )
    
    print(f"Sell Decision: {sell_decision.should_sell}")
    print(f"Confidence: {sell_decision.confidence:.3f}")
    print(f"Reasoning: {sell_decision.reasoning}")
    print(f"Triggered Signals: {len([s for s in sell_decision.signals_triggered if s.triggered])}")
    
    # Analyze triggered signals by category
    categories = {}
    for signal in sell_decision.signals_triggered:
        if signal.triggered:
            category = getattr(signal, 'category', 'Technical')  # Default to Technical for SellSignal
            if category not in categories:
                categories[category] = []
            categories[category].append(signal)
    
    print("\nTriggered Signals by Category:")
    for category, signals in categories.items():
        print(f"  {category}: {len(signals)} signals")
        for signal in signals:
            print(f"    - {signal.name}: strength={signal.strength:.3f}, confidence={signal.confidence:.3f}")
    
    return sell_decision

def test_analysis_report_generation(features_df, buy_decision, sell_decision):
    """Test generation of comprehensive analysis report"""
    print("\n=== Testing Analysis Report Generation ===")
    
    # Create a comprehensive analysis report
    report = {
        "timestamp": datetime.now().isoformat(),
        "symbol": "TEST",
        "current_price": features_df['Close'].iloc[-1],
        "technical_indicators": {},
        "buy_analysis": {
            "should_buy": buy_decision.should_buy,
            "confidence": buy_decision.confidence,
            "reasoning": buy_decision.reasoning,
            "triggered_signals": len([s for s in buy_decision.signals_triggered if s.triggered]),
            "signals_by_category": {}
        },
        "sell_analysis": {
            "should_sell": sell_decision.should_sell,
            "confidence": sell_decision.confidence,
            "reasoning": sell_decision.reasoning,
            "triggered_signals": len([s for s in sell_decision.signals_triggered if s.triggered]),
            "signals_by_category": {}
        }
    }
    
    # Add technical indicators to report
    key_technical_indicators = [
        'rsi_14', 'stoch_k', 'willr_14', 'cci_14', 'roc_10', 'mfi_14',
        'bb_position', 'atr_14', 'macd', 'macd_signal', 'adx', 'aroon_osc'
    ]
    
    for indicator in key_technical_indicators:
        if indicator in features_df.columns:
            report["technical_indicators"][indicator] = float(features_df[indicator].iloc[-1])
    
    # Add buy signals by category
    for signal in buy_decision.signals_triggered:
        if signal.triggered:
            category = signal.category
            if category not in report["buy_analysis"]["signals_by_category"]:
                report["buy_analysis"]["signals_by_category"][category] = []
            report["buy_analysis"]["signals_by_category"][category].append({
                "name": signal.name,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "reasoning": signal.reasoning
            })
    
    # Add sell signals by category
    for signal in sell_decision.signals_triggered:
        if signal.triggered:
            category = getattr(signal, 'category', 'Technical')  # Default to Technical for SellSignal
            if category not in report["sell_analysis"]["signals_by_category"]:
                report["sell_analysis"]["signals_by_category"][category] = []
            report["sell_analysis"]["signals_by_category"][category].append({
                "name": signal.name,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "reasoning": signal.reasoning
            })
    
    print("Analysis Report Generated Successfully")
    print(f"Report contains {len(report['technical_indicators'])} technical indicators")
    print(f"Buy analysis has {report['buy_analysis']['triggered_signals']} triggered signals")
    print(f"Sell analysis has {report['sell_analysis']['triggered_signals']} triggered signals")
    
    return report

def main():
    """Main test function"""
    print("=== Comprehensive Signal Integration Test ===")
    
    try:
        # Test feature engineering
        features_df = test_feature_engineering_comprehensive()
        
        # Test buy logic
        buy_decision = test_buy_logic_comprehensive(features_df)
        
        # Test sell logic
        sell_decision = test_sell_logic_comprehensive(features_df)
        
        # Test analysis report generation
        report = test_analysis_report_generation(features_df, buy_decision, sell_decision)
        
        print("\n=== Test Summary ===")
        print(f"Feature Engineering: {'PASS' if len(features_df.columns) > 100 else 'FAIL'} ({len(features_df.columns)} features)")
        print(f"Buy Logic Integration: {'PASS' if buy_decision else 'FAIL'}")
        print(f"Sell Logic Integration: {'PASS' if sell_decision else 'FAIL'}")
        print(f"Analysis Report Generation: {'PASS' if report else 'FAIL'}")
        
        # Check if we have signals from multiple categories
        buy_categories = set()
        if hasattr(buy_decision, 'signals_triggered'):
            for signal in buy_decision.signals_triggered:
                if hasattr(signal, 'category') and signal.category:
                    buy_categories.add(signal.category)
        
        print(f"Buy Signal Categories: {len(buy_categories)} ({', '.join(buy_categories)})")
        
        # Verify we have the required number of indicators
        expected_min_indicators = 25
        actual_indicators = len(features_df.columns)
        print(f"Technical Indicators: {'PASS' if actual_indicators >= expected_min_indicators else 'FAIL'} ({actual_indicators} >= {expected_min_indicators})")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)