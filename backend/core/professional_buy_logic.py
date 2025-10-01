"""
Professional Buy Logic Framework
Implements institutional-grade buy logic with proper risk management,
entry confirmation, and market context awareness.
"""

import logging
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

# Import shared market context classes from professional_sell_logic
from .professional_sell_logic import MarketTrend, MarketContext

logger = logging.getLogger(__name__)

class BuyReason(Enum):
    """Professional buy reasons for audit trail"""
    TECHNICAL_BREAKOUT = "technical_breakout"
    VALUE_OPPORTUNITY = "value_opportunity"
    MOMENTUM_CONFIRMATION = "momentum_confirmation"
    SENTIMENT_IMPROVEMENT = "sentiment_improvement"
    ML_BULLISH_PREDICTION = "ml_bullish_prediction"
    RISK_REWARD_OPTIMAL = "risk_reward_optimal"
    MARKET_REGIME_FAVORABLE = "market_regime_favorable"

@dataclass
class BuySignal:
    """Individual buy signal with strength and reasoning"""
    name: str
    strength: float  # 0.0 to 1.0
    weight: float    # Signal importance weight
    triggered: bool
    reasoning: str
    confidence: float
    category: str = ""  # Category of the signal (Technical, Value, Sentiment, ML, Market)

@dataclass
class StockMetrics:
    """Stock-specific metrics for buy decisions"""
    current_price: float
    entry_price: float
    quantity: int
    volatility: float
    atr: float  # Average True Range
    rsi: float
    macd: float
    macd_signal: float
    sma_20: float
    sma_50: float
    sma_200: float
    support_level: float
    resistance_level: float
    volume_ratio: float
    price_to_book: float
    price_to_earnings: float

@dataclass
class BuyDecision:
    """Professional buy decision output"""
    should_buy: bool
    buy_quantity: int
    buy_percentage: float  # 0.0 to 1.0 (partial vs full entry)
    reason: BuyReason
    confidence: float
    urgency: float  # 0.0 to 1.0
    signals_triggered: List[BuySignal]
    target_entry_price: float
    stop_loss_price: float
    take_profit_price: float
    reasoning: str

class ProfessionalBuyLogic:
    """
    Professional-grade buy logic implementation
    Features:
    - Minimum 2-4 signal confirmation
    - Dynamic entry levels
    - Market independence
    - Risk-reward optimization
    - Market context awareness
    - OPTIMIZED BUY LOGIC: Enhanced signal detection and entry timing
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Professional thresholds - OPTIMIZED for better opportunity capture
        self.min_signals_required = config.get("min_buy_signals", 2)  # Minimum 2 signals
        self.max_signals_required = config.get("max_buy_signals", 4)  # Maximum 4 signals
        self.min_confidence_threshold = config.get("min_buy_confidence", 0.40)  # REDUCED from 0.50 to 0.40
        self.min_weighted_score = config.get("min_weighted_buy_score", 0.04)  # REDUCED from 0.12 to 0.04 (FIXED: Lowered threshold)
        
        # OPTIMIZED BUY LOGIC: Enhanced parameters for better entry timing
        self.signal_sensitivity_multiplier = config.get("signal_sensitivity_multiplier", 1.2)
        self.early_entry_buffer_pct = config.get("early_entry_buffer_pct", 0.005)  # Reduced for earlier entries
        self.aggressive_entry_threshold = config.get("aggressive_entry_threshold", 0.75)
        
        # OPTIMIZED BUY LOGIC: Dynamic signal thresholds
        self.dynamic_signal_thresholds = config.get("dynamic_signal_thresholds", True)
        self.signal_strength_boost = config.get("signal_strength_boost", 0.1)
        
        # OPTIMIZED BUY LOGIC: Enhanced ML integration
        self.ml_signal_weight_boost = config.get("ml_signal_weight_boost", 0.15)
        self.ml_confidence_multiplier = config.get("ml_confidence_multiplier", 1.3)
        
        # OPTIMIZED BUY LOGIC: Improved momentum detection
        self.momentum_confirmation_window = config.get("momentum_confirmation_window", 3)
        self.momentum_strength_threshold = config.get("momentum_strength_threshold", 0.02)
        
        # ADAPTIVE THRESHOLDS: Adjust based on market conditions
        self.adaptive_thresholds_enabled = config.get("adaptive_thresholds_enabled", True)
        self.market_volatility_threshold = config.get("market_volatility_threshold", 0.03)
        self.low_confidence_multiplier = config.get("low_confidence_multiplier", 0.8)
        self.high_confidence_multiplier = config.get("high_confidence_multiplier", 1.2)
        
        # DYNAMIC STOP-LOSS: Read from live_config.json instead of hardcoded
        self._load_dynamic_config()
        
        logger.info("Professional Buy Logic initialized with enhanced optimization parameters")
    
    def _load_dynamic_config(self):
        """Load dynamic configuration from live_config.json"""
        try:
            import json
            import os
            
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'live_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    live_config = json.load(f)
                
                # Get dynamic stop-loss percentage from frontend config
                self.stop_loss_pct = live_config.get("stop_loss_pct", 0.05)
                self.take_profit_ratio = live_config.get("take_profit_ratio", 2.0)
                
                logger.info(f"📊 Loaded dynamic config - Stop Loss: {self.stop_loss_pct:.1%}, Take Profit Ratio: {self.take_profit_ratio}")
            else:
                logger.warning("live_config.json not found, using defaults")
                self.stop_loss_pct = 0.05
                self.take_profit_ratio = 2.0
                
        except Exception as e:
            logger.error(f"Failed to load dynamic config: {e}")
            self.stop_loss_pct = 0.05
            self.take_profit_ratio = 2.0
    
    def refresh_dynamic_config(self):
        """Refresh dynamic configuration from live_config.json (call this periodically)"""
        self._load_dynamic_config()
    
    def evaluate_buy_decision(
        self,
        ticker: str,
        stock_metrics: StockMetrics,
        market_context: MarketContext,
        technical_analysis: Dict,
        sentiment_analysis: Dict,
        ml_analysis: Dict,
        portfolio_context: Dict
    ) -> BuyDecision:
        """
        Main entry point for professional buy evaluation
        """
        logger.info(f"=== PROFESSIONAL BUY EVALUATION: {ticker} ===")
        logger.info(f"Current Price: {stock_metrics.current_price:.2f}")
        logger.info(f"Market Context: {market_context.trend.value} (strength: {market_context.trend_strength:.2f})")
        
        # Step 1: Generate all buy signals with enhanced sensitivity
        signals = self._generate_buy_signals(
            stock_metrics, market_context, technical_analysis, 
            sentiment_analysis, ml_analysis
        )
        
        # Log signal generation with detailed information
        logger.info(f"Generated {len(signals)} total signals")
        triggered_signals = [s for s in signals if s.triggered]
        logger.info(f"{len(triggered_signals)} triggered signals:")
        
        # Group signals by category for better visualization
        category_signals = {}
        for signal in triggered_signals:
            category = signal.category or "Uncategorized"
            if category not in category_signals:
                category_signals[category] = []
            category_signals[category].append(signal)
        
        for category, signals_list in category_signals.items():
            logger.info(f"  {category} Signals:")
            for signal in signals_list:
                logger.info(f"    - {signal.name}: strength={signal.strength:.3f}, weight={signal.weight:.3f}, confidence={signal.confidence:.3f}")
                logger.info(f"      Reasoning: {signal.reasoning}")
        
        # Log non-triggered signals for completeness
        non_triggered_signals = [s for s in signals if not s.triggered]
        if non_triggered_signals:
            logger.info(f"{len(non_triggered_signals)} non-triggered signals:")
            for signal in non_triggered_signals:
                logger.info(f"  - {signal.name}: Not triggered - {signal.reasoning}")
        
        # Step 2: Apply cross-category confirmation (at least 2 categories must align)
        cross_category_confirmed = self._check_cross_category_confirmation(signals)
        categories_triggered = set(s.category for s in triggered_signals if s.category)
        logger.info(f"Cross-category confirmation: {cross_category_confirmed} (Categories: {', '.join(categories_triggered) if categories_triggered else 'None'})")
        if not cross_category_confirmed:
            logger.info("❌ BUY REJECTED: Cross-category confirmation failed - signals not aligned across multiple categories")
            detailed_reasoning = f"Cross-category confirmation failed - only {len(categories_triggered)} categories triggered: {', '.join(categories_triggered) if categories_triggered else 'None'}. Need at least 2 different categories."
            return BuyDecision(
                should_buy=False,
                buy_quantity=0,
                buy_percentage=0.0,
                reason=BuyReason.TECHNICAL_BREAKOUT,
                confidence=0.0,
                urgency=0.0,
                signals_triggered=signals,
                target_entry_price=0.0,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                reasoning=detailed_reasoning
            )
        
        # Step 3: Apply bearish block filter
        bearish_blocked = self._check_bearish_block(signals)
        bearish_signals = [s for s in triggered_signals if "bearish" in s.name.lower() or "overbought" in s.name.lower()]
        bearish_percentage = len(bearish_signals) / len(triggered_signals) if triggered_signals else 0
        logger.info(f"Bearish block filter: {bearish_blocked} ({len(bearish_signals)} bearish signals, {bearish_percentage:.1%} of triggered signals)")
        if bearish_blocked:
            logger.info("❌ BUY REJECTED: Bearish block filter triggered - too many bearish signals")
            detailed_reasoning = f"Bearish block filter triggered - {len(bearish_signals)} bearish signals ({bearish_percentage:.1%} of triggered signals). Threshold is 30%."
            return BuyDecision(
                should_buy=False,
                buy_quantity=0,
                buy_percentage=0.0,
                reason=BuyReason.TECHNICAL_BREAKOUT,
                confidence=0.0,
                urgency=0.0,
                signals_triggered=signals,
                target_entry_price=0.0,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                reasoning=detailed_reasoning
            )
        
        # Step 4: Calculate weighted signal score and confidence
        triggered_signals = [s for s in signals if s.triggered]
        total_weight = sum(s.weight for s in triggered_signals)
        weighted_score = sum(s.strength * s.weight for s in triggered_signals)

        # Calculate confidence based on signal agreement
        signal_confidences = [s.confidence for s in triggered_signals]
        avg_confidence = np.mean(signal_confidences) if signal_confidences else 0.0

        # Professional decision criteria with adjusted thresholds
        signals_count = len(triggered_signals)
        # Check if signal count is within the required range (min 2, max 4)
        meets_signal_threshold = self.min_signals_required <= signals_count <= self.max_signals_required
        meets_confidence_threshold = avg_confidence >= self.min_confidence_threshold
        meets_weighted_threshold = weighted_score >= self.min_weighted_score

        logger.info(f"Signal Analysis Summary:")
        logger.info(f"  Signal Count: {signals_count} (Required: {self.min_signals_required}-{self.max_signals_required}) - {'✅ PASS' if meets_signal_threshold else '❌ FAIL'}")
        logger.info(f"  Average Confidence: {avg_confidence:.3f} (Threshold: {self.min_confidence_threshold}) - {'✅ PASS' if meets_confidence_threshold else '❌ FAIL'}")
        logger.info(f"  Weighted Score: {weighted_score:.3f} (Threshold: {self.min_weighted_score}) - {'✅ PASS' if meets_weighted_threshold else '❌ FAIL'}")

        # PROFESSIONAL BUY LOGIC: All three conditions must be met (similar to sell logic)
        # This prevents the system from generating buy signals when conditions are marginal
        should_buy = meets_signal_threshold and meets_confidence_threshold and meets_weighted_threshold

        # Additional quality checks for professional trading
        if should_buy:
            logger.info("✅ ALL THRESHOLD CHECKS PASSED - Proceeding with buy decision")
            # Step 5: Calculate entry levels with enhanced timing
            entry_levels = self._calculate_optimized_entry_levels(stock_metrics, market_context)
            
            # Step 6: Apply market context filters
            base_decision = BuyDecision(
                should_buy=True,
                buy_quantity=0,  # Will be determined by portfolio manager
                buy_percentage=0.0,  # Will be calculated
                reason=BuyReason.TECHNICAL_BREAKOUT,
                confidence=avg_confidence,
                urgency=0.5,  # Will be adjusted
                signals_triggered=triggered_signals,
                target_entry_price=entry_levels["target_entry"],
                stop_loss_price=entry_levels["stop_loss"],
                take_profit_price=entry_levels["take_profit"],
                reasoning=f"Professional buy confirmed: {signals_count} signals, "
                         f"confidence {avg_confidence:.3f}, weighted score {weighted_score:.3f}"
            )
            
            # Apply market context filters
            final_decision = self._apply_market_context_filters(base_decision, market_context)
            
            # Calculate position sizing with enhanced optimization
            final_decision = self._calculate_enhanced_position_sizing(final_decision, weighted_score, triggered_signals)
            
            logger.info(f"Final Buy Decision: Should Buy: {final_decision.should_buy}, "
                       f"Confidence: {final_decision.confidence:.3f}, "
                       f"Buy Percentage: {final_decision.buy_percentage:.3f}")
            
            return final_decision
        else:
            logger.info("❌ THRESHOLD CHECKS FAILED - Generating hold decision")
            # Provide detailed reasoning for why buy was rejected
            rejection_reasons = []
            if not meets_signal_threshold:
                rejection_reasons.append(f"Signal count {signals_count} not in range [{self.min_signals_required}, {self.max_signals_required}]")
            if not meets_confidence_threshold:
                rejection_reasons.append(f"Confidence {avg_confidence:.3f} below threshold {self.min_confidence_threshold}")
            if not meets_weighted_threshold:
                rejection_reasons.append(f"Weighted score {weighted_score:.3f} below threshold {self.min_weighted_score}")
            
            detailed_reasoning = " | ".join(rejection_reasons)
            logger.info(f"Rejection Reasons: {detailed_reasoning}")
            
            # Log additional diagnostic information
            logger.info(f"🔍 DETAILED DIAGNOSTIC INFORMATION:")
            logger.info(f"   - Minimum Signals Required: {self.min_signals_required}")
            logger.info(f"   - Maximum Signals Required: {self.max_signals_required}")
            logger.info(f"   - Minimum Confidence Threshold: {self.min_confidence_threshold}")
            logger.info(f"   - Minimum Weighted Score Threshold: {self.min_weighted_score}")
            logger.info(f"   - Signal Sensitivity Multiplier: {getattr(self, 'signal_sensitivity_multiplier', 1.0)}")
            logger.info(f"   - ML Signal Weight Boost: {getattr(self, 'ml_signal_weight_boost', 0.0)}")
            
            return BuyDecision(
                should_buy=False,
                buy_quantity=0,
                buy_percentage=0.0,
                reason=BuyReason.TECHNICAL_BREAKOUT,
                confidence=avg_confidence,
                urgency=0.0,
                signals_triggered=signals,
                target_entry_price=0.0,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                reasoning=detailed_reasoning
            )

    def _check_bearish_block(self, signals: List[BuySignal]) -> bool:
        """Bearish block filter: If bearish > 30%, block trade (REDUCED from 40% for more opportunities)"""
        triggered_signals = [s for s in signals if s.triggered]
        if not triggered_signals:
            return False
            
        bearish_signals = [s for s in triggered_signals if "bearish" in s.name.lower() or "overbought" in s.name.lower()]
        bearish_percentage = len(bearish_signals) / len(triggered_signals)
        return bearish_percentage > 0.3  # REDUCED from 0.4 to 0.3

    def _generate_buy_signals(
        self,
        stock_metrics: StockMetrics,
        market_context: MarketContext,
        technical_analysis: Dict,
        sentiment_analysis: Dict,
        ml_analysis: Dict
    ) -> List[BuySignal]:
        """Generate comprehensive buy signals with professional weighting"""
        signals = []

        # Get dynamic category weights based on market regime
        dynamic_weights = self._get_dynamic_category_weights(market_context)

        # 1. Technical Analysis Signals
        technical_signals = self._generate_technical_signals(technical_analysis, stock_metrics, dynamic_weights["Technical"])
        signals.extend(technical_signals)

        # 2. Value & Risk Signals
        value_signals = self._generate_value_signals(stock_metrics, dynamic_weights["Value"])
        signals.extend(value_signals)

        # 3. Sentiment Signals
        sentiment_signals = self._generate_sentiment_signals(sentiment_analysis, dynamic_weights["Sentiment"])
        signals.extend(sentiment_signals)

        # 4. ML/AI Signals
        ml_signals = self._generate_ml_signals(ml_analysis, dynamic_weights["ML"])
        signals.extend(ml_signals)

        # 5. Market Structure Signals
        market_signals = self._generate_market_signals(market_context, dynamic_weights["Market"])
        signals.extend(market_signals)

        return signals

    def _get_dynamic_category_weights(self, market_context: MarketContext) -> Dict[str, float]:
        """Dynamic weighting: Adjusts by market regime (bullish/bearish phases)"""
        # Start with base weights
        weights = self.category_weights.copy()
        
        # Adjust weights based on market trend
        if market_context.trend in [MarketTrend.STRONG_UPTREND, MarketTrend.UPTREND]:
            # In bullish market, increase weight on momentum/technical signals
            weights["Technical"] = min(0.30, weights["Technical"] + 0.05)
            weights["ML"] = min(0.25, weights["ML"] + 0.05)
            # Reduce weight on value signals as they might be less relevant
            weights["Value"] = max(0.15, weights["Value"] - 0.05)
        elif market_context.trend in [MarketTrend.STRONG_DOWNTREND, MarketTrend.DOWNTREND]:
            # In bearish market, increase weight on value and sentiment signals
            weights["Value"] = min(0.25, weights["Value"] + 0.05)
            weights["Sentiment"] = min(0.25, weights["Sentiment"] + 0.05)
            # Reduce weight on technical signals as they might be misleading
            weights["Technical"] = max(0.20, weights["Technical"] - 0.05)
        
        # Ensure weights sum to 1.0
        total_weight = sum(weights.values())
        if total_weight != 1.0:
            weights = {k: v/total_weight for k, v in weights.items()}
            
        return weights

    def _check_cross_category_confirmation(self, signals: List[BuySignal]) -> bool:
        """Cross-category confirmation: At least 2 categories must align"""
        triggered_signals = [s for s in signals if s.triggered]
        categories = set(s.category for s in triggered_signals if s.category)
        return len(categories) >= 2

    def _generate_technical_signals(self, technical: Dict, stock: StockMetrics, category_weight: float = 0.25) -> List[BuySignal]:
        """Generate technical analysis buy signals with enhanced sensitivity"""
        signals = []

        # OPTIMIZED BUY LOGIC: Enhanced RSI signal detection
        rsi = technical.get("rsi", 50)
        if rsi < 35:  # Slightly higher threshold for earlier entries
            strength = min((35 - rsi) / 25, 1.0) * self.signal_sensitivity_multiplier
            signals.append(BuySignal(
                name="rsi_oversold",
                strength=strength,
                weight=category_weight * 0.12,  # Increased from 0.08 to 0.12 (FIXED: Stronger weight for RSI signals)
                triggered=True,
                reasoning=f"RSI oversold at {rsi:.1f}",
                confidence=0.8,
                category="Technical"
            ))

        # OPTIMIZED BUY LOGIC: Enhanced MACD signal detection
        macd = technical.get("macd", 0)
        macd_signal = technical.get("macd_signal", 0)
        if macd > macd_signal:  # Removed requirement for positive MACD for earlier entries
            strength = min(abs(macd - macd_signal) / abs(macd_signal) if macd_signal != 0 else 1, 1.0) * self.signal_sensitivity_multiplier
            signals.append(BuySignal(
                name="macd_bullish",
                strength=strength,
                weight=category_weight * 0.10,  # Increased from 0.07 to 0.10 (FIXED: Stronger weight for MACD signals)
                triggered=True,
                reasoning="MACD bullish crossover",
                confidence=0.7,
                category="Technical"
            ))

        # OPTIMIZED BUY LOGIC: Enhanced moving average signals
        sma_20 = technical.get("sma_20", stock.current_price)
        sma_50 = technical.get("sma_50", stock.current_price)
        if stock.current_price > sma_20 * 0.99:  # Slightly relaxed condition
            strength = (stock.current_price - sma_20 * 0.99) / (sma_20 * 0.01)
            signals.append(BuySignal(
                name="ma_support",
                strength=min(strength, 1.0) * self.signal_sensitivity_multiplier,
                weight=category_weight * 0.09,  # Increased from 0.06 to 0.09 (FIXED: Stronger weight for MA signals)
                triggered=True,
                reasoning="Price near key moving averages",
                confidence=0.75,
                category="Technical"
            ))

        # OPTIMIZED BUY LOGIC: Enhanced support level detection
        support = technical.get("support_level", stock.current_price * 1.1)
        if support > 0 and stock.current_price > support * 1.01:
            strength = (stock.current_price - support) / support
            signals.append(BuySignal(
                name="support_bounce",
                strength=min(strength * 2, 1.0) * self.signal_sensitivity_multiplier,  # Amplify support bounces
                weight=category_weight * 0.12,  # Increased from 0.09 to 0.12 (FIXED: Stronger weight for support signals)
                triggered=True,
                reasoning=f"Support bounce at {support:.2f}",
                confidence=0.85,
                category="Technical"
            ))

        # OPTIMIZED BUY LOGIC: Enhanced breakout signal detection
        resistance = technical.get("resistance_level", stock.current_price * 0.9)
        if stock.current_price > resistance * 1.005:  # Earlier breakout detection
            strength = (stock.current_price - resistance) / resistance
            signals.append(BuySignal(
                name="breakout",
                strength=min(strength * 1.5, 1.0) * self.signal_sensitivity_multiplier,  #Amplify breakouts
                weight=category_weight * 0.15,  # Increased from 0.10 to 0.15 (FIXED: Stronger weight for breakout signals)
                triggered=True,
                reasoning=f"Breakout above resistance {resistance:.2f}",
                confidence=0.9,
                category="Technical"
            ))

        return signals

    def _generate_value_signals(self, stock: StockMetrics, category_weight: float = 0.20) -> List[BuySignal]:
        """Generate value-based buy signals"""
        signals = []

        # Low P/E Ratio Signal
        pe_ratio = stock.price_to_earnings
        if 0 < pe_ratio < 15:  # Reasonable P/E threshold
            strength = min((15 - pe_ratio) / 15, 1.0)
            signals.append(BuySignal(
                name="low_pe_ratio",
                strength=strength,
                weight=category_weight * 0.10,  # Increased from 0.08 to 0.10 (FIXED: Stronger weight for P/E signals)
                triggered=True,
                reasoning=f"Low P/E ratio: {pe_ratio:.1f}",
                confidence=0.6,
                category="Value"
            ))

        # Low P/B Ratio Signal
        pb_ratio = stock.price_to_book
        if 0 < pb_ratio < 1.5:  # Below book value or reasonable threshold
            strength = min((1.5 - pb_ratio) / 1.5, 1.0)
            signals.append(BuySignal(
                name="low_pb_ratio",
                strength=strength,
                weight=category_weight * 0.09,  # Increased from 0.07 to 0.09 (FIXED: Stronger weight for P/B signals)
                triggered=True,
                reasoning=f"Low P/B ratio: {pb_ratio:.2f}",
                confidence=0.65,
                category="Value"
            ))

        # Volatility Compression Signal (before breakout)
        if stock.volatility < 0.02:  # Low volatility (2%)
            strength = min((0.02 - stock.volatility) / 0.02, 1.0)
            signals.append(BuySignal(
                name="volatility_compression",
                strength=strength,
                weight=category_weight * 0.07,  # Increased from 0.05 to 0.07 (FIXED: Stronger weight for volatility signals)
                triggered=True,
                reasoning=f"Low volatility: {stock.volatility:.1%}",
                confidence=0.5,
                category="Value"
            ))

        # ATR-based entry opportunity
        if stock.atr > 0 and stock.atr < stock.current_price * 0.03:  # Low ATR
            strength = min((stock.current_price * 0.03 - stock.atr) / (stock.current_price * 0.03), 1.0)
            signals.append(BuySignal(
                name="low_atr_opportunity",
                strength=strength,
                weight=category_weight * 0.07,  # Increased from 0.05 to 0.07 (FIXED: Stronger weight for ATR signals)
                triggered=True,
                reasoning=f"Low ATR: {stock.atr:.2f}",
                confidence=0.55,
                category="Value"
            ))

        return signals

    def _generate_sentiment_signals(self, sentiment: Dict, category_weight: float = 0.20) -> List[BuySignal]:
        """Generate sentiment-based buy signals"""
        signals = []

        # Positive sentiment signal
        sentiment_score = sentiment.get("overall_sentiment", 0)
        if sentiment_score > 0.2:  # Positive sentiment threshold
            strength = min(sentiment_score / 0.8, 1.0)  # Scale to 0.8 max
            signals.append(BuySignal(
                name="positive_sentiment",
                strength=strength,
                weight=category_weight * 0.15,  # Increased from 0.12 to 0.15 (FIXED: Stronger weight for sentiment signals)
                triggered=True,
                reasoning=f"Positive sentiment: {sentiment_score:.2f}",
                confidence=0.6,
                category="Sentiment"
            ))

        # News sentiment improvement
        news_trend = sentiment.get("news_trend", 0)
        if news_trend > 0.3:
            strength = min(news_trend / 0.7, 1.0)
            signals.append(BuySignal(
                name="news_improvement",
                strength=strength,
                weight=category_weight * 0.10,  # Increased from 0.08 to 0.10 (FIXED: Stronger weight for news signals)
                triggered=True,
                reasoning="Improving news sentiment",
                confidence=0.5,
                category="Sentiment"
            ))

        return signals

    def _generate_ml_signals(self, ml_analysis: Dict, category_weight: float = 0.20) -> List[BuySignal]:
        """Generate ML/AI-based buy signals with enhanced weighting"""
        signals = []

        # OPTIMIZED BUY LOGIC: Enhanced ML signal detection
        ml_prediction = ml_analysis.get("prediction_direction", 0)
        if ml_prediction > 0.01:  # Lower threshold for earlier entries
            strength = min(ml_prediction / 0.10, 1.0) * self.signal_sensitivity_multiplier
            signals.append(BuySignal(
                name="ml_bullish_prediction",
                strength=strength,
                # OPTIMIZED BUY LOGIC: Boost ML signal weight
                weight=category_weight * (0.15 + self.ml_signal_weight_boost),  # Increased from 0.10 to 0.15 (FIXED: Stronger ML signal weight)
                triggered=True,
                # OPTIMIZED BUY LOGIC: Boost ML confidence
                confidence=ml_analysis.get("confidence", 0.5) * self.ml_confidence_multiplier,
                category="ML"
            ))

        # OPTIMIZED BUY LOGIC: Enhanced RL recommendation
        rl_recommendation = ml_analysis.get("rl_recommendation", "HOLD")
        if rl_recommendation in ["BUY", "STRONG_BUY"]:
            rl_confidence = ml_analysis.get("rl_confidence", 0.5)
            # Boost confidence for strong recommendations
            confidence_multiplier = 1.5 if rl_recommendation == "STRONG_BUY" else 1.2
            signals.append(BuySignal(
                name="rl_buy_recommendation",
                strength=rl_confidence * self.signal_sensitivity_multiplier,
                # OPTIMIZED BUY LOGIC: Boost RL signal weight
                weight=category_weight * (0.10 + self.ml_signal_weight_boost * 0.5),  # Increased from 0.05 to 0.10 (FIXED: Stronger RL signal weight)
                triggered=True,
                confidence=rl_confidence * confidence_multiplier,
                category="ML"
            ))

        return signals

    def _generate_market_signals(self, market_context: MarketContext, category_weight: float = 0.15) -> List[BuySignal]:
        """Generate market structure buy signals"""
        signals = []

        # Low market stress signal (favorable for entries)
        if market_context.market_stress < 0.4:
            strength = (0.4 - market_context.market_stress) / 0.4
            signals.append(BuySignal(
                name="low_market_stress",
                strength=strength,
                weight=category_weight * 0.08,  # Increased from 0.06 to 0.08 (FIXED: Stronger weight for market stress signals)
                triggered=True,
                reasoning=f"Low market stress: {market_context.market_stress:.1%}",
                confidence=0.7,
                category="Market"
            ))

        # Sector outperformance
        if market_context.sector_performance > 0.02:
            strength = min(market_context.sector_performance / 0.05, 1.0)
            signals.append(BuySignal(
                name="sector_strength",
                strength=strength,
                weight=category_weight * 0.06,  # Increased from 0.04 to 0.06 (FIXED: Stronger weight for sector signals)
                triggered=True,
                reasoning="Sector outperforming market",
                confidence=0.6,
                category="Market"
            ))

        return signals

    def _calculate_optimized_entry_levels(self, stock: StockMetrics, market_context: MarketContext) -> Dict:
        """Calculate optimized entry levels for better timing"""
        
        #OPTIMIZED BUY LOGIC: Earlier entry opportunities
        base_entry = stock.current_price * (1 - self.early_entry_buffer_pct)

        # Volatility-adjusted entry with enhanced sensitivity
        volatility_multiplier = 1 + (stock.volatility / 0.02) * 0.5  # Reduced multiplier for earlier entries
        volatility_entry = stock.current_price * (1 - self.early_entry_buffer_pct * volatility_multiplier)

        # Support-based entry with buffer
        support_entry = stock.support_level * 1.005 if stock.support_level > 0 else stock.current_price  # Reduced buffer

        # Choose the most aggressive (lowest) entry for better timing
        target_entry = min(base_entry, volatility_entry, support_entry)

        # Stop-loss based on ATR with dynamic adjustment
        stop_loss = target_entry * (1 - self.stop_loss_pct * (1 - min(stock.volatility / 0.03, 1)))

        # Take-profit based on risk-reward ratio with enhanced targets
        risk = target_entry - stop_loss
        take_profit = target_entry + (risk * self.take_profit_ratio * 1.2)  # Increased profit target

        return {
            "target_entry": target_entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "base_entry": base_entry,
            "support_entry": support_entry
        }

    def _apply_market_context_filters(self, decision: BuyDecision, market_context: MarketContext) -> BuyDecision:
        """Apply market context filters to buy decision"""

        if not decision.should_buy:
            logger.info("Market context filters skipped - no buy decision")
            return decision

        logger.info(f"Applying market context filters - Market Trend: {market_context.trend.value}")

        # Be more conservative in downtrends
        if market_context.trend in [MarketTrend.DOWNTREND, MarketTrend.STRONG_DOWNTREND]:
            original_confidence = decision.confidence
            decision.confidence *= self.downtrend_buy_multiplier
            logger.info(f"Downtrend filter applied - confidence reduced from {original_confidence:.3f} to {decision.confidence:.3f}")
            if decision.confidence < self.min_confidence_threshold:
                logger.info(f"BUY BLOCKED: Market in {market_context.trend.value}, confidence {decision.confidence:.3f} < threshold {self.min_confidence_threshold}")
                decision.should_buy = False
                decision.reasoning += " | BLOCKED: Downtrend"
                return decision

        # Be more aggressive in uptrends
        elif market_context.trend in [MarketTrend.STRONG_UPTREND, MarketTrend.UPTREND]:
            original_confidence = decision.confidence
            original_urgency = decision.urgency
            decision.confidence *= self.uptrend_buy_multiplier
            decision.urgency = min(decision.urgency * 1.2, 1.0)
            logger.info(f"Uptrend filter applied - confidence increased from {original_confidence:.3f} to {decision.confidence:.3f}, "
                       f"urgency increased from {original_urgency:.3f} to {decision.urgency:.3f}")

        return decision

    def _calculate_enhanced_position_sizing(
        self,
        decision: BuyDecision,
        weighted_score: float,
        triggered_signals: List[BuySignal]
    ) -> BuyDecision:
        """Calculate enhanced position sizing based on signal quality and ML predictions"""

        if not decision.should_buy:
            logger.info("Position sizing skipped - no buy decision")
            return decision

        logger.info(f"Calculating enhanced position sizing - Weighted Score: {weighted_score:.3f}")

       # Base position scale based on weighted score
        position_scale = min(weighted_score * 1.5, 1.0)  # Increased multiplier for better scaling
        position_scale = max(position_scale, 0.1)  # Minimum 10% position
        logger.info(f"Base position scale from weighted score: {position_scale:.3f}")

        # OPTIMIZED BUY LOGIC: Boost position size for high-confidence ML signals
        ml_signals = [s for s in triggered_signals if s.category == "ML"]
        if ml_signals:
            ml_confidence = np.mean([s.confidence for s in ml_signals])
            logger.info(f"ML signals detected - average confidence: {ml_confidence:.3f}")
            if ml_confidence > 0.7:
                original_scale = position_scale
                position_scale *= 1.3  # 30% boost for high-confidence ML signals
                logger.info(f"High-confidence ML boost applied - position scale increased from {original_scale:.3f} to {position_scale:.3f}")
            elif ml_confidence > 0.5:
                original_scale = position_scale
                position_scale *= 1.1  # 10% boost for medium-confidence ML signals
                logger.info(f"Medium-confidence ML boost applied - position scale increased from {original_scale:.3f} to {position_scale:.3f}")

        # OPTIMIZED BUY LOGIC: Adjust for aggressive entry opportunities
        if weighted_score > self.aggressive_entry_threshold:
            original_scale = position_scale
            position_scale *= 1.2  # 20% boost for high-conviction setups
            logger.info(f"Aggressive entry boost applied - position scale increased from {original_scale:.3f} to {position_scale:.3f}")

        # Cap position size
        position_scale = min(position_scale, 1.0)
        logger.info(f"Final position scale (capped): {position_scale:.3f}")

        # Set position scale (the actual quantity will be calculated in the integration layer)
        decision.buy_quantity = 1  # Placeholder - actual quantity calculated in integration layer
        decision.buy_percentage = position_scale
        decision.reasoning += f" | ENHANCED POSITION SIZING ({position_scale:.1%})"

        return decision
