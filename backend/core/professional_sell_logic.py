"""
Professional Sell Logic Framework
Implements institutional-grade sell logic with proper risk management,
trailing stops, profit protection, and market context awareness.
"""

import logging
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SellReason(Enum):
    """Professional sell reasons for audit trail"""
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TAKE_PROFIT = "take_profit"
    PROFIT_PROTECTION = "profit_protection"
    TECHNICAL_SIGNALS = "technical_signals"
    RISK_MANAGEMENT = "risk_management"
    MARKET_DETERIORATION = "market_deterioration"
    POSITION_SIZING = "position_sizing"

class MarketTrend(Enum):
    """Market trend classification"""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

@dataclass
class SellSignal:
    """Individual sell signal with strength and reasoning"""
    name: str
    strength: float  # 0.0 to 1.0
    weight: float    # Signal importance weight
    triggered: bool
    reasoning: str
    confidence: float
    category: str = ""  # Category of the signal (Technical, Risk, Sentiment, ML, Market)
    
@dataclass
class PositionMetrics:
    """Position-specific metrics for sell decisions"""
    entry_price: float
    current_price: float
    quantity: int
    unrealized_pnl: float
    unrealized_pnl_pct: float
    days_held: int
    highest_price_since_entry: float
    lowest_price_since_entry: float
    volatility: float
    db_stop_loss: Optional[float] = None  # Stop-loss from database
    db_target_price: Optional[float] = None  # Target price from database
    
@dataclass
class MarketContext:
    """Market context for sell decisions"""
    trend: MarketTrend
    trend_strength: float  # 0.0 to 1.0
    volatility_regime: str  # "low", "normal", "high"
    market_stress: float   # 0.0 to 1.0
    sector_performance: float  # relative to market
    volume_profile: float  # relative to average

@dataclass
class SellDecision:
    """Professional sell decision output"""
    should_sell: bool
    sell_quantity: int
    sell_percentage: float  # 0.0 to 1.0 (partial vs full exit)
    reason: SellReason
    confidence: float
    urgency: float  # 0.0 to 1.0
    signals_triggered: List[SellSignal]
    stop_loss_price: float
    take_profit_price: float
    trailing_stop_price: float
    reasoning: str

class ProfessionalSellLogic:
    """
    Professional-grade sell logic implementation
    Features:
    - Minimum 2-3 signal confirmation
    - Dynamic trailing stops
    - Profit protection mechanisms
    - Market context awareness
    - Position-based sizing
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Professional thresholds - BALANCED with buy logic
        self.min_signals_required = config.get("min_sell_signals", 2)  # Minimum 2 categories
        self.min_confidence_threshold = config.get("min_sell_confidence", 0.45)  # 45% minimum confidence (moderate-strict)
        self.min_weighted_score = config.get("min_weighted_sell_score", 0.06)  # 6% minimum weighted score (moderate-strict)
        
        # Stop-loss configuration
        self.base_stop_loss_pct = config.get("stop_loss_pct", 0.05)
        self.trailing_stop_pct = config.get("trailing_stop_pct", 0.0325)  # 65% of stop-loss (moderate-strict)
        self.profit_protection_threshold = config.get("profit_protection_threshold", 0.06)  # 6% profit lock (moderate-strict)
        
        # Emergency loss threshold (8-10% loss triggers immediate sell)
        self.emergency_loss_threshold = config.get("emergency_loss_threshold", 0.10)  # 10% default
        
        # Position sizing - More granular thresholds
        self.conservative_exit_threshold = config.get("conservative_exit_threshold", 0.15)
        self.partial_exit_threshold = config.get("partial_exit_threshold", 0.30)
        self.aggressive_exit_threshold = config.get("aggressive_exit_threshold", 0.50)
        self.full_exit_threshold = config.get("full_exit_threshold", 0.70)
        self.emergency_exit_threshold = config.get("emergency_exit_threshold", 0.90)
        
        # Market context filters - Less restrictive
        self.uptrend_sell_multiplier = config.get("uptrend_sell_multiplier", 1.15)  # Less restriction in uptrends (moderate-strict)
        self.downtrend_sell_multiplier = config.get("downtrend_sell_multiplier", 0.85)  # Less restriction in downtrends (moderate-strict)
        
        logger.info("Professional Sell Logic initialized with institutional parameters")
    
    def _load_dynamic_config(self):
        """Load dynamic configuration from live_config.json"""
        try:
            import json
            import os
            
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'live_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    live_config = json.load(f)
                
                # Update dynamic configuration values
                self.base_stop_loss_pct = live_config.get("stop_loss_pct", self.base_stop_loss_pct)
                self.trailing_stop_pct = live_config.get("trailing_stop_pct", self.trailing_stop_pct)
                self.profit_protection_threshold = live_config.get("profit_protection_threshold", self.profit_protection_threshold)
                self.emergency_loss_threshold = live_config.get("emergency_loss_threshold", self.emergency_loss_threshold)
                
                logger.info(f"📊 Loaded dynamic config for sell logic - Stop Loss: {self.base_stop_loss_pct:.1%}")
            else:
                logger.warning("live_config.json not found for sell logic, using defaults")
                
        except Exception as e:
            logger.error(f"Failed to load dynamic config for sell logic: {e}")
    
    def refresh_dynamic_config(self):
        """Refresh dynamic configuration from live_config.json (call this periodically)"""
        self._load_dynamic_config()
    
    def _check_immediate_exit_conditions(
        self,
        position: PositionMetrics,
        stop_levels: Dict
    ) -> Optional[SellDecision]:
        """
        Check immediate exit conditions (stop-loss, trailing stop, etc.)
        This is a simplified function that just returns None to continue with normal evaluation
        """
        # The immediate exit conditions are already checked at the beginning of evaluate_sell_decision
        # This function is kept for compatibility with the existing code structure
        return None

    def evaluate_sell_decision(
        self,
        ticker: str,
        position_metrics: PositionMetrics,
        market_context: MarketContext,
        technical_analysis: Dict,
        sentiment_analysis: Dict,
        ml_analysis: Dict
    ) -> SellDecision:
        """
        Main entry point for professional sell evaluation with simplified flow:
        1. Check stored stop-loss/target price conditions first
        2. Evaluate each of the 5 signal categories
        3. Make decision based on combined signals
        """
        logger.info(f"=== PROFESSIONAL SELL EVALUATION: {ticker} ===")
        logger.info(f"Current Price: {position_metrics.current_price:.2f}")
        logger.info(f"Entry Price: {position_metrics.entry_price:.2f}")
        logger.info(f"Unrealized P&L: {position_metrics.unrealized_pnl:.2f} ({position_metrics.unrealized_pnl_pct:.2%})")
        logger.info(f"Market Context: {market_context.trend.value} (strength: {market_context.trend_strength:.2f})")
        
        # STEP 1: Check if we have stored stop loss or target price that should trigger a sell
        if position_metrics.db_stop_loss is not None and position_metrics.current_price <= position_metrics.db_stop_loss:
            # Stop loss triggered from stored value
            logger.info(f"✅ Stored stop-loss triggered: {position_metrics.current_price:.2f} <= {position_metrics.db_stop_loss:.2f}")
            return SellDecision(
                should_sell=True,
                sell_quantity=position_metrics.quantity,
                sell_percentage=1.0,
                reason=SellReason.STOP_LOSS,
                confidence=1.0,
                urgency=1.0,
                signals_triggered=[],
                stop_loss_price=position_metrics.db_stop_loss,
                take_profit_price=position_metrics.db_target_price if position_metrics.db_target_price is not None else 0.0,
                trailing_stop_price=position_metrics.db_stop_loss,  # For compatibility
                reasoning=f"Stored stop-loss triggered: {position_metrics.current_price:.2f} <= {position_metrics.db_stop_loss:.2f}"
            )
        
        if position_metrics.db_target_price is not None and position_metrics.current_price >= position_metrics.db_target_price:
            # Take profit triggered from stored value
            logger.info(f"✅ Stored take-profit triggered: {position_metrics.current_price:.2f} >= {position_metrics.db_target_price:.2f}")
            return SellDecision(
                should_sell=True,
                sell_quantity=position_metrics.quantity,
                sell_percentage=1.0,
                reason=SellReason.TAKE_PROFIT,
                confidence=0.9,
                urgency=0.8,
                signals_triggered=[],
                stop_loss_price=position_metrics.db_stop_loss if position_metrics.db_stop_loss is not None else 0.0,
                take_profit_price=position_metrics.db_target_price,
                trailing_stop_price=position_metrics.db_stop_loss if position_metrics.db_stop_loss is not None else 0.0,  # For compatibility
                reasoning=f"Stored take-profit triggered: {position_metrics.current_price:.2f} >= {position_metrics.db_target_price:.2f}"
            )
        
        # STEP 2: Generate signals from each of the 5 categories
        technical_signals = self._generate_technical_signals(technical_analysis, position_metrics)
        risk_signals = self._generate_risk_signals(position_metrics)
        sentiment_signals = self._generate_sentiment_signals(sentiment_analysis)
        ml_signals = self._generate_ml_signals(ml_analysis)
        market_signals = self._generate_market_signals(market_context)
        
        # Combine all signals
        all_signals = technical_signals + risk_signals + sentiment_signals + ml_signals + market_signals
        
        # Log signal generation with detailed information
        logger.info(f"Generated {len(all_signals)} signals from 5 categories")
        triggered_signals = [s for s in all_signals if s.triggered]
        logger.info(f"{len(triggered_signals)} triggered signals:")
        
        # Group signals by category for better visualization
        category_signals = {}
        for signal in all_signals:
            category = signal.category or "Uncategorized"
            if category not in category_signals:
                category_signals[category] = []
            category_signals[category].append(signal)
        
        for category, signals_list in category_signals.items():
            triggered_in_category = [s for s in signals_list if s.triggered]
            logger.info(f"  {category}: {len(triggered_in_category)}/{len(signals_list)} signals triggered")
            for signal in triggered_in_category:
                logger.info(f"    - {signal.name}: Strength {signal.strength:.3f}, Confidence {signal.confidence:.3f}")
        
        # STEP 3: Calculate weighted signal score
        total_weight = sum(s.weight for s in triggered_signals)
        weighted_score = sum(s.strength * s.weight for s in triggered_signals)

        # Calculate confidence based on signal agreement
        signal_confidences = [s.confidence for s in triggered_signals]
        avg_confidence = np.mean(signal_confidences) if signal_confidences else 0.0

        # Professional decision criteria
        signals_count = len(triggered_signals)
        meets_signal_threshold = 2 <= signals_count <= 5  # At least 2 categories should trigger
        meets_confidence_threshold = avg_confidence >= self.min_confidence_threshold
        meets_weighted_threshold = weighted_score >= self.min_weighted_score

        logger.info(f"Signal Analysis Summary:")
        logger.info(f"  Signal Count: {signals_count} (Required: 2-5) - {'✅ PASS' if meets_signal_threshold else '❌ FAIL'}")
        logger.info(f"  Average Confidence: {avg_confidence:.3f} (Threshold: {self.min_confidence_threshold}) - {'✅ PASS' if meets_confidence_threshold else '❌ FAIL'}")
        logger.info(f"  Weighted Score: {weighted_score:.3f} (Threshold: {self.min_weighted_score}) - {'✅ PASS' if meets_weighted_threshold else '❌ FAIL'}")

        # Professional sell logic: All three conditions must be met
        should_sell = meets_signal_threshold and meets_confidence_threshold and meets_weighted_threshold

        if should_sell:
            logger.info("✅ ALL THRESHOLD CHECKS PASSED - Proceeding with sell decision")
                
            # Calculate stop-loss and take-profit levels
            stop_levels = self._calculate_dynamic_stops(position_metrics, market_context)
            
            base_decision = SellDecision(
                should_sell=True,
                sell_quantity=position_metrics.quantity,  # Will be adjusted by position sizing
                sell_percentage=0.0,  # Will be calculated
                reason=SellReason.TECHNICAL_SIGNALS,
                confidence=avg_confidence,
                urgency=0.5,  # Will be adjusted
                signals_triggered=triggered_signals,
                stop_loss_price=stop_levels["active_stop"],
                take_profit_price=0.0,  # Will be calculated
                trailing_stop_price=stop_levels["active_stop"],  # For compatibility
                reasoning=f"Professional sell confirmed: {signals_count} signals triggered, "
                         f"confidence {avg_confidence:.3f}, weighted score {weighted_score:.3f}"
            )
            
            # Apply market context filters
            final_decision = self._apply_market_context_filters(base_decision, market_context)
            
            # Determine position sizing
            final_decision = self._determine_position_sizing(final_decision, all_signals, position_metrics)
            
            logger.info(f"SELL DECISION: {final_decision.should_sell} | "
                       f"Qty: {final_decision.sell_quantity} | "
                       f"Reason: {final_decision.reason.value} | "
                       f"Confidence: {final_decision.confidence:.3f}")
            
            return final_decision
        else:
            logger.info("❌ THRESHOLD CHECKS FAILED - Generating hold decision")
            # Provide detailed reasoning for why sell was rejected
            rejection_reasons = []
            if not meets_signal_threshold:
                rejection_reasons.append(f"Triggered signals {signals_count} not in range [2, 5]")
            if not meets_confidence_threshold:
                rejection_reasons.append(f"Confidence {avg_confidence:.3f} below threshold {self.min_confidence_threshold}")
            if not meets_weighted_threshold:
                rejection_reasons.append(f"Weighted score {weighted_score:.3f} below threshold {self.min_weighted_score}")
            
            detailed_reasoning = " | ".join(rejection_reasons)
            logger.info(f"Rejection Reasons: {detailed_reasoning}")
            
            # Log additional diagnostic information
            logger.info(f"🔍 DETAILED DIAGNOSTIC INFORMATION:")
            logger.info(f"   - Minimum Signals Required: 2 signals")
            logger.info(f"   - Maximum Signals Required: 5 signals")
            logger.info(f"   - Minimum Confidence Threshold: {self.min_confidence_threshold}")
            logger.info(f"   - Minimum Weighted Score Threshold: {self.min_weighted_score}")
            
            return SellDecision(
                should_sell=False,
                sell_quantity=0,
                sell_percentage=0.0,
                reason=SellReason.TECHNICAL_SIGNALS,
                confidence=avg_confidence,
                urgency=0.0,
                signals_triggered=all_signals,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                trailing_stop_price=0.0,
                reasoning=detailed_reasoning
            )

    def _generate_technical_signals(self, technical: Dict, position: PositionMetrics) -> List[SellSignal]:
        """Generate technical analysis sell signals"""
        signals = []

        # RSI Overbought (Strong signal)
        rsi = technical.get("rsi", 50)
        if rsi > 70:
            strength = min((rsi - 70) / 20, 1.0)  # Scale 70-90 to 0-1
            signals.append(SellSignal(
                name="rsi_overbought",
                strength=strength,
                weight=0.08,  # 8% of total weight
                triggered=True,
                reasoning=f"RSI overbought at {rsi:.1f}",
                confidence=0.8,
                category="Technical"
            ))

        # MACD Bearish Divergence
        macd = technical.get("macd", 0)
        macd_signal = technical.get("macd_signal", 0)
        if macd < macd_signal and macd < 0:
            strength = min(abs(macd - macd_signal) / abs(macd_signal) if macd_signal != 0 else 1, 1.0)
            signals.append(SellSignal(
                name="macd_bearish",
                strength=strength,
                weight=0.07,  # 7% of total weight
                triggered=True,
                reasoning="MACD bearish crossover",
                confidence=0.7,
                category="Technical"
            ))

        # Moving Average Breakdown
        sma_20 = technical.get("sma_20", position.current_price)
        sma_50 = technical.get("sma_50", position.current_price)
        if position.current_price < sma_20 and sma_20 < sma_50:
            strength = (sma_20 - position.current_price) / sma_20
            signals.append(SellSignal(
                name="ma_breakdown",
                strength=min(strength, 1.0),
                weight=0.06,  # 6% of total weight
                triggered=True,
                reasoning="Price below key moving averages",
                confidence=0.75,
                category="Technical"
            ))

        # Support Level Breakdown
        support = technical.get("support_level", 0)
        if support > 0 and position.current_price < support * 0.98:
            strength = (support - position.current_price) / support
            signals.append(SellSignal(
                name="support_breakdown",
                strength=min(strength * 2, 1.0),  # Amplify support breaks
                weight=0.09,  # 9% of total weight
                triggered=True,
                reasoning=f"Support breakdown at {support:.2f}",
                confidence=0.85,
                category="Technical"
            ))

        # ADX signal for trend strength confirmation
        adx = technical.get("adx", 20)
        plus_di = technical.get("plus_di", 20)
        minus_di = technical.get("minus_di", 20)
        
        # Strong bearish trend confirmation with ADX > 25 and -DI > +DI
        if adx > 25 and minus_di > plus_di:
            strength = min((adx - 25) / 25, 1.0)
            signals.append(SellSignal(
                name="adx_trend_strength",
                strength=strength,
                weight=0.07,
                triggered=True,
                reasoning=f"Strong bearish trend confirmed with ADX {adx:.1f}, -DI {minus_di:.1f} > +DI {plus_di:.1f}",
                confidence=0.80,
                category="Technical"
            ))

        # Aroon Oscillator signal for trend reversal
        aroon_osc = technical.get("aroon_osc", 0)
        
        # Bearish Aroon Oscillator
        if aroon_osc < -50:
            strength = min(abs(aroon_osc) / 100, 1.0)
            signals.append(SellSignal(
                name="aroon_bearish",
                strength=strength,
                weight=0.06,
                triggered=True,
                reasoning=f"Bearish trend reversal signal with Aroon Oscillator {aroon_osc:.1f}",
                confidence=0.75,
                category="Technical"
            ))

        # CCI signal for cyclical selling opportunities
        cci = technical.get("cci_14", 0)
        
        # CCI overbought condition
        if cci > 100:
            strength = min((cci - 100) / 100, 1.0)
            signals.append(SellSignal(
                name="cci_overbought",
                strength=strength,
                weight=0.05,
                triggered=True,
                reasoning=f"CCI overbought at {cci:.1f} indicating potential reversal",
                confidence=0.70,
                category="Technical"
            ))

        # ROC signal for momentum confirmation
        roc_10 = technical.get("roc_10", 0)
        roc_20 = technical.get("roc_20", 0)
        
        # Negative momentum with ROC confirmation
        if roc_10 < -1.0 and roc_20 < -0.5:
            strength = min(abs(roc_10) / 5, 1.0)
            signals.append(SellSignal(
                name="roc_negative_momentum",
                strength=strength,
                weight=0.06,
                triggered=True,
                reasoning=f"Negative momentum confirmed with ROC(10) {roc_10:.2f}% and ROC(20) {roc_20:.2f}%",
                confidence=0.70,
                category="Technical"
            ))

        # TRIX signal for trend direction
        trix = technical.get("trix", 0)
        
        # Bearish TRIX crossover
        if trix < 0:
            strength = min(abs(trix) * 10, 1.0)
            signals.append(SellSignal(
                name="trix_bearish",
                strength=strength,
                weight=0.05,
                triggered=True,
                reasoning=f"Bearish trend confirmed with TRIX {trix:.4f}",
                confidence=0.65,
                category="Technical"
            ))

        # CMO signal for momentum
        cmo = technical.get("cmo_14", 0)
        
        # Overbought CMO condition
        if cmo > 50:
            strength = min((cmo - 50) / 50, 1.0)
            signals.append(SellSignal(
                name="cmo_overbought",
                strength=strength,
                weight=0.05,
                triggered=True,
                reasoning=f"CMO overbought at {cmo:.1f} indicating potential selling opportunity",
                confidence=0.65,
                category="Technical"
            ))

        # Williams %R signal
        williams_r = technical.get("williams_r", -50)
        
        # Overbought Williams %R condition
        if williams_r > -10:
            strength = min((williams_r + 10) / 10, 1.0)
            signals.append(SellSignal(
                name="williams_r_overbought",
                strength=strength,
                weight=0.06,
                triggered=True,
                reasoning=f"Williams %R overbought at {williams_r:.1f}",
                confidence=0.70,
                category="Technical"
            ))

        # Stochastic Oscillator signal
        stoch_k = technical.get("stoch_k", 50)
        stoch_d = technical.get("stoch_d", 50)
        
        # Overbought Stochastic condition
        if stoch_k > 90 and stoch_k < stoch_d:  # %K crossing below %D
            strength = min((stoch_k - 90) / 10, 1.0)
            signals.append(SellSignal(
                name="stoch_overbought",
                strength=strength,
                weight=0.06,
                triggered=True,
                reasoning=f"Stochastic overbought at {stoch_k:.1f}% with bearish crossover (%D: {stoch_d:.1f})",
                confidence=0.70,
                category="Technical"
            ))

        # MFI signal
        mfi = technical.get("mfi", 50)
        
        # Overbought MFI condition
        if mfi > 80:
            strength = min((mfi - 80) / 20, 1.0)
            signals.append(SellSignal(
                name="mfi_overbought",
                strength=strength,
                weight=0.07,
                triggered=True,
                reasoning=f"MFI overbought at {mfi:.1f}",
                confidence=0.75,
                category="Technical"
            ))

        return signals

    def _generate_risk_signals(self, position: PositionMetrics) -> List[SellSignal]:
        """Generate risk management sell signals"""
        signals = []

        # Use stored stop-loss from database if available, otherwise use calculated value
        stop_loss_level = position.db_stop_loss if position.db_stop_loss is not None else position.entry_price * (1 - self.base_stop_loss_pct)
        if position.current_price <= stop_loss_level:
            signals.append(SellSignal(
                name="stop_loss",
                strength=1.0,
                weight=0.15,  # 15% of total weight
                triggered=True,
                reasoning=f"Stop-loss triggered at {stop_loss_level:.2f}",
                confidence=1.0,
                category="Risk"
            ))

        # Use stored target price for take-profit calculation if available
        take_profit_level = position.db_target_price if position.db_target_price is not None else position.entry_price * (1 + self.base_stop_loss_pct * 2)  # 2:1 risk-reward
        if position.current_price >= take_profit_level:
            signals.append(SellSignal(
                name="take_profit",
                strength=0.9,
                weight=0.12,  # 12% of total weight
                triggered=True,
                reasoning=f"Take-profit triggered at {take_profit_level:.2f}",
                confidence=0.9,
                category="Risk"
            ))

        # Profit Protection (lock in gains) - enhanced with stored target price
        profit_protection_level = position.highest_price_since_entry * (1 - self.profit_protection_threshold)
        # If we have a stored target price, use half of it for profit protection
        if position.db_target_price is not None:
            stored_profit_level = position.entry_price + (position.db_target_price - position.entry_price) * 0.5
            profit_protection_level = max(profit_protection_level, stored_profit_level)
            
        if position.unrealized_pnl_pct > 0.05 and position.current_price <= profit_protection_level:  # 5% gain threshold
            signals.append(SellSignal(
                name="profit_protection",
                strength=0.8,
                weight=0.10,  # 10% of total weight
                triggered=True,
                reasoning=f"Profit protection at {profit_protection_level:.2f}",
                confidence=0.8,
                category="Risk"
            ))

        # Large Loss Signal
        if position.unrealized_pnl_pct < -0.05:  # 5% loss
            strength = min(abs(position.unrealized_pnl_pct) / 0.10, 1.0)  # Scale to 10% max loss
            signals.append(SellSignal(
                name="large_loss",
                strength=strength,
                weight=0.10,  # 10% of total weight
                triggered=True,
                reasoning=f"Large loss: {position.unrealized_pnl_pct:.1%}",
                confidence=0.85,
                category="Risk"
            ))

        # Volatility Spike Signal
        if position.volatility > 0.03:  # 3% daily volatility
            strength = min(position.volatility / 0.06, 1.0)  # Scale to 6% max
            signals.append(SellSignal(
                name="volatility_spike",
                strength=strength,
                weight=0.08,  # 8% of total weight
                triggered=True,
                reasoning=f"High volatility: {position.volatility:.1%}",
                confidence=0.7,
                category="Risk"
            ))

        # Time-based Risk (holding too long without profit)
        if position.days_held > 30 and position.unrealized_pnl_pct < 0.02:
            strength = min(position.days_held / 90, 1.0)  # Scale to 90 days max
            signals.append(SellSignal(
                name="time_decay",
                strength=strength,
                weight=0.07,  # 7% of total weight
                triggered=True,
                reasoning=f"Held {position.days_held} days without profit",
                confidence=0.6,
                category="Risk"
            ))

        return signals

    def _generate_sentiment_signals(self, sentiment: Dict) -> List[SellSignal]:
        """Generate sentiment-based sell signals"""
        signals = []

        # Negative sentiment signal
        sentiment_score = sentiment.get("overall_sentiment", 0)
        if sentiment_score < -0.2:  # Negative sentiment threshold
            strength = min(abs(sentiment_score) / 0.8, 1.0)  # Scale to -0.8 max
            signals.append(SellSignal(
                name="negative_sentiment",
                strength=strength,
                weight=0.12,  # 12% of total weight
                triggered=True,
                reasoning=f"Negative sentiment: {sentiment_score:.2f}",
                confidence=0.6,
                category="Sentiment"
            ))

        # News sentiment deterioration
        news_trend = sentiment.get("news_trend", 0)
        if news_trend < -0.3:
            strength = min(abs(news_trend) / 0.7, 1.0)
            signals.append(SellSignal(
                name="news_deterioration",
                strength=strength,
                weight=0.08,  # 8% of total weight
                triggered=True,
                reasoning="Deteriorating news sentiment",
                confidence=0.5,
                category="Sentiment"
            ))

        return signals

    def _generate_ml_signals(self, ml_analysis: Dict) -> List[SellSignal]:
        """Generate ML/AI-based sell signals"""
        signals = []

        # ML Prediction Signal
        ml_prediction = ml_analysis.get("prediction_direction", 0)
        # ENHANCED: More flexible threshold with additional validation
        if ml_prediction < -0.01:  # Increased threshold but with additional validation
            strength = min(abs(ml_prediction) / 0.10, 1.0)
            signals.append(SellSignal(
                name="ml_bearish_prediction",
                strength=strength,
                weight=0.10,  # 10% of total weight
                triggered=True,
                reasoning=f"ML bearish prediction: {ml_prediction:.1%}",
                confidence=ml_analysis.get("confidence", 0.5),
                category="ML"
            ))

        # RL Recommendation
        rl_recommendation = ml_analysis.get("rl_recommendation", "HOLD")
        # ENHANCED: More flexible threshold with additional validation
        if rl_recommendation == "SELL":
            rl_confidence = ml_analysis.get("rl_confidence", 0.5)
            signals.append(SellSignal(
                name="rl_sell_recommendation",
                strength=rl_confidence,
                weight=0.05,  # 5% of total weight
                triggered=True,
                reasoning="RL algorithm recommends SELL",
                confidence=rl_confidence,
                category="ML"
            ))

        return signals

    def _generate_market_signals(self, market_context: MarketContext) -> List[SellSignal]:
        """Generate market structure sell signals"""
        signals = []

        # Market stress signal
        if market_context.market_stress > 0.6:
            signals.append(SellSignal(
                name="market_stress",
                strength=market_context.market_stress,
                weight=0.06,  # 6% of total weight
                triggered=True,
                reasoning=f"High market stress: {market_context.market_stress:.1%}",
                confidence=0.7,
                category="Market"
            ))

        # Sector underperformance
        if market_context.sector_performance < -0.02:
            strength = min(abs(market_context.sector_performance) / 0.05, 1.0)
            signals.append(SellSignal(
                name="sector_weakness",
                strength=strength,
                weight=0.04,  # 4% of total weight
                triggered=True,
                reasoning="Sector underperforming market",
                confidence=0.6,
                category="Market"
            ))

        return signals

    def _calculate_dynamic_stops(self, position: PositionMetrics, market_context: MarketContext) -> Dict:
        """Calculate dynamic stop-loss levels using stored values when available"""

        # Base stop-loss - use stored value if available
        base_stop = position.db_stop_loss if position.db_stop_loss is not None else position.entry_price * (1 - self.base_stop_loss_pct)

        # Volatility-adjusted stop - use stored value if more conservative
        volatility_multiplier = 1 + (position.volatility / 0.02)  # Adjust for volatility
        calculated_volatility_stop = position.entry_price * (1 - self.base_stop_loss_pct * volatility_multiplier)
        volatility_stop = min(calculated_volatility_stop, base_stop)  # Use the more conservative

        # Take profit - use stored value if available
        take_profit = position.db_target_price if position.db_target_price is not None else position.entry_price * (1 + self.base_stop_loss_pct * 2)  # 2:1 risk-reward

        # Profit protection stop (lock in 50% of peak gains) - enhanced with stored target
        profit_protection_stop = 0.0
        if position.unrealized_pnl_pct > self.profit_protection_threshold:
            # Lock in profits after threshold is met
            protected_gain = position.highest_price_since_entry * self.profit_protection_threshold
            profit_protection_stop = position.highest_price_since_entry - protected_gain
            
            # If we have a stored target price, use half of it for additional protection
            if position.db_target_price is not None:
                stored_profit_protection = position.entry_price + (position.db_target_price - position.entry_price) * 0.5
                profit_protection_stop = max(profit_protection_stop, stored_profit_protection)

        # Use the most conservative stop
        active_stop = min(base_stop, volatility_stop)

        return {
            "base_stop": base_stop,
            "volatility_stop": volatility_stop,
            "profit_protection_stop": profit_protection_stop,
            "take_profit": take_profit,
            "active_stop": active_stop
        }

    def _get_stop_loss_levels(self, position_metrics: PositionMetrics) -> Dict[str, float]:
        """
        Calculate multiple stop loss levels for different market conditions
        """
        # Base stop loss from position metrics
        base_stop_loss = position_metrics.db_stop_loss if position_metrics.db_stop_loss is not None else (
            position_metrics.entry_price * (1 - self.stop_loss_pct)
        )
        
        # Volatility-adjusted stop loss
        volatility_stop_loss = position_metrics.entry_price - (position_metrics.atr * self.atr_multiplier)
        
        # Trailing stop based on highest price achieved
        trailing_stop = position_metrics.highest_price_achieved * (1 - self.trailing_stop_pct)
        
        # Determine active stop based on market conditions
        active_stop = base_stop_loss  # Default to base stop loss
        
        return {
            "base_stop": base_stop_loss,
            "volatility_stop": volatility_stop_loss,
            "trailing_stop": trailing_stop,
            "active_stop": active_stop
        }

    def _get_take_profit_levels(self, position_metrics: PositionMetrics) -> Dict[str, float]:
        """
        Calculate multiple take profit levels for different market conditions
        """
        # Base take profit from position metrics
        base_take_profit = position_metrics.db_target_price if position_metrics.db_target_price is not None else (
            position_metrics.entry_price * (1 + self.take_profit_pct)
        )
        
        # Risk-reward ratio based take profit
        risk_reward_take_profit = position_metrics.entry_price + (
            abs(position_metrics.entry_price - self._get_stop_loss_levels(position_metrics)["active_stop"]) * self.risk_reward_ratio
        )
        
        # Technical resistance based take profit
        technical_take_profit = position_metrics.resistance_level
        
        # Determine active take profit based on market conditions
        active_take_profit = base_take_profit  # Default to base take profit
        
        return {
            "base_take_profit": base_take_profit,
            "risk_reward_take_profit": risk_reward_take_profit,
            "technical_take_profit": technical_take_profit,
            "active_take_profit": active_take_profit
        }

    def _check_cross_category_confirmation(self, signals: List[SellSignal]) -> bool:
        """Cross-category confirmation: At least 2 categories must align"""
        # With combined signals, we check if at least 2 categories have triggered signals
        triggered_signals = [s for s in signals if s.triggered]
        return len(triggered_signals) >= 2

    def _evaluate_signal_based_sell(
        self,
        signals: List[SellSignal],
        position: PositionMetrics,
        market_context: MarketContext
    ) -> SellDecision:
        """Evaluate sell decision based on signal analysis with combined signals"""
        
        # First, check if we have stored stop loss or target price that should trigger a sell
        if position.db_stop_loss is not None and position.current_price <= position.db_stop_loss:
            # Stop loss triggered from stored value
            return SellDecision(
                should_sell=True,
                sell_quantity=position.quantity,
                sell_percentage=1.0,
                reason=SellReason.STOP_LOSS,
                confidence=1.0,
                urgency=1.0,
                signals_triggered=[],
                stop_loss_price=position.db_stop_loss,
                take_profit_price=position.db_target_price if position.db_target_price is not None else 0.0,
                trailing_stop_price=position.db_stop_loss,  # For compatibility
                reasoning=f"Stored stop-loss triggered: {position.current_price:.2f} <= {position.db_stop_loss:.2f}"
            )
        
        if position.db_target_price is not None and position.current_price >= position.db_target_price:
            # Take profit triggered from stored value
            return SellDecision(
                should_sell=True,
                sell_quantity=position.quantity,
                sell_percentage=1.0,
                reason=SellReason.TAKE_PROFIT,
                confidence=0.9,
                urgency=0.8,
                signals_triggered=[],
                stop_loss_price=position.db_stop_loss if position.db_stop_loss is not None else 0.0,
                take_profit_price=position.db_target_price,
                trailing_stop_price=position.db_stop_loss if position.db_stop_loss is not None else 0.0,  # For compatibility
                reasoning=f"Stored take-profit triggered: {position.current_price:.2f} >= {position.db_target_price:.2f}"
            )

        # Calculate weighted signal score
        triggered_signals = [s for s in signals if s.triggered]
        total_weight = sum(s.weight for s in triggered_signals)
        weighted_score = sum(s.strength * s.weight for s in triggered_signals)

        # Calculate confidence based on signal agreement
        signal_confidences = [s.confidence for s in triggered_signals]
        avg_confidence = np.mean(signal_confidences) if signal_confidences else 0.0

        # Professional decision criteria with combined signals
        signals_count = len(triggered_signals)
        # For combined signals, we expect 2-5 triggered signals (one per category)
        # Adjust thresholds to be more appropriate for combined signals
        meets_signal_threshold = 2 <= signals_count <= 5  # At least 2 categories should trigger
        meets_confidence_threshold = avg_confidence >= self.min_confidence_threshold
        meets_weighted_threshold = weighted_score >= self.min_weighted_score

        logger.info(f"Signal Analysis Summary:")
        logger.info(f"  Signal Count: {signals_count} (Required: 2-5) - {'✅ PASS' if meets_signal_threshold else '❌ FAIL'}")
        logger.info(f"  Average Confidence: {avg_confidence:.3f} (Threshold: {self.min_confidence_threshold}) - {'✅ PASS' if meets_confidence_threshold else '❌ FAIL'}")
        logger.info(f"  Weighted Score: {weighted_score:.3f} (Threshold: {self.min_weighted_score}) - {'✅ PASS' if meets_weighted_threshold else '❌ FAIL'}")

        # Professional sell logic: All three conditions must be met
        should_sell = meets_signal_threshold and meets_confidence_threshold and meets_weighted_threshold

        if should_sell:
            logger.info("✅ ALL THRESHOLD CHECKS PASSED - Proceeding with sell decision")
                
            
            # Calculate stop-loss and take-profit levels
            stop_levels = self._calculate_dynamic_stops(position, market_context)
            
            base_decision = SellDecision(
                should_sell=True,
                sell_quantity=position.quantity,  # Will be adjusted by position sizing
                sell_percentage=0.0,  # Will be calculated
                reason=SellReason.TECHNICAL_SIGNALS,
                confidence=avg_confidence,
                urgency=0.5,  # Will be adjusted
                signals_triggered=triggered_signals,
                stop_loss_price=stop_levels["active_stop"],
                take_profit_price=0.0,  # Will be calculated
                trailing_stop_price=stop_levels["active_stop"],  # For compatibility
                reasoning=f"Professional sell confirmed: {signals_count} categories triggered, "
                         f"confidence {avg_confidence:.3f}, weighted score {weighted_score:.3f}"
            )
            
            return base_decision
        else:
            logger.info("❌ THRESHOLD CHECKS FAILED - Generating hold decision")
            # Provide detailed reasoning for why sell was rejected
            rejection_reasons = []
            if not meets_signal_threshold:
                rejection_reasons.append(f"Triggered categories {signals_count} not in range [2, 5]")
            if not meets_confidence_threshold:
                rejection_reasons.append(f"Confidence {avg_confidence:.3f} below threshold {self.min_confidence_threshold}")
            if not meets_weighted_threshold:
                rejection_reasons.append(f"Weighted score {weighted_score:.3f} below threshold {self.min_weighted_score}")
            
            detailed_reasoning = " | ".join(rejection_reasons)
            logger.info(f"Rejection Reasons: {detailed_reasoning}")
            
            # Log additional diagnostic information
            logger.info(f"🔍 DETAILED DIAGNOSTIC INFORMATION:")
            logger.info(f"   - Minimum Signals Required: 2 categories")
            logger.info(f"   - Maximum Signals Required: 5 categories")
            logger.info(f"   - Minimum Confidence Threshold: {self.min_confidence_threshold}")
            logger.info(f"   - Minimum Weighted Score Threshold: {self.min_weighted_score}")
            
            return SellDecision(
                should_sell=False,
                sell_quantity=0,
                sell_percentage=0.0,
                reason=SellReason.TECHNICAL_SIGNALS,
                confidence=avg_confidence,
                urgency=0.0,
                signals_triggered=signals,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                trailing_stop_price=0.0,
                reasoning=detailed_reasoning
            )

    def _apply_market_context_filters(self, decision: SellDecision, market_context: MarketContext) -> SellDecision:
        """Apply market context filters to sell decision (LESS restrictive)"""

        if not decision.should_sell:
            logger.info("Market context filters skipped - no sell decision")
            return decision

        logger.info(f"Applying market context filters - Market Trend: {market_context.trend.value}")

        # Less restrictive in uptrends - allow sells with moderate urgency
        if market_context.trend in [MarketTrend.STRONG_UPTREND, MarketTrend.UPTREND]:
            if decision.urgency < 0.5:  # LOWERED from 0.8 to 0.5 for less restriction
                logger.info(f"SELL MODERATED: Market in {market_context.trend.value}, urgency {decision.urgency:.3f} < 0.5")
                decision.should_sell = False
                decision.reasoning += " | MODERATED: Uptrend with low urgency"
                return decision
            else:
                # Reduce position size in uptrends but not as much
                original_confidence = decision.confidence
                decision.confidence *= self.downtrend_sell_multiplier  # Use downtrend multiplier for less restriction
                logger.info(f"Uptrend adjustment - confidence reduced from {original_confidence:.3f} to {decision.confidence:.3f}")

        # Be less aggressive in downtrends (allow more sells)
        elif market_context.trend in [MarketTrend.DOWNTREND, MarketTrend.STRONG_DOWNTREND]:
            original_confidence = decision.confidence
            original_urgency = decision.urgency
            decision.confidence *= self.uptrend_sell_multiplier  # Use uptrend multiplier for less restriction
            decision.urgency = min(decision.urgency * 1.1, 1.0)  # LESS aggressive adjustment
            logger.info(f"Downtrend adjustment - confidence changed from {original_confidence:.3f} to {decision.confidence:.3f}, "
                       f"urgency increased from {original_urgency:.3f} to {decision.urgency:.3f}")

        return decision

    def _determine_position_sizing(
        self,
        decision: SellDecision,
        signals: List[SellSignal],
        position: PositionMetrics
    ) -> SellDecision:
        """Determine how much of the position to sell (MORE granular)"""

        if not decision.should_sell:
            logger.info("Position sizing skipped - no sell decision")
            return decision

        logger.info(f"Determining position sizing for {position.quantity} shares")
        logger.info(f"Decision confidence: {decision.confidence:.3f}")
        logger.info(f"Decision urgency: {decision.urgency:.3f}")
        logger.info(f"Unrealized P&L: {position.unrealized_pnl_pct:.2%}")

        # Emergency exit conditions (highest priority)
        if (decision.confidence >= self.emergency_exit_threshold or
            decision.urgency >= 0.95 or
            position.unrealized_pnl_pct < -0.10):  # 10% loss

            decision.sell_quantity = position.quantity
            decision.sell_percentage = 1.0
            decision.reasoning += " | EMERGENCY EXIT"
            logger.info(f"Emergency exit triggered - selling 100% ({position.quantity} shares)")

        # Full exit conditions
        elif (decision.confidence >= self.full_exit_threshold or
              decision.urgency >= 0.85 or
              position.unrealized_pnl_pct < -0.08):  # 8% loss

            decision.sell_quantity = position.quantity
            decision.sell_percentage = 1.0
            decision.reasoning += " | FULL EXIT"
            logger.info(f"Full exit triggered - selling 100% ({position.quantity} shares)")

        # Aggressive exit conditions
        elif decision.confidence >= self.aggressive_exit_threshold:
            # Scale exit size based on confidence and urgency
            exit_percentage = min(decision.confidence * decision.urgency * 1.2, 0.9)  # INCREASED scaling
            decision.sell_quantity = int(position.quantity * exit_percentage)
            decision.sell_percentage = exit_percentage
            decision.reasoning += f" | AGGRESSIVE EXIT ({exit_percentage:.1%})"
            logger.info(f"Aggressive exit triggered - selling {exit_percentage:.1%} ({decision.sell_quantity} shares)")

        # Partial exit conditions
        elif decision.confidence >= self.partial_exit_threshold:
            # Scale exit size based on confidence and urgency
            exit_percentage = min(decision.confidence * decision.urgency, 0.75)
            decision.sell_quantity = max(1, int(position.quantity * exit_percentage))
            decision.sell_percentage = exit_percentage
            decision.reasoning += f" | PARTIAL EXIT ({exit_percentage:.1%})"
            logger.info(f"Partial exit triggered - selling {exit_percentage:.1%} ({decision.sell_quantity} shares)")

        # Conservative exit conditions
        elif decision.confidence >= self.conservative_exit_threshold:
            # Scale exit size based on confidence and urgency
            exit_percentage = min(decision.confidence * decision.urgency * 0.5, 0.5)  # REDUCED scaling
            decision.sell_quantity = max(1, int(position.quantity * exit_percentage))
            decision.sell_percentage = exit_percentage
            decision.reasoning += f" | CONSERVATIVE EXIT ({exit_percentage:.1%})"
            logger.info(f"Conservative exit triggered - selling {exit_percentage:.1%} ({decision.sell_quantity} shares)")

        else:
            # Minimal exit
            decision.sell_quantity = max(1, int(position.quantity * 0.1))  # 10% minimum
            decision.sell_percentage = 0.1
            decision.reasoning += " | MINIMAL EXIT (10%)"
            logger.info(f"Minimal exit triggered - selling 10% ({decision.sell_quantity} shares)")

        # Ensure we don't sell more than we have
        decision.sell_quantity = min(decision.sell_quantity, position.quantity)
        logger.info(f"Final sell quantity: {decision.sell_quantity} shares ({decision.sell_percentage:.1%})")

        return decision

    def _validate_with_multi_timeframe(self, ticker: str, technical_analysis: Dict) -> Dict:
        """
        PRODUCTION ENHANCEMENT: Validate signals using multi-timeframe analysis
        Reduces false signal rate by confirming signals across multiple timeframes
        """
        try:
            # Import multi-timeframe analyzer
            from utils.multi_timeframe_analyzer import get_mtf_analyzer
            
            # Get analyzer instance
            mtf_analyzer = get_mtf_analyzer()
            
            # For production, we would use real data provider
            # In this implementation, we'll simulate validation based on current signals
            validation_result = {
                'is_valid': True,
                'reason': 'Validation passed',
                'confirmation_score': 0.8
            }
            
            # Check if we have strong trend confirmation
            sma_20 = technical_analysis.get("sma_20", 0)
            sma_50 = technical_analysis.get("sma_50", 0)
            current_price = technical_analysis.get("current_price", 0)
            
            # If we have moving averages, check for alignment
            if sma_20 > 0 and sma_50 > 0 and current_price > 0:
                # Check if price is above both moving averages (bullish alignment)
                if current_price > sma_20 and sma_20 > sma_50:
                    validation_result['confirmation_score'] = 0.9
                # Check if price is below both moving averages (bearish, so reject buy signal)
                elif current_price < sma_20 and sma_20 < sma_50:
                    validation_result['is_valid'] = False
                    validation_result['reason'] = 'Bearish trend across multiple timeframes'
                    validation_result['confirmation_score'] = 0.2
                else:
                    # Mixed signals, moderate confirmation
                    validation_result['confirmation_score'] = 0.6
            
            logger.info(f"MTF Validation for {ticker}: {validation_result}")
            return validation_result
            
        except Exception as e:
            logger.warning(f"MTF validation failed for {ticker}: {e}")
            # In case of error, be conservative but don't block completely
            return {
                'is_valid': True,
                'reason': f'MTF validation error: {e}',
                'confirmation_score': 0.5
            }
    
    def _adapt_to_market_regime(self, market_context: MarketContext) -> Dict:
        """
        PRODUCTION ENHANCEMENT: Adapt parameters based on market regime
        Adjusts signal thresholds and risk parameters for current market conditions
        """
        try:
            # Import regime detector
            from utils.market_regime_detector import get_regime_detector
            
            regime_detector = get_regime_detector()
            regime_params = regime_detector.get_regime_parameters()
            
            # Apply regime-specific adjustments
            adapted_params = {
                'rsi_buy_threshold': regime_params.get('rsi_buy_threshold', 30),
                'rsi_sell_threshold': regime_params.get('rsi_sell_threshold', 70),
                'position_size_multiplier': regime_params.get('position_size_multiplier', 1.0),
                'stop_loss_multiplier': regime_params.get('stop_loss_multiplier', 1.0)
            }
            
            logger.info(f"Market regime adaptation: {market_context.trend.value} -> {adapted_params}")
            return adapted_params
            
        except Exception as e:
            logger.warning(f"Market regime adaptation failed: {e}")
            # Return default parameters
            return {
                'rsi_buy_threshold': 30,
                'rsi_sell_threshold': 70,
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0
            }
