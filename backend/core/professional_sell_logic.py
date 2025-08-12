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
        
        # Professional thresholds
        self.min_signals_required = config.get("min_sell_signals", 2)
        self.min_confidence_threshold = config.get("min_sell_confidence", 0.65)
        self.min_weighted_score = config.get("min_weighted_sell_score", 0.40)
        
        # Stop-loss configuration
        self.base_stop_loss_pct = config.get("stop_loss_pct", 0.05)
        self.trailing_stop_pct = config.get("trailing_stop_pct", 0.03)
        self.profit_protection_threshold = config.get("profit_protection_threshold", 0.05)
        
        # Position sizing
        self.partial_exit_threshold = config.get("partial_exit_threshold", 0.50)
        self.full_exit_threshold = config.get("full_exit_threshold", 0.75)
        
        # Market context filters
        self.uptrend_sell_multiplier = config.get("uptrend_sell_multiplier", 1.5)
        self.downtrend_sell_multiplier = config.get("downtrend_sell_multiplier", 0.8)
        
        logger.info("Professional Sell Logic initialized with institutional parameters")
    
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
        Main entry point for professional sell evaluation
        """
        logger.info(f"=== PROFESSIONAL SELL EVALUATION: {ticker} ===")
        
        # Step 1: Generate all sell signals
        signals = self._generate_sell_signals(
            position_metrics, market_context, technical_analysis, 
            sentiment_analysis, ml_analysis
        )
        
        # Step 2: Calculate dynamic stop-loss levels
        stop_levels = self._calculate_dynamic_stops(position_metrics, market_context)
        
        # Step 3: Check immediate exit conditions (stop-loss, etc.)
        immediate_exit = self._check_immediate_exit_conditions(
            position_metrics, stop_levels
        )
        
        if immediate_exit:
            return immediate_exit
        
        # Step 4: Evaluate signal-based sell decision
        signal_decision = self._evaluate_signal_based_sell(
            signals, position_metrics, market_context
        )
        
        # Step 5: Apply market context filters
        final_decision = self._apply_market_context_filters(
            signal_decision, market_context
        )
        
        # Step 6: Determine position sizing
        final_decision = self._determine_position_sizing(
            final_decision, signals, position_metrics
        )
        
        logger.info(f"SELL DECISION: {final_decision.should_sell} | "
                   f"Qty: {final_decision.sell_quantity} | "
                   f"Reason: {final_decision.reason.value} | "
                   f"Confidence: {final_decision.confidence:.3f}")
        
        return final_decision

    def _generate_sell_signals(
        self,
        position_metrics: PositionMetrics,
        market_context: MarketContext,
        technical_analysis: Dict,
        sentiment_analysis: Dict,
        ml_analysis: Dict
    ) -> List[SellSignal]:
        """Generate comprehensive sell signals with professional weighting"""
        signals = []

        # 1. Technical Analysis Signals (Weight: 30%)
        technical_signals = self._generate_technical_signals(technical_analysis, position_metrics)
        signals.extend(technical_signals)

        # 2. Risk Management Signals (Weight: 25%)
        risk_signals = self._generate_risk_signals(position_metrics)
        signals.extend(risk_signals)

        # 3. Sentiment Signals (Weight: 20%)
        sentiment_signals = self._generate_sentiment_signals(sentiment_analysis)
        signals.extend(sentiment_signals)

        # 4. ML/AI Signals (Weight: 15%)
        ml_signals = self._generate_ml_signals(ml_analysis)
        signals.extend(ml_signals)

        # 5. Market Structure Signals (Weight: 10%)
        market_signals = self._generate_market_signals(market_context)
        signals.extend(market_signals)

        return signals

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
                confidence=0.8
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
                confidence=0.7
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
                confidence=0.75
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
                confidence=0.85
            ))

        return signals

    def _generate_risk_signals(self, position: PositionMetrics) -> List[SellSignal]:
        """Generate risk management sell signals"""
        signals = []

        # Unrealized Loss Signal
        if position.unrealized_pnl_pct < -0.02:  # 2% loss threshold
            strength = min(abs(position.unrealized_pnl_pct) / 0.10, 1.0)  # Scale to 10% max
            signals.append(SellSignal(
                name="unrealized_loss",
                strength=strength,
                weight=0.10,  # 10% of total weight
                triggered=True,
                reasoning=f"Unrealized loss: {position.unrealized_pnl_pct:.1%}",
                confidence=0.9
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
                confidence=0.7
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
                confidence=0.6
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
                confidence=0.6
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
                confidence=0.5
            ))

        return signals

    def _generate_ml_signals(self, ml_analysis: Dict) -> List[SellSignal]:
        """Generate ML/AI-based sell signals"""
        signals = []

        # ML Prediction Signal
        ml_prediction = ml_analysis.get("prediction_direction", 0)
        if ml_prediction < -0.02:  # Negative prediction
            strength = min(abs(ml_prediction) / 0.10, 1.0)
            signals.append(SellSignal(
                name="ml_bearish_prediction",
                strength=strength,
                weight=0.10,  # 10% of total weight
                triggered=True,
                reasoning=f"ML bearish prediction: {ml_prediction:.1%}",
                confidence=ml_analysis.get("confidence", 0.5)
            ))

        # RL Recommendation
        rl_recommendation = ml_analysis.get("rl_recommendation", "HOLD")
        if rl_recommendation == "SELL":
            rl_confidence = ml_analysis.get("rl_confidence", 0.5)
            signals.append(SellSignal(
                name="rl_sell_recommendation",
                strength=rl_confidence,
                weight=0.05,  # 5% of total weight
                triggered=True,
                reasoning="RL algorithm recommends SELL",
                confidence=rl_confidence
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
                confidence=0.7
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
                confidence=0.6
            ))

        return signals

    def _calculate_dynamic_stops(self, position: PositionMetrics, market_context: MarketContext) -> Dict:
        """Calculate dynamic stop-loss levels"""

        # Base stop-loss
        base_stop = position.entry_price * (1 - self.base_stop_loss_pct)

        # Volatility-adjusted stop
        volatility_multiplier = 1 + (position.volatility / 0.02)  # Adjust for volatility
        volatility_stop = position.entry_price * (1 - self.base_stop_loss_pct * volatility_multiplier)

        # Trailing stop based on highest price
        trailing_stop = position.highest_price_since_entry * (1 - self.trailing_stop_pct)

        # Profit protection stop (lock in profits after 5%+ gains)
        profit_protection_stop = None
        if position.unrealized_pnl_pct > self.profit_protection_threshold:
            # Lock in 50% of profits
            profit_to_lock = position.unrealized_pnl_pct * 0.5
            profit_protection_stop = position.entry_price * (1 + profit_to_lock)

        # Choose the most restrictive (highest) stop
        active_stop = max(base_stop, volatility_stop, trailing_stop)
        if profit_protection_stop:
            active_stop = max(active_stop, profit_protection_stop)

        return {
            "base_stop": base_stop,
            "volatility_stop": volatility_stop,
            "trailing_stop": trailing_stop,
            "profit_protection_stop": profit_protection_stop,
            "active_stop": active_stop
        }

    def _check_immediate_exit_conditions(self, position: PositionMetrics, stop_levels: Dict) -> Optional[SellDecision]:
        """Check for immediate exit conditions (stop-loss, etc.)"""

        # Hard stop-loss hit
        if position.current_price <= stop_levels["active_stop"]:
            reason = SellReason.STOP_LOSS
            if stop_levels["profit_protection_stop"] and position.current_price <= stop_levels["profit_protection_stop"]:
                reason = SellReason.PROFIT_PROTECTION
            elif position.current_price <= stop_levels["trailing_stop"]:
                reason = SellReason.TRAILING_STOP

            return SellDecision(
                should_sell=True,
                sell_quantity=position.quantity,
                sell_percentage=1.0,
                reason=reason,
                confidence=1.0,
                urgency=1.0,
                signals_triggered=[],
                stop_loss_price=stop_levels["active_stop"],
                take_profit_price=0.0,
                trailing_stop_price=stop_levels["trailing_stop"],
                reasoning=f"Stop-loss triggered: {position.current_price:.2f} <= {stop_levels['active_stop']:.2f}"
            )

        return None

    def _evaluate_signal_based_sell(
        self,
        signals: List[SellSignal],
        position: PositionMetrics,
        market_context: MarketContext
    ) -> SellDecision:
        """Evaluate sell decision based on signal analysis"""

        # Calculate weighted signal score
        triggered_signals = [s for s in signals if s.triggered]
        total_weight = sum(s.weight for s in triggered_signals)
        weighted_score = sum(s.strength * s.weight for s in triggered_signals)

        # Calculate confidence based on signal agreement
        signal_confidences = [s.confidence for s in triggered_signals]
        avg_confidence = np.mean(signal_confidences) if signal_confidences else 0.0

        # Professional decision criteria
        signals_count = len(triggered_signals)
        meets_signal_threshold = signals_count >= self.min_signals_required
        meets_confidence_threshold = avg_confidence >= self.min_confidence_threshold
        meets_weighted_threshold = weighted_score >= self.min_weighted_score

        logger.info(f"Signal Analysis: {signals_count} signals, "
                   f"weighted_score: {weighted_score:.3f}, "
                   f"confidence: {avg_confidence:.3f}")

        should_sell = meets_signal_threshold and meets_confidence_threshold and meets_weighted_threshold

        return SellDecision(
            should_sell=should_sell,
            sell_quantity=0,  # Will be determined later
            sell_percentage=0.0,
            reason=SellReason.TECHNICAL_SIGNALS,
            confidence=avg_confidence,
            urgency=min(weighted_score, 1.0),
            signals_triggered=triggered_signals,
            stop_loss_price=0.0,
            take_profit_price=0.0,
            trailing_stop_price=0.0,
            reasoning=f"Signal-based decision: {signals_count} signals, score: {weighted_score:.3f}"
        )

    def _apply_market_context_filters(self, decision: SellDecision, market_context: MarketContext) -> SellDecision:
        """Apply market context filters to sell decision"""

        if not decision.should_sell:
            return decision

        # Don't sell in strong uptrends unless urgency is very high
        if market_context.trend in [MarketTrend.STRONG_UPTREND, MarketTrend.UPTREND]:
            if decision.urgency < 0.8:  # High urgency threshold in uptrends
                logger.info(f"SELL BLOCKED: Market in {market_context.trend.value}, urgency {decision.urgency:.3f} < 0.8")
                decision.should_sell = False
                decision.reasoning += " | BLOCKED: Strong uptrend"
                return decision
            else:
                # Reduce position size in uptrends
                decision.confidence *= self.downtrend_sell_multiplier

        # Be more aggressive in downtrends
        elif market_context.trend in [MarketTrend.DOWNTREND, MarketTrend.STRONG_DOWNTREND]:
            decision.confidence *= self.uptrend_sell_multiplier
            decision.urgency = min(decision.urgency * 1.2, 1.0)

        return decision

    def _determine_position_sizing(
        self,
        decision: SellDecision,
        signals: List[SellSignal],
        position: PositionMetrics
    ) -> SellDecision:
        """Determine how much of the position to sell"""

        if not decision.should_sell:
            return decision

        # Full exit conditions
        if (decision.confidence >= self.full_exit_threshold or
            decision.urgency >= 0.9 or
            position.unrealized_pnl_pct < -0.08):  # 8% loss

            decision.sell_quantity = position.quantity
            decision.sell_percentage = 1.0
            decision.reasoning += " | FULL EXIT"

        # Partial exit conditions
        elif decision.confidence >= self.partial_exit_threshold:
            # Scale exit size based on confidence and urgency
            exit_percentage = min(decision.confidence * decision.urgency, 0.75)
            decision.sell_quantity = int(position.quantity * exit_percentage)
            decision.sell_percentage = exit_percentage
            decision.reasoning += f" | PARTIAL EXIT ({exit_percentage:.1%})"

        else:
            # Conservative exit
            decision.sell_quantity = max(int(position.quantity * 0.25), 1)
            decision.sell_percentage = 0.25
            decision.reasoning += " | CONSERVATIVE EXIT (25%)"

        return decision
