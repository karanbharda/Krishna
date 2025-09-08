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
    - Minimum 2-3 signal confirmation
    - Dynamic entry levels
    - Market independence
    - Risk-reward optimization
    - Market context awareness
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Professional thresholds - adjusted for better balance
        self.min_signals_required = config.get("min_buy_signals", 2)
        self.min_confidence_threshold = config.get("min_buy_confidence", 0.50)  # Reduced from 0.65
        self.min_weighted_score = config.get("min_weighted_buy_score", 0.30)   # Reduced from 0.40
        
        # Entry configuration
        self.base_entry_buffer_pct = config.get("entry_buffer_pct", 0.01)
        self.stop_loss_pct = config.get("buy_stop_loss_pct", 0.05)
        self.take_profit_ratio = config.get("take_profit_ratio", 2.0)
        
        # Position sizing
        self.partial_entry_threshold = config.get("partial_entry_threshold", 0.40)  # Reduced from 0.50
        self.full_entry_threshold = config.get("full_entry_threshold", 0.65)        # Reduced from 0.75
        
        # Market context filters
        self.downtrend_buy_multiplier = config.get("downtrend_buy_multiplier", 0.7)
        self.uptrend_buy_multiplier = config.get("uptrend_buy_multiplier", 1.2)
        
        logger.info("Professional Buy Logic initialized with adjusted parameters")
    
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
        
        # Step 1: Generate all buy signals
        signals = self._generate_buy_signals(
            stock_metrics, market_context, technical_analysis, 
            sentiment_analysis, ml_analysis
        )
        
        # Step 2: Calculate dynamic entry levels
        entry_levels = self._calculate_dynamic_entry_levels(stock_metrics, market_context)
        
        # Step 3: Check immediate entry conditions (breakouts, etc.)
        immediate_entry = self._check_immediate_entry_conditions(
            stock_metrics, entry_levels
        )
        
        if immediate_entry:
            return immediate_entry
        
        # Step 4: Evaluate signal-based buy decision
        signal_decision = self._evaluate_signal_based_buy(
            signals, stock_metrics, market_context
        )
        
        # Step 5: Apply market context filters
        final_decision = self._apply_market_context_filters(
            signal_decision, market_context
        )
        
        # Step 6: Determine position sizing
        final_decision = self._determine_position_sizing(
            final_decision, signals, stock_metrics, portfolio_context
        )
        
        # Step 7: Apply risk-reward optimization
        final_decision = self._optimize_risk_reward(
            final_decision, stock_metrics, entry_levels
        )
        
        logger.info(f"BUY DECISION: {final_decision.should_buy} | "
                   f"Qty: {final_decision.buy_quantity} | "
                   f"Reason: {final_decision.reason.value} | "
                   f"Confidence: {final_decision.confidence:.3f}")
        
        return final_decision

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

        # 1. Technical Analysis Signals (Weight: 30%)
        technical_signals = self._generate_technical_signals(technical_analysis, stock_metrics)
        signals.extend(technical_signals)

        # 2. Value & Risk Signals (Weight: 25%)
        value_signals = self._generate_value_signals(stock_metrics)
        signals.extend(value_signals)

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

    def _generate_technical_signals(self, technical: Dict, stock: StockMetrics) -> List[BuySignal]:
        """Generate technical analysis buy signals"""
        signals = []

        # RSI Oversold (Strong signal)
        rsi = technical.get("rsi", 50)
        if rsi < 30:
            strength = min((30 - rsi) / 20, 1.0)  # Scale 30-10 to 0-1
            signals.append(BuySignal(
                name="rsi_oversold",
                strength=strength,
                weight=0.08,  # 8% of total weight
                triggered=True,
                reasoning=f"RSI oversold at {rsi:.1f}",
                confidence=0.8
            ))

        # MACD Bullish Crossover
        macd = technical.get("macd", 0)
        macd_signal = technical.get("macd_signal", 0)
        if macd > macd_signal and macd > 0:
            strength = min(abs(macd - macd_signal) / abs(macd_signal) if macd_signal != 0 else 1, 1.0)
            signals.append(BuySignal(
                name="macd_bullish",
                strength=strength,
                weight=0.07,  # 7% of total weight
                triggered=True,
                reasoning="MACD bullish crossover",
                confidence=0.7
            ))

        # Moving Average Support
        sma_20 = technical.get("sma_20", stock.current_price)
        sma_50 = technical.get("sma_50", stock.current_price)
        if stock.current_price > sma_20 and sma_20 > sma_50:
            strength = (stock.current_price - sma_20) / sma_20
            signals.append(BuySignal(
                name="ma_support",
                strength=min(strength, 1.0),
                weight=0.06,  # 6% of total weight
                triggered=True,
                reasoning="Price above key moving averages",
                confidence=0.75
            ))

        # Support Level Bounce
        support = technical.get("support_level", stock.current_price * 1.1)
        if support > 0 and stock.current_price > support * 1.02:
            strength = (stock.current_price - support) / support
            signals.append(BuySignal(
                name="support_bounce",
                strength=min(strength * 2, 1.0),  # Amplify support bounces
                weight=0.09,  # 9% of total weight
                triggered=True,
                reasoning=f"Support bounce at {support:.2f}",
                confidence=0.85
            ))

        # Breakout Signal
        resistance = technical.get("resistance_level", stock.current_price * 0.9)
        if stock.current_price > resistance * 1.01:
            strength = (stock.current_price - resistance) / resistance
            signals.append(BuySignal(
                name="breakout",
                strength=min(strength * 1.5, 1.0),  # Amplify breakouts
                weight=0.10,  # 10% of total weight
                triggered=True,
                reasoning=f"Breakout above resistance {resistance:.2f}",
                confidence=0.9
            ))

        return signals

    def _generate_value_signals(self, stock: StockMetrics) -> List[BuySignal]:
        """Generate value-based buy signals"""
        signals = []

        # Low P/E Ratio Signal
        pe_ratio = stock.price_to_earnings
        if 0 < pe_ratio < 15:  # Reasonable P/E threshold
            strength = min((15 - pe_ratio) / 15, 1.0)
            signals.append(BuySignal(
                name="low_pe_ratio",
                strength=strength,
                weight=0.08,  # 8% of total weight
                triggered=True,
                reasoning=f"Low P/E ratio: {pe_ratio:.1f}",
                confidence=0.6
            ))

        # Low P/B Ratio Signal
        pb_ratio = stock.price_to_book
        if 0 < pb_ratio < 1.5:  # Below book value or reasonable threshold
            strength = min((1.5 - pb_ratio) / 1.5, 1.0)
            signals.append(BuySignal(
                name="low_pb_ratio",
                strength=strength,
                weight=0.07,  # 7% of total weight
                triggered=True,
                reasoning=f"Low P/B ratio: {pb_ratio:.2f}",
                confidence=0.65
            ))

        # Volatility Compression Signal (before breakout)
        if stock.volatility < 0.02:  # Low volatility (2%)
            strength = min((0.02 - stock.volatility) / 0.02, 1.0)
            signals.append(BuySignal(
                name="volatility_compression",
                strength=strength,
                weight=0.05,  # 5% of total weight
                triggered=True,
                reasoning=f"Low volatility: {stock.volatility:.1%}",
                confidence=0.5
            ))

        # ATR-based entry opportunity
        if stock.atr > 0 and stock.atr < stock.current_price * 0.03:  # Low ATR
            strength = min((stock.current_price * 0.03 - stock.atr) / (stock.current_price * 0.03), 1.0)
            signals.append(BuySignal(
                name="low_atr_opportunity",
                strength=strength,
                weight=0.05,  # 5% of total weight
                triggered=True,
                reasoning=f"Low ATR: {stock.atr:.2f}",
                confidence=0.55
            ))

        return signals

    def _generate_sentiment_signals(self, sentiment: Dict) -> List[BuySignal]:
        """Generate sentiment-based buy signals"""
        signals = []

        # Positive sentiment signal
        sentiment_score = sentiment.get("overall_sentiment", 0)
        if sentiment_score > 0.2:  # Positive sentiment threshold
            strength = min(sentiment_score / 0.8, 1.0)  # Scale to 0.8 max
            signals.append(BuySignal(
                name="positive_sentiment",
                strength=strength,
                weight=0.12,  # 12% of total weight
                triggered=True,
                reasoning=f"Positive sentiment: {sentiment_score:.2f}",
                confidence=0.6
            ))

        # News sentiment improvement
        news_trend = sentiment.get("news_trend", 0)
        if news_trend > 0.3:
            strength = min(news_trend / 0.7, 1.0)
            signals.append(BuySignal(
                name="news_improvement",
                strength=strength,
                weight=0.08,  # 8% of total weight
                triggered=True,
                reasoning="Improving news sentiment",
                confidence=0.5
            ))

        return signals

    def _generate_ml_signals(self, ml_analysis: Dict) -> List[BuySignal]:
        """Generate ML/AI-based buy signals"""
        signals = []

        # ML Prediction Signal
        ml_prediction = ml_analysis.get("prediction_direction", 0)
        if ml_prediction > 0.02:  # Positive prediction
            strength = min(ml_prediction / 0.10, 1.0)
            signals.append(BuySignal(
                name="ml_bullish_prediction",
                strength=strength,
                weight=0.10,  # 10% of total weight
                triggered=True,
                reasoning=f"ML bullish prediction: {ml_prediction:.1%}",
                confidence=ml_analysis.get("confidence", 0.5)
            ))

        # RL Recommendation
        rl_recommendation = ml_analysis.get("rl_recommendation", "HOLD")
        if rl_recommendation == "BUY":
            rl_confidence = ml_analysis.get("rl_confidence", 0.5)
            signals.append(BuySignal(
                name="rl_buy_recommendation",
                strength=rl_confidence,
                weight=0.05,  # 5% of total weight
                triggered=True,
                reasoning="RL algorithm recommends BUY",
                confidence=rl_confidence
            ))

        return signals

    def _generate_market_signals(self, market_context: MarketContext) -> List[BuySignal]:
        """Generate market structure buy signals"""
        signals = []

        # Low market stress signal (favorable for entries)
        if market_context.market_stress < 0.4:
            strength = (0.4 - market_context.market_stress) / 0.4
            signals.append(BuySignal(
                name="low_market_stress",
                strength=strength,
                weight=0.06,  # 6% of total weight
                triggered=True,
                reasoning=f"Low market stress: {market_context.market_stress:.1%}",
                confidence=0.7
            ))

        # Sector outperformance
        if market_context.sector_performance > 0.02:
            strength = min(market_context.sector_performance / 0.05, 1.0)
            signals.append(BuySignal(
                name="sector_strength",
                strength=strength,
                weight=0.04,  # 4% of total weight
                triggered=True,
                reasoning="Sector outperforming market",
                confidence=0.6
            ))

        return signals

    def _calculate_dynamic_entry_levels(self, stock: StockMetrics, market_context: MarketContext) -> Dict:
        """Calculate dynamic entry levels"""

        # Base entry with buffer
        base_entry = stock.current_price * (1 - self.base_entry_buffer_pct)

        # Volatility-adjusted entry
        volatility_multiplier = 1 + (stock.volatility / 0.02)  # Adjust for volatility
        volatility_entry = stock.current_price * (1 - self.base_entry_buffer_pct * volatility_multiplier)

        # Support-based entry
        support_entry = stock.support_level * 1.01 if stock.support_level > 0 else stock.current_price

        # Choose the most conservative (lowest) entry
        target_entry = min(base_entry, volatility_entry, support_entry)

        # Stop-loss based on ATR
        stop_loss = target_entry * (1 - self.stop_loss_pct)

        # Take-profit based on risk-reward ratio
        risk = target_entry - stop_loss
        take_profit = target_entry + (risk * self.take_profit_ratio)

        return {
            "target_entry": target_entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "base_entry": base_entry,
            "support_entry": support_entry
        }

    def _check_immediate_entry_conditions(self, stock: StockMetrics, entry_levels: Dict) -> Optional[BuyDecision]:
        """Check for immediate entry conditions (breakouts, etc.)"""

        # Breakout above resistance - immediate entry
        if stock.current_price > entry_levels["support_entry"] * 1.03:  # Strong breakout
            return BuyDecision(
                should_buy=True,
                buy_quantity=0,  # Will be determined later
                buy_percentage=1.0,
                reason=BuyReason.TECHNICAL_BREAKOUT,
                confidence=0.9,
                urgency=0.9,
                signals_triggered=[],
                target_entry_price=stock.current_price,
                stop_loss_price=stock.current_price * (1 - self.stop_loss_pct),
                take_profit_price=stock.current_price * (1 + self.take_profit_ratio * self.stop_loss_pct),
                reasoning=f"Strong breakout: {stock.current_price:.2f} > {entry_levels['support_entry'] * 1.03:.2f}"
            )

        return None

    def _evaluate_signal_based_buy(
        self,
        signals: List[BuySignal],
        stock: StockMetrics,
        market_context: MarketContext
    ) -> BuyDecision:
        """Evaluate buy decision based on signal analysis"""

        # Calculate weighted signal score
        triggered_signals = [s for s in signals if s.triggered]
        total_weight = sum(s.weight for s in triggered_signals)
        weighted_score = sum(s.strength * s.weight for s in triggered_signals)

        # Calculate confidence based on signal agreement
        signal_confidences = [s.confidence for s in triggered_signals]
        avg_confidence = np.mean(signal_confidences) if signal_confidences else 0.0

        # Professional decision criteria with adjusted thresholds
        signals_count = len(triggered_signals)
        meets_signal_threshold = signals_count >= self.min_signals_required
        meets_confidence_threshold = avg_confidence >= self.min_confidence_threshold
        meets_weighted_threshold = weighted_score >= self.min_weighted_score

        logger.info(f"Signal Analysis: {signals_count} signals, "
                   f"weighted_score: {weighted_score:.3f}, "
                   f"confidence: {avg_confidence:.3f}")

        # PROFESSIONAL BUY LOGIC: All three conditions must be met (similar to sell logic)
        # This prevents the system from generating buy signals when conditions are marginal
        should_buy = meets_signal_threshold and meets_confidence_threshold and meets_weighted_threshold

        # Additional quality checks for professional trading
        if should_buy:
            # Check if we have at least one strong signal (strength > 0.7)
            strong_signals = [s for s in triggered_signals if s.strength > 0.7]
            if len(strong_signals) == 0:
                # No strong signals, reduce confidence
                logger.info("No strong signals detected, reducing confidence")
                should_buy = False

        return BuyDecision(
            should_buy=should_buy,
            buy_quantity=0,  # Will be determined later
            buy_percentage=0.0,
            reason=BuyReason.TECHNICAL_BREAKOUT,
            confidence=avg_confidence,
            urgency=min(weighted_score * 1.2, 1.0),  # More conservative urgency scaling
            signals_triggered=triggered_signals,
            target_entry_price=0.0,
            stop_loss_price=0.0,
            take_profit_price=0.0,
            reasoning=f"Signal-based decision: {signals_count} signals, score: {weighted_score:.3f}, confidence: {avg_confidence:.3f}"
        )

    def _apply_market_context_filters(self, decision: BuyDecision, market_context: MarketContext) -> BuyDecision:
        """Apply market context filters to buy decision"""

        if not decision.should_buy:
            return decision

        # Be more conservative in downtrends
        if market_context.trend in [MarketTrend.DOWNTREND, MarketTrend.STRONG_DOWNTREND]:
            decision.confidence *= self.downtrend_buy_multiplier
            if decision.confidence < self.min_confidence_threshold:
                logger.info(f"BUY BLOCKED: Market in {market_context.trend.value}, confidence {decision.confidence:.3f} < threshold")
                decision.should_buy = False
                decision.reasoning += " | BLOCKED: Downtrend"
                return decision

        # Be more aggressive in uptrends
        elif market_context.trend in [MarketTrend.STRONG_UPTREND, MarketTrend.UPTREND]:
            decision.confidence *= self.uptrend_buy_multiplier
            decision.urgency = min(decision.urgency * 1.2, 1.0)

        return decision

    def _determine_position_sizing(
        self,
        decision: BuyDecision,
        signals: List[BuySignal],
        stock: StockMetrics,
        portfolio_context: Dict
    ) -> BuyDecision:
        """Determine how much of the position to buy"""

        if not decision.should_buy:
            return decision

        # Extract portfolio context
        available_cash = portfolio_context.get("available_cash", 0)
        total_value = portfolio_context.get("total_value", 1)
        max_exposure_per_stock = total_value * 0.25  # 25% max per stock
        current_stock_exposure = portfolio_context.get("current_stock_exposure", 0)

        # Full entry conditions
        if (decision.confidence >= self.full_entry_threshold or
            decision.urgency >= 0.9):

            decision.buy_quantity = int((max_exposure_per_stock - current_stock_exposure) / stock.current_price)
            decision.buy_percentage = 1.0
            decision.reasoning += " | FULL ENTRY"

        # Partial entry conditions
        elif decision.confidence >= self.partial_entry_threshold:
            # Scale entry size based on confidence and urgency
            entry_percentage = min(decision.confidence * decision.urgency, 0.75)
            decision.buy_quantity = int(((max_exposure_per_stock - current_stock_exposure) * entry_percentage) / stock.current_price)
            decision.buy_percentage = entry_percentage
            decision.reasoning += f" | PARTIAL ENTRY ({entry_percentage:.1%})"

        else:
            # Conservative entry
            decision.buy_quantity = int(((max_exposure_per_stock - current_stock_exposure) * 0.25) / stock.current_price)
            decision.buy_percentage = 0.25
            decision.reasoning += " | CONSERVATIVE ENTRY (25%)"

        # Ensure we don't exceed available cash
        max_affordable_qty = int(available_cash / stock.current_price)
        decision.buy_quantity = min(decision.buy_quantity, max_affordable_qty)

        # FIX: If calculated quantity is zero or negative, set should_buy to False
        # This prevents the system from defaulting to 1 share when no valid quantity can be calculated
        if decision.buy_quantity <= 0:
            decision.should_buy = False
            decision.buy_quantity = 0
            decision.buy_percentage = 0.0
            decision.reasoning += " | BLOCKED: No valid quantity calculated"
            
        return decision

    def _optimize_risk_reward(
        self,
        decision: BuyDecision,
        stock: StockMetrics,
        entry_levels: Dict
    ) -> BuyDecision:
        """Optimize risk-reward parameters"""

        if not decision.should_buy:
            return decision

        # Set optimized entry, stop-loss, and take-profit levels
        decision.target_entry_price = entry_levels["target_entry"]
        decision.stop_loss_price = entry_levels["stop_loss"]
        decision.take_profit_price = entry_levels["take_profit"]

        # Calculate actual risk-reward ratio
        risk = decision.target_entry_price - decision.stop_loss_price
        reward = decision.take_profit_price - decision.target_entry_price
        risk_reward_ratio = reward / risk if risk > 0 else 0

        # Adjust decision based on risk-reward
        if risk_reward_ratio < 1.5:  # Minimum acceptable risk-reward
            logger.info(f"RISK-REWARD SUBOPTIMAL: {risk_reward_ratio:.2f} < 1.5")
            decision.reasoning += f" | RISK-REWARD: {risk_reward_ratio:.2f}"
            # Block the trade if risk-reward is suboptimal
            decision.should_buy = False

        return decision