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
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Professional thresholds - adjusted for better balance
        self.min_signals_required = config.get("min_buy_signals", 2)  # Minimum 2 signals
        self.max_signals_required = config.get("max_buy_signals", 4)  # Maximum 4 signals
        self.min_confidence_threshold = config.get("min_buy_confidence", 0.50)
        self.min_weighted_score = config.get("min_weighted_buy_score", 0.30)
        
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
        
        # Balanced weights: Technical 25%, Value 20%, Sentiment 20%, ML 20%, Market Structure 15%
        self.category_weights = {
            "Technical": 0.25,
            "Value": 0.20,
            "Sentiment": 0.20,
            "ML": 0.20,
            "Market": 0.15
        }
        
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
        
        # Step 2: Apply cross-category confirmation (at least 2 categories must align)
        if not self._check_cross_category_confirmation(signals):
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
                reasoning="Cross-category confirmation failed: Less than 2 categories aligned"
            )
        
        # Step 3: Apply bearish block filter (If bearish > 40%, block trade)
        if self._check_bearish_block(signals):
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
                reasoning="Bearish block filter activated: Bearish signals > 40%"
            )
        
        # Step 4: Calculate dynamic entry levels
        entry_levels = self._calculate_dynamic_entry_levels(stock_metrics, market_context)
        
        # Step 5: Check immediate entry conditions (breakouts, etc.)
        immediate_entry = self._check_immediate_entry_conditions(
            stock_metrics, entry_levels
        )
        
        if immediate_entry:
            return immediate_entry
        
        # Step 6: Evaluate signal-based buy decision
        signal_decision = self._evaluate_signal_based_buy(
            signals, stock_metrics, market_context
        )
        
        # Step 7: Apply market context filters
        final_decision = self._apply_market_context_filters(
            signal_decision, market_context
        )
        
        # Step 8: Apply risk-reward check (Requires ≥ 1.5 before entering)
        final_decision = self._apply_risk_reward_check(
            final_decision, stock_metrics, entry_levels
        )
        
        # Step 9: Determine position sizing with smoothed scaling
        final_decision = self._determine_smoothed_position_sizing(
            final_decision, signals, stock_metrics, portfolio_context
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

    def _check_bearish_block(self, signals: List[BuySignal]) -> bool:
        """Bearish block filter: If bearish > 40%, block trade"""
        triggered_signals = [s for s in signals if s.triggered]
        if not triggered_signals:
            return False
            
        bearish_signals = [s for s in triggered_signals if "bearish" in s.name.lower() or "overbought" in s.name.lower()]
        bearish_percentage = len(bearish_signals) / len(triggered_signals)
        return bearish_percentage > 0.4

    def _generate_technical_signals(self, technical: Dict, stock: StockMetrics, category_weight: float = 0.25) -> List[BuySignal]:
        """Generate technical analysis buy signals"""
        signals = []

        # RSI Oversold (Strong signal)
        rsi = technical.get("rsi", 50)
        if rsi < 30:
            strength = min((30 - rsi) / 20, 1.0)  # Scale 30-10 to 0-1
            signals.append(BuySignal(
                name="rsi_oversold",
                strength=strength,
                weight=category_weight * 0.08,  # 8% of Technical category weight
                triggered=True,
                reasoning=f"RSI oversold at {rsi:.1f}",
                confidence=0.8,
                category="Technical"
            ))

        # MACD Bullish Crossover
        macd = technical.get("macd", 0)
        macd_signal = technical.get("macd_signal", 0)
        if macd > macd_signal and macd > 0:
            strength = min(abs(macd - macd_signal) / abs(macd_signal) if macd_signal != 0 else 1, 1.0)
            signals.append(BuySignal(
                name="macd_bullish",
                strength=strength,
                weight=category_weight * 0.07,  # 7% of Technical category weight
                triggered=True,
                reasoning="MACD bullish crossover",
                confidence=0.7,
                category="Technical"
            ))

        # Moving Average Support
        sma_20 = technical.get("sma_20", stock.current_price)
        sma_50 = technical.get("sma_50", stock.current_price)
        if stock.current_price > sma_20 and sma_20 > sma_50:
            strength = (stock.current_price - sma_20) / sma_20
            signals.append(BuySignal(
                name="ma_support",
                strength=min(strength, 1.0),
                weight=category_weight * 0.06,  # 6% of Technical category weight
                triggered=True,
                reasoning="Price above key moving averages",
                confidence=0.75,
                category="Technical"
            ))

        # Support Level Bounce
        support = technical.get("support_level", stock.current_price * 1.1)
        if support > 0 and stock.current_price > support * 1.02:
            strength = (stock.current_price - support) / support
            signals.append(BuySignal(
                name="support_bounce",
                strength=min(strength * 2, 1.0),  # Amplify support bounces
                weight=category_weight * 0.09,  # 9% of Technical category weight
                triggered=True,
                reasoning=f"Support bounce at {support:.2f}",
                confidence=0.85,
                category="Technical"
            ))

        # Breakout Signal
        resistance = technical.get("resistance_level", stock.current_price * 0.9)
        if stock.current_price > resistance * 1.01:
            strength = (stock.current_price - resistance) / resistance
            signals.append(BuySignal(
                name="breakout",
                strength=min(strength * 1.5, 1.0),  # Amplify breakouts
                weight=category_weight * 0.10,  # 10% of Technical category weight
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
                weight=category_weight * 0.08,  # 8% of Value category weight
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
                weight=category_weight * 0.07,  # 7% of Value category weight
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
                weight=category_weight * 0.05,  # 5% of Value category weight
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
                weight=category_weight * 0.05,  # 5% of Value category weight
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
                weight=category_weight * 0.12,  # 12% of Sentiment category weight
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
                weight=category_weight * 0.08,  # 8% of Sentiment category weight
                triggered=True,
                reasoning="Improving news sentiment",
                confidence=0.5,
                category="Sentiment"
            ))

        return signals

    def _generate_ml_signals(self, ml_analysis: Dict, category_weight: float = 0.20) -> List[BuySignal]:
        """Generate ML/AI-based buy signals"""
        signals = []

        # ML Prediction Signal
        ml_prediction = ml_analysis.get("prediction_direction", 0)
        if ml_prediction > 0.02:  # Positive prediction
            strength = min(ml_prediction / 0.10, 1.0)
            signals.append(BuySignal(
                name="ml_bullish_prediction",
                strength=strength,
                weight=category_weight * 0.10,  # 10% of ML category weight
                triggered=True,
                reasoning=f"ML bullish prediction: {ml_prediction:.1%}",
                confidence=ml_analysis.get("confidence", 0.5),
                category="ML"
            ))

        # RL Recommendation
        rl_recommendation = ml_analysis.get("rl_recommendation", "HOLD")
        if rl_recommendation == "BUY":
            rl_confidence = ml_analysis.get("rl_confidence", 0.5)
            signals.append(BuySignal(
                name="rl_buy_recommendation",
                strength=rl_confidence,
                weight=category_weight * 0.05,  # 5% of ML category weight
                triggered=True,
                reasoning="RL algorithm recommends BUY",
                confidence=rl_confidence,
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
                weight=category_weight * 0.06,  # 6% of Market category weight
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
                weight=category_weight * 0.04,  # 4% of Market category weight
                triggered=True,
                reasoning="Sector outperforming market",
                confidence=0.6,
                category="Market"
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
        # Make this condition more stringent to prevent too many buy signals
        if (stock.current_price > entry_levels["support_entry"] * 1.05 and  # Increased threshold from 1.03 to 1.05
            entry_levels["support_entry"] > 0):  # Ensure support level is valid
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
                reasoning=f"Strong breakout: {stock.current_price:.2f} > {entry_levels['support_entry'] * 1.05:.2f}"
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
        # Check if signal count is within the required range (min 2, max 4)
        meets_signal_threshold = self.min_signals_required <= signals_count <= self.max_signals_required
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

        # Additional check: Ensure we have meaningful signals
        if should_buy and signals_count > 0:
            # Calculate average signal strength
            avg_strength = np.mean([s.strength for s in triggered_signals])
            # If average strength is too low, don't buy
            if avg_strength < 0.4:  # Increased threshold from 0.3 to 0.4
                should_buy = False
                logger.info(f"Average signal strength too low: {avg_strength:.3f}")

        # Ensure there's a clear bullish bias
        if should_buy:
            # Count bullish vs bearish signals
            bullish_signals = [s for s in triggered_signals if s.weight > 0 and ("oversold" in s.name or "bullish" in s.name or "bounce" in s.name or "breakout" in s.name or "low" in s.name or "compression" in s.name)]
            bearish_signals = [s for s in signals if s.triggered and s.weight > 0 and ("overbought" in s.name or "bearish" in s.name or "breakdown" in s.name)]
            
            # If more bearish than bullish signals, don't buy
            if len(bearish_signals) >= len(bullish_signals):
                should_buy = False
                logger.info(f"Bearish signals ({len(bearish_signals)}) >= Bullish signals ({len(bullish_signals)}), blocking buy")

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

    def _apply_risk_reward_check(
        self,
        decision: BuyDecision,
        stock: StockMetrics,
        entry_levels: Dict
    ) -> BuyDecision:
        """R:R check: Requires ≥ 1.5 before entering"""

        if not decision.should_buy:
            return decision

        # Calculate actual risk-reward ratio
        risk = entry_levels["target_entry"] - entry_levels["stop_loss"]
        reward = entry_levels["take_profit"] - entry_levels["target_entry"]
        risk_reward_ratio = reward / risk if risk > 0 else 0

        # Adjust decision based on risk-reward
        if risk_reward_ratio < 1.5:  # Minimum acceptable risk-reward
            logger.info(f"RISK-REWARD SUBOPTIMAL: {risk_reward_ratio:.2f} < 1.5")
            decision.reasoning += f" | RISK-REWARD: {risk_reward_ratio:.2f}"
            # Block the trade if risk-reward is suboptimal
            decision.should_buy = False

        return decision

    def _determine_smoothed_position_sizing(
        self,
        decision: BuyDecision,
        signals: List[BuySignal],
        stock: StockMetrics,
        portfolio_context: Dict
    ) -> BuyDecision:
        """Smoothed position sizing: Uses confidence scaling instead of hard cutoffs"""

        if not decision.should_buy:
            return decision

        # Extract portfolio context
        available_cash = portfolio_context.get("available_cash", 0)
        total_value = portfolio_context.get("total_value", 1)
        max_exposure_per_stock = total_value * 0.25  # 25% max per stock
        current_stock_exposure = portfolio_context.get("current_stock_exposure", 0)

        # Smoothed position sizing based on confidence and urgency
        # Scale from 0.1 (10%) to 1.0 (100%) based on confidence and urgency
        position_scale = min(decision.confidence * decision.urgency * 1.2, 1.0)
        position_scale = max(position_scale, 0.1)  # Minimum 10% position

        # Calculate position value
        target_position_value = (max_exposure_per_stock - current_stock_exposure) * position_scale
        decision.buy_quantity = int(target_position_value / stock.current_price)
        decision.buy_percentage = position_scale
        decision.reasoning += f" | SMOOTHED POSITION ({position_scale:.1%})"

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
