"""
Professional Buy Logic Framework
Implements institutional-grade buy logic with proper risk management,
entry confirmation, and market context awareness.
"""

import logging
from typing import Dict, List, Tuple, Optional, NamedTuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

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
    # Additional fields for value analysis
    earnings_growth: float = 0.05  # Default 5% growth
    return_on_equity: float = 0.10  # Default 10% ROE
    free_cash_flow_yield: float = 0.05  # Default 5% FCF yield
    debt_to_equity: float = 0.5  # Default 0.5 debt-to-equity
    dividend_yield: float = 0.0  # Default 0% dividend yield
    payout_ratio: float = 0.0  # Default 0% payout ratio
    earnings_quality: float = 0.5  # Default 50% earnings quality
    insider_ownership: float = 0.0  # Default 0% insider ownership
    sector_pe: float = 20.0  # Default sector P/E of 20

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
    - Minimum 3-5 signal confirmation
    - Dynamic entry levels
    - Market independence
    - Risk-reward optimization
    - Market context awareness
    - REALISTIC BUY LOGIC: No forced signals, only genuine opportunities
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Professional thresholds - REALISTIC for better opportunity capture with combined signals
        self.min_signals_required = config.get("min_buy_signals", 3)  # Minimum 3 categories
        self.max_signals_required = config.get("max_buy_signals", 5)  # Maximum 5 categories
        self.min_confidence_threshold = config.get("min_buy_confidence", 0.60)  # Increased to 60% minimum confidence
        self.min_weighted_score = config.get("min_weighted_buy_score", 0.15)  # Increased to 15% minimum weighted score
        
        # REALISTIC BUY LOGIC: Enhanced parameters for better entry timing
        self.signal_sensitivity_multiplier = config.get("signal_sensitivity_multiplier", 0.9)  # Reduced sensitivity
        self.early_entry_buffer_pct = config.get("early_entry_buffer_pct", 0.01)  # 1% early entry buffer
        self.aggressive_entry_threshold = config.get("aggressive_entry_threshold", 0.90)  # 90% for aggressive entry
        
        # REALISTIC BUY LOGIC: Dynamic signal thresholds
        self.dynamic_signal_thresholds = config.get("dynamic_signal_thresholds", True)
        self.signal_strength_boost = config.get("signal_strength_boost", 0.03)  # 3% boost
        
        # REALISTIC BUY LOGIC: Enhanced ML integration
        self.ml_signal_weight_boost = config.get("ml_signal_weight_boost", 0.05)  # 5% ML boost
        self.ml_confidence_multiplier = config.get("ml_confidence_multiplier", 1.10)  # 1.10x ML multiplier
        
        # REALISTIC BUY LOGIC: Improved momentum detection
        self.momentum_confirmation_window = config.get("momentum_confirmation_window", 5)
        self.momentum_strength_threshold = config.get("momentum_strength_threshold", 0.035)  # 3.5% momentum threshold
        
        # ADAPTIVE THRESHOLDS: Adjust based on market conditions
        self.adaptive_thresholds_enabled = config.get("adaptive_thresholds_enabled", True)
        self.market_volatility_threshold = config.get("market_volatility_threshold", 0.025)
        self.low_confidence_multiplier = config.get("low_confidence_multiplier", 0.6)
        self.high_confidence_multiplier = config.get("high_confidence_multiplier", 1.05)
        
        # Market context filters - More restrictive
        self.uptrend_buy_multiplier = config.get("uptrend_buy_multiplier", 1.05)  # Less boost in uptrends
        self.downtrend_buy_multiplier = config.get("downtrend_buy_multiplier", 0.70)  # More restriction in downtrends
        
        # DYNAMIC STOP-LOSS: Read from live_config.json instead of hardcoded
        self._load_dynamic_config()
        
        # Define category weights for signal generation
        self.category_weights = {
            "Technical": 0.25,   # 25% weight to technical signals
            "Value": 0.20,       # 20% weight to value signals
            "Sentiment": 0.20,   # 20% weight to sentiment signals
            "ML": 0.20,          # 20% weight to ML signals
            "Market": 0.15        # 15% weight to market structure signals
        }
        
        logger.info("Professional Buy Logic initialized with realistic parameters")
    
    def _load_dynamic_config(self):
        """Load dynamic configuration from live_config.json"""
        try:
            import json
            import os
            
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'live_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    live_config = json.load(f)
                
                # Get dynamic stop-loss percentage from frontend config (already as decimal)
                self.stop_loss_pct = live_config.get("stop_loss_pct", 0.03)
                
                # Get target price configuration from frontend config
                target_level = live_config.get("target_price_level", "MEDIUM")
                target_multiplier = live_config.get("target_price_multiplier", 0.02)  # Already as decimal from frontend
                
                # Map target level to percentage (as decimal)
                target_percentages = {
                    "LOW": 0.04,      # 4% target price
                    "MEDIUM": 0.02,   # 2% target price  
                    "HIGH": 0.06,     # 6% target price
                    "CUSTOM": target_multiplier  # Use custom value (already as decimal)
                }
                
                # Store as decimal for calculation
                self.target_price_pct = target_percentages.get(target_level, 0.02)
                
                logger.info(f"📊 Loaded dynamic config - Stop Loss: {self.stop_loss_pct:.1%}, Target Price: {self.target_price_pct:.1%} (Level: {target_level})")
            else:
                logger.warning("live_config.json not found, using defaults")
                self.stop_loss_pct = 0.03
                self.target_price_pct = 0.02
                
        except Exception as e:
            logger.error(f"Failed to load dynamic config: {e}")
            self.stop_loss_pct = 0.03
            self.target_price_pct = 0.02
    
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
        logger.info(f"Generated {len(signals)} combined signals (one per category)")
        triggered_signals = [s for s in signals if s.triggered]
        logger.info(f"{len(triggered_signals)} triggered category signals:")
        
        # Group signals by category for better visualization
        category_signals = {}
        for signal in signals:  # Show all signals, not just triggered ones
            category = signal.category or "Uncategorized"
            if category not in category_signals:
                category_signals[category] = []
            category_signals[category].append(signal)
        
        for category, signals_list in category_signals.items():
            for signal in signals_list:
                status = "✅ TRIGGERED" if signal.triggered else "❌ NOT TRIGGERED"
                logger.info(f"  {category} Signal:")
                logger.info(f"    - {signal.name}: {status}")
                logger.info(f"      Strength: {signal.strength:.3f}, Weight: {signal.weight:.3f}, Confidence: {signal.confidence:.3f}")
                logger.info(f"      Reasoning: {signal.reasoning}")
        
        # Log non-triggered signals for completeness
        non_triggered_signals = [s for s in signals if not s.triggered]
        if non_triggered_signals:
            logger.info(f"{len(non_triggered_signals)} non-triggered signals:")
            for signal in non_triggered_signals:
                logger.info(f"  - {signal.name}: Not triggered - {signal.reasoning}")
        
        # Step 2: Apply cross-category confirmation (at least 3 categories must align)
        cross_category_confirmed = self._check_cross_category_confirmation(signals)
        categories_triggered = set(s.category for s in triggered_signals if s.category)
        logger.info(f"Cross-category confirmation: {cross_category_confirmed} (Categories: {', '.join(categories_triggered) if categories_triggered else 'None'})")
        if not cross_category_confirmed:
            logger.info("❌ BUY REJECTED: Cross-category confirmation failed - signals not aligned across multiple categories")
            detailed_reasoning = f"Cross-category confirmation failed - only {len(categories_triggered)} categories triggered: {', '.join(categories_triggered) if categories_triggered else 'None'}. Need at least 3 different categories."
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
            detailed_reasoning = f"Bearish block filter triggered - {len(bearish_signals)} bearish signals ({bearish_percentage:.1%} of triggered signals). Threshold is 20%."
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
        # For combined signals, we expect 3-5 triggered signals (one per category)
        # Adjust thresholds to be more appropriate for combined signals
        meets_signal_threshold = 3 <= signals_count <= 5  # At least 3 categories should trigger
        meets_confidence_threshold = avg_confidence >= self.min_confidence_threshold
        meets_weighted_threshold = weighted_score >= self.min_weighted_score

        logger.info(f"Signal Analysis Summary:")
        logger.info(f"  Signal Count: {signals_count} (Required: 3-5) - {'✅ PASS' if meets_signal_threshold else '❌ FAIL'}")
        logger.info(f"  Average Confidence: {avg_confidence:.3f} (Threshold: {self.min_confidence_threshold}) - {'✅ PASS' if meets_confidence_threshold else '❌ FAIL'}")
        logger.info(f"  Weighted Score: {weighted_score:.3f} (Threshold: {self.min_weighted_score}) - {'✅ PASS' if meets_weighted_threshold else '❌ FAIL'}")

        # REALISTIC BUY LOGIC: All three conditions must be met (similar to sell logic)
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
                reasoning=f"Professional buy confirmed: {signals_count} categories triggered, "
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
                rejection_reasons.append(f"Triggered categories {signals_count} not in range [3, 5]")
            if not meets_confidence_threshold:
                rejection_reasons.append(f"Confidence {avg_confidence:.3f} below threshold {self.min_confidence_threshold}")
            if not meets_weighted_threshold:
                rejection_reasons.append(f"Weighted score {weighted_score:.3f} below threshold {self.min_weighted_score}")
            
            detailed_reasoning = " | ".join(rejection_reasons)
            logger.info(f"Rejection Reasons: {detailed_reasoning}")
            
            # Log additional diagnostic information
            logger.info(f"🔍 DETAILED DIAGNOSTIC INFORMATION:")
            logger.info(f"   - Minimum Signals Required: 3 categories")
            logger.info(f"   - Maximum Signals Required: 5 categories")
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
        """Bearish block filter: If bearish > 20%, block trade"""
        triggered_signals = [s for s in signals if s.triggered]
        if not triggered_signals:
            return False
            
        bearish_signals = [s for s in triggered_signals if "bearish" in s.name.lower() or "overbought" in s.name.lower()]
        bearish_percentage = len(bearish_signals) / len(triggered_signals)
        return bearish_percentage > 0.20  # REDUCED from 0.25 to 0.20

    def _generate_buy_signals(
        self,
        stock_metrics: StockMetrics,
        market_context: MarketContext,
        technical_analysis: Dict,
        sentiment_analysis: Dict,
        ml_analysis: Dict
    ) -> List[BuySignal]:
        """Generate comprehensive buy signals with professional weighting - ONE signal per category"""
        signals = []

        # Get dynamic category weights based on market regime
        dynamic_weights = self._get_dynamic_category_weights(market_context)

        # 1. Technical Analysis Signals - Combine all technical signals into one
        technical_signals = self._generate_technical_signals(technical_analysis, stock_metrics, dynamic_weights["Technical"])
        if technical_signals:
            # Combine all technical signals into one signal
            triggered_tech_signals = [s for s in technical_signals if s.triggered]
            if triggered_tech_signals:
                # Calculate combined strength as average of triggered signals
                combined_strength = sum(s.strength for s in triggered_tech_signals) / len(triggered_tech_signals)
                combined_weight = dynamic_weights["Technical"]  # Use category weight
                combined_confidence = sum(s.confidence for s in triggered_tech_signals) / len(triggered_tech_signals)
                
                # Create single combined technical signal
                combined_tech_signal = BuySignal(
                    name="technical_combined",
                    strength=combined_strength,
                    weight=combined_weight,
                    triggered=True,
                    reasoning=f"Combined technical signal from {len(triggered_tech_signals)} indicators",
                    confidence=combined_confidence,
                    category="Technical"
                )
                signals.append(combined_tech_signal)

        # 2. Value & Risk Signals - Combine all value signals into one
        value_signals = self._generate_value_signals(stock_metrics, dynamic_weights["Value"])
        if value_signals:
            # Combine all value signals into one signal
            triggered_value_signals = [s for s in value_signals if s.triggered]
            if triggered_value_signals:
                # Calculate combined strength as average of triggered signals
                combined_strength = sum(s.strength for s in triggered_value_signals) / len(triggered_value_signals)
                combined_weight = dynamic_weights["Value"]  # Use category weight
                combined_confidence = sum(s.confidence for s in triggered_value_signals) / len(triggered_value_signals)
                
                # Create single combined value signal
                combined_value_signal = BuySignal(
                    name="value_combined",
                    strength=combined_strength,
                    weight=combined_weight,
                    triggered=True,
                    reasoning=f"Combined value signal from {len(triggered_value_signals)} indicators",
                    confidence=combined_confidence,
                    category="Value"
                )
                signals.append(combined_value_signal)

        # 3. Sentiment Signals - Combine all sentiment signals into one
        sentiment_signals = self._generate_sentiment_signals(sentiment_analysis, dynamic_weights["Sentiment"])
        if sentiment_signals:
            # Combine all sentiment signals into one signal
            triggered_sentiment_signals = [s for s in sentiment_signals if s.triggered]
            if triggered_sentiment_signals:
                # Calculate combined strength as average of triggered signals
                combined_strength = sum(s.strength for s in triggered_sentiment_signals) / len(triggered_sentiment_signals)
                combined_weight = dynamic_weights["Sentiment"]  # Use category weight
                combined_confidence = sum(s.confidence for s in triggered_sentiment_signals) / len(triggered_sentiment_signals)
                
                # Create single combined sentiment signal
                combined_sentiment_signal = BuySignal(
                    name="sentiment_combined",
                    strength=combined_strength,
                    weight=combined_weight,
                    triggered=True,
                    reasoning=f"Combined sentiment signal from {len(triggered_sentiment_signals)} indicators",
                    confidence=combined_confidence,
                    category="Sentiment"
                )
                signals.append(combined_sentiment_signal)

        # 4. ML/AI Signals - Combine all ML signals into one
        ml_signals = self._generate_ml_signals(ml_analysis, dynamic_weights["ML"])
        if ml_signals:
            # Combine all ML signals into one signal
            triggered_ml_signals = [s for s in ml_signals if s.triggered]
            if triggered_ml_signals:
                # Calculate combined strength as average of triggered signals
                combined_strength = sum(s.strength for s in triggered_ml_signals) / len(triggered_ml_signals)
                combined_weight = dynamic_weights["ML"]  # Use category weight
                combined_confidence = sum(s.confidence for s in triggered_ml_signals) / len(triggered_ml_signals)
                
                # Create single combined ML signal
                combined_ml_signal = BuySignal(
                    name="ml_combined",
                    strength=combined_strength,
                    weight=combined_weight,
                    triggered=True,
                    reasoning=f"Combined ML signal from {len(triggered_ml_signals)} models",
                    confidence=combined_confidence,
                    category="ML"
                )
                signals.append(combined_ml_signal)

        # 5. Market Structure Signals - Combine all market signals into one
        market_signals = self._generate_market_signals(market_context, dynamic_weights["Market"])
        if market_signals:
            # Combine all market signals into one signal
            triggered_market_signals = [s for s in market_signals if s.triggered]
            if triggered_market_signals:
                # Calculate combined strength as average of triggered signals
                combined_strength = sum(s.strength for s in triggered_market_signals) / len(triggered_market_signals)
                combined_weight = dynamic_weights["Market"]  # Use category weight
                combined_confidence = sum(s.confidence for s in triggered_market_signals) / len(triggered_market_signals)
                
                # Create single combined market signal
                combined_market_signal = BuySignal(
                    name="market_combined",
                    strength=combined_strength,
                    weight=combined_weight,
                    triggered=True,
                    reasoning=f"Combined market signal from {len(triggered_market_signals)} indicators",
                    confidence=combined_confidence,
                    category="Market"
                )
                signals.append(combined_market_signal)

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
        """Cross-category confirmation: At least 3 categories must align"""
        # With combined signals, we check if at least 3 categories have triggered signals
        triggered_signals = [s for s in signals if s.triggered]
        categories = set(s.category for s in triggered_signals if s.category)
        return len(categories) >= 3


    def _calculate_advanced_indicators(self, prices: List[float], volumes: List[float] = None,
                                     highs: List[float] = None, lows: List[float] = None,
                                     put_volume: float = 0, call_volume: float = 0,
                                     bid_volume: float = 0, ask_volume: float = 0,
                                     bid_prices: List[float] = None, ask_prices: List[float] = None) -> Dict[str, Any]:
        """
        Calculate advanced technical indicators
        Returns dictionary with all advanced indicator values and signals
        """
        if not prices:
            return {}

        # Set defaults for optional data
        if volumes is None:
            volumes = [1000000] * len(prices)
        if highs is None:
            highs = prices
        if lows is None:
            lows = prices
        if bid_prices is None:
            bid_prices = []
        if ask_prices is None:
            ask_prices = []

        indicators = {}

        # Money Flow Index (MFI)
        if len(prices) >= 14 and len(volumes) >= 14 and len(highs) >= 14 and len(lows) >= 14:
            typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs[-14:], lows[-14:], prices[-14:])]
            money_flows = []
            for i in range(1, len(typical_prices)):
                if typical_prices[i] > typical_prices[i-1]:
                    money_flows.append(typical_prices[i] * volumes[-14+i])
                elif typical_prices[i] < typical_prices[i-1]:
                    money_flows.append(-typical_prices[i] * volumes[-14+i])
                else:
                    money_flows.append(0)

            positive_flow = sum(mf for mf in money_flows if mf > 0)
            negative_flow = abs(sum(mf for mf in money_flows if mf < 0))

            if negative_flow == 0:
                mfi = 100.0
            else:
                money_ratio = positive_flow / negative_flow
                mfi = 100 - (100 / (1 + money_ratio))

            indicators['mfi'] = round(mfi, 2)
            if mfi < 20:
                indicators['mfi_signal'] = "BULLISH"
            elif mfi > 80:
                indicators['mfi_signal'] = "BEARISH"
            else:
                indicators['mfi_signal'] = "NEUTRAL"

        # Put/Call Ratio
        if call_volume > 0:
            pc_ratio = put_volume / call_volume
            indicators['pc_ratio'] = round(pc_ratio, 3)
            if pc_ratio > 1.2:
                indicators['pc_ratio_signal'] = "BULLISH"  # Contrarian - high fear
            elif pc_ratio < 0.8:
                indicators['pc_ratio_signal'] = "BEARISH"  # Contrarian - high greed
            else:
                indicators['pc_ratio_signal'] = "NEUTRAL"

        # Order Book Analysis
        if bid_volume > 0 or ask_volume > 0:
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
                indicators['order_book_imbalance'] = round(imbalance, 3)

                if abs(imbalance) > 0.3:
                    indicators['order_flow'] = "BULLISH" if imbalance > 0 else "BEARISH"
                else:
                    indicators['order_flow'] = "NEUTRAL"

        # Stochastic Oscillator
        if len(prices) >= 14 and len(highs) >= 14 and len(lows) >= 14:
            current_price = prices[-1]
            lowest_low = min(lows[-14:])
            highest_high = max(highs[-14:])

            if highest_high != lowest_low:
                stoch_k = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100
                indicators['stoch_k'] = round(stoch_k, 2)

                if stoch_k < 20:
                    indicators['stoch_signal'] = "BULLISH"
                elif stoch_k > 80:
                    indicators['stoch_signal'] = "BEARISH"
                else:
                    indicators['stoch_signal'] = "NEUTRAL"

        # Williams %R
        if len(prices) >= 14 and len(highs) >= 14 and len(lows) >= 14:
            current_price = prices[-1]
            highest_high = max(highs[-14:])
            lowest_low = min(lows[-14:])

            if highest_high != lowest_low:
                williams_r = ((highest_high - current_price) / (highest_high - lowest_low)) * -100
                indicators['williams_r'] = round(williams_r, 2)

                if williams_r < -80:
                    indicators['williams_r_signal'] = "BULLISH"
                elif williams_r > -20:
                    indicators['williams_r_signal'] = "BEARISH"
                else:
                    indicators['williams_r_signal'] = "NEUTRAL"

        # Bollinger Bands
        if len(prices) >= 20:
            middle_band = np.mean(prices[-20:])
            std = np.std(prices[-20:])
            upper_band = middle_band + (std * 2)
            lower_band = middle_band - (std * 2)

            current_price = prices[-1]
            if upper_band != lower_band:
                position = (current_price - lower_band) / (upper_band - lower_band)
                indicators['bb_position'] = round(position, 2)
                indicators['bb_upper'] = round(upper_band, 2)
                indicators['bb_middle'] = round(middle_band, 2)
                indicators['bb_lower'] = round(lower_band, 2)

                if position < 0.1:
                    indicators['bb_signal'] = "BULLISH"
                elif position > 0.9:
                    indicators['bb_signal'] = "BEARISH"
                else:
                    indicators['bb_signal'] = "NEUTRAL"

        # Volume Rate of Change
        if len(prices) >= 2 and len(volumes) >= 2:
            current_volume = volumes[-1]
            previous_volume = volumes[-2]
            if previous_volume > 0:
                volume_roc = ((current_volume - previous_volume) / previous_volume) * 100
                indicators['volume_roc'] = round(volume_roc, 2)

    def _check_price_rsi_divergence(self, current_price: float, technical: Dict) -> str:
        """Check for RSI divergence patterns"""
        try:
            # Get price and RSI data for divergence analysis
            recent_prices = technical.get("recent_prices", [])
            recent_rsi = technical.get("recent_rsi", [])

            if len(recent_prices) < 10 or len(recent_rsi) < 10:
                return "insufficient_data"

            # Look for price making lower lows while RSI makes higher lows (bullish divergence)
            price_lows = []
            rsi_lows = []

            # Find local lows in price and RSI
            for i in range(1, len(recent_prices) - 1):
                if (recent_prices[i] < recent_prices[i-1] and
                    recent_prices[i] < recent_prices[i+1] and
                    recent_prices[i] < current_price * 0.98):  # Recent low
                    price_lows.append((i, recent_prices[i]))

            for i in range(1, len(recent_rsi) - 1):
                if (recent_rsi[i] < recent_rsi[i-1] and
                    recent_rsi[i] < recent_rsi[i+1]):
                    rsi_lows.append((i, recent_rsi[i]))

            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                # Check if recent price low is lower but RSI low is higher (bullish divergence)
                recent_price_low = min([p[1] for p in price_lows[-2:]])
                recent_rsi_low = min([r[1] for r in rsi_lows[-2:]])

                older_price_low = min([p[1] for p in price_lows[:-2]])
                older_rsi_low = min([r[1] for r in rsi_lows[:-2]])

                if (recent_price_low < older_price_low and
                    recent_rsi_low > older_rsi_low):
                    return "bullish_divergence"

            # Check for trend direction
            sma_20 = technical.get("sma_20", current_price)
            if current_price > sma_20 * 1.02:
                return "bullish"
            elif current_price < sma_20 * 0.98:
                return "bearish"
            else:
                return "neutral"

        except Exception as e:
            logger.warning(f"Error in divergence analysis: {e}")
            return "neutral"

    def _generate_technical_signals(self, technical: Dict, stock: StockMetrics, category_weight: float = 0.25) -> List[BuySignal]:
        """Generate technical analysis buy signals with enhanced sensitivity"""
        signals = []

        # REALISTIC BUY LOGIC: Enhanced RSI signal detection
        rsi = technical.get("rsi", 50)
        rsi_14 = technical.get("rsi_14", 50)
        rsi_5 = technical.get("rsi_5", 50)

        # RSI Oversold with trend confirmation and divergence detection
        if rsi < 25 and rsi_14 < 30:  # Stricter RSI thresholds
            # Enhanced divergence detection
            price_trend = self._check_price_rsi_divergence(stock.current_price, technical)
            strength = min((25 - rsi) / 15, 1.0) * self.signal_sensitivity_multiplier

            if price_trend == "bullish_divergence":
                strength *= 1.1  # Boost for divergence
                confidence = 0.75
                reasoning = f"RSI oversold at {rsi:.1f} with bullish divergence"
            elif price_trend == "bullish":
                confidence = 0.70
                reasoning = f"RSI oversold at {rsi:.1f} in uptrend"
            else:
                confidence = 0.60
                reasoning = f"RSI oversold at {rsi:.1f}"

            signals.append(BuySignal(
                name="rsi_oversold",
                strength=strength,
                weight=category_weight * 0.12,
                triggered=True,
                reasoning=reasoning,
                confidence=confidence,
                category="Technical"
            ))

        # REALISTIC BUY LOGIC: Enhanced MACD signal detection
        macd = technical.get("macd", 0)
        macd_signal = technical.get("macd_signal", 0)
        macd_hist = technical.get("macd_histogram", 0)

        if macd > macd_signal and macd > 0.5:  # Stricter MACD threshold
            # Check histogram expansion (increasing bullish momentum)
            hist_expansion = macd_hist > technical.get("macd_histogram_prev", 0)
            strength = min(abs(macd - macd_signal) / abs(macd_signal) if macd_signal != 0 else 1, 1.0) * self.signal_sensitivity_multiplier

            if hist_expansion:
                strength *= 1.05
                confidence = 0.65
                reasoning = "MACD bullish crossover with expanding histogram"
            else:
                confidence = 0.60
                reasoning = "MACD bullish crossover"

            signals.append(BuySignal(
                name="macd_bullish",
                strength=strength,
                weight=category_weight * 0.10,
                triggered=True,
                reasoning=reasoning,
                confidence=confidence,
                category="Technical"
            ))

        # REALISTIC BUY LOGIC: Enhanced moving average signals
        sma_20 = technical.get("sma_20", stock.current_price)
        sma_50 = technical.get("sma_50", stock.current_price)
        if stock.current_price > sma_20 * 1.01:  # Stricter condition
            strength = (stock.current_price - sma_20 * 1.01) / (sma_20 * 0.02)
            signals.append(BuySignal(
                name="ma_support",
                strength=min(strength, 1.0) * self.signal_sensitivity_multiplier,
                weight=category_weight * 0.09,
                triggered=True,
                reasoning="Price near key moving averages",
                confidence=0.65,
                category="Technical"
            ))

        # REALISTIC BUY LOGIC: Enhanced support level detection
        support = technical.get("support_level", stock.current_price * 1.1)
        if support > 0 and stock.current_price > support * 1.02:  # Stricter support bounce
            strength = (stock.current_price - support) / support
            signals.append(BuySignal(
                name="support_bounce",
                strength=min(strength * 1.2, 1.0) * self.signal_sensitivity_multiplier,
                weight=category_weight * 0.12,
                triggered=True,
                reasoning=f"Support bounce at {support:.2f}",
                confidence=0.75,
                category="Technical"
            ))

        # REALISTIC BUY LOGIC: Enhanced breakout signal detection
        resistance = technical.get("resistance_level", stock.current_price * 0.9)
        if stock.current_price > resistance * 1.02:  # Stricter breakout detection
            strength = (stock.current_price - resistance) / resistance
            signals.append(BuySignal(
                name="breakout",
                strength=min(strength * 1.1, 1.0) * self.signal_sensitivity_multiplier,
                weight=category_weight * 0.15,
                triggered=True,
                reasoning=f"Breakout above resistance {resistance:.2f}",
                confidence=0.80,
                category="Technical"
            ))

        # ADVANCED TECHNICAL INDICATORS - NEW ADDITIONS

        # Money Flow Index (MFI) signal
        mfi = technical.get("mfi", 50)
        if mfi < 10:  # Stricter oversold condition
            strength = min((10 - mfi) / 10, 1.0) * self.signal_sensitivity_multiplier
            signals.append(BuySignal(
                name="mfi_oversold",
                strength=strength,
                weight=category_weight * 0.10,
                triggered=True,
                reasoning=f"MFI oversold at {mfi:.1f}",
                confidence=0.75,
                category="Technical"
            ))

        # Stochastic Oscillator signal
        stoch_k = technical.get("stoch_k", 50)
        if stoch_k < 10:  # Stricter oversold condition
            strength = min((10 - stoch_k) / 10, 1.0) * self.signal_sensitivity_multiplier
            signals.append(BuySignal(
                name="stoch_oversold",
                strength=strength,
                weight=category_weight * 0.09,
                triggered=True,
                reasoning=f"Stochastic oversold at {stoch_k:.1f}%",
                confidence=0.70,
                category="Technical"
            ))

        # Bollinger Bands signal (position near lower band)
        bb_position = technical.get("bb_position", 0.5)
        if bb_position < 0.02:  # Stricter lower band condition
            strength = min((0.02 - bb_position) / 0.02, 1.0) * self.signal_sensitivity_multiplier
            signals.append(BuySignal(
                name="bb_lower_band",
                strength=strength,
                weight=category_weight * 0.08,
                triggered=True,
                reasoning=f"Price near lower Bollinger Band ({bb_position:.2f})",
                confidence=0.65,
                category="Technical"
            ))

        # Williams %R signal
        williams_r = technical.get("williams_r", -50)
        if williams_r < -90:  # Stricter oversold condition
            strength = min((-90 - williams_r) / 10, 1.0) * self.signal_sensitivity_multiplier
            signals.append(BuySignal(
                name="williams_r_oversold",
                strength=strength,
                weight=category_weight * 0.08,
                triggered=True,
                reasoning=f"Williams %R oversold at {williams_r:.1f}",
                confidence=0.65,
                category="Technical"
            ))

        # Volume Rate of Change signal
        volume_roc = technical.get("volume_roc", 0)
        if volume_roc > 100:  # Stricter volume increase
            strength = min(volume_roc / 200, 1.0) * self.signal_sensitivity_multiplier
            signals.append(BuySignal(
                name="volume_surge",
                strength=strength,
                weight=category_weight * 0.07,
                triggered=True,
                reasoning=f"Volume surge: {volume_roc:.1f}% increase",
                confidence=0.60,
                category="Technical"
            ))

        # Order book imbalance signal (bullish pressure)
        order_imbalance = technical.get("order_book_imbalance", 0)
        if order_imbalance > 0.5:  # Stricter bullish order flow
            strength = min(order_imbalance / 0.7, 1.0) * self.signal_sensitivity_multiplier
            signals.append(BuySignal(
                name="bullish_order_flow",
                strength=strength,
                weight=category_weight * 0.11,
                triggered=True,
                reasoning=f"Bullish order book imbalance: {order_imbalance:.3f}",
                confidence=0.80,
                category="Technical"
            ))

        return signals

    def _generate_value_signals(self, stock: StockMetrics, category_weight: float = 0.20) -> List[BuySignal]:
        """Generate sophisticated value-based buy signals with fundamental analysis"""
        signals = []

        # PEG Ratio Analysis (P/E adjusted for growth)
        # Ensure values are properly converted to floats
        pe_ratio = float(stock.price_to_earnings) if not isinstance(stock.price_to_earnings, (int, float)) else stock.price_to_earnings
        eps_growth = stock.earnings_growth if hasattr(stock, 'earnings_growth') else 0.05  # Default 5% growth
        eps_growth = float(eps_growth) if not isinstance(eps_growth, (int, float)) else eps_growth

        if pe_ratio > 0 and eps_growth > 0:
            peg_ratio = pe_ratio / (eps_growth * 100)  # Convert to percentage
            if peg_ratio < 0.6:  # Stricter PEG < 0.6 indicates undervaluation
                strength = min((0.6 - peg_ratio) / 0.6, 1.0)
                signals.append(BuySignal(
                    name="attractive_peg_ratio",
                    strength=strength,
                    weight=category_weight * 0.12,
                    triggered=True,
                    reasoning=f"Attractive PEG ratio: {peg_ratio:.2f} (P/E: {pe_ratio:.1f}, Growth: {eps_growth:.1%})",
                    confidence=0.65,
                    category="Value"
                ))

        # Enhanced P/E Analysis with sector comparison
        if 0 < pe_ratio < 10:  # Stricter P/E threshold
            # Check if P/E is significantly below sector average
            # Ensure values are properly converted to floats
            sector_pe = stock.sector_pe if hasattr(stock, 'sector_pe') else 20  # Default sector PE
            sector_pe = float(sector_pe) if not isinstance(sector_pe, (int, float)) else sector_pe
            pe_discount = (sector_pe - pe_ratio) / sector_pe if sector_pe > 0 else 0

            if pe_discount > 0.5:  # 50% discount to sector (stricter)
                strength = min(pe_discount, 1.0)
                signals.append(BuySignal(
                    name="low_pe_ratio",
                    strength=strength,
                    weight=category_weight * 0.10,
                    triggered=True,
                    reasoning=f"Low P/E ratio: {pe_ratio:.1f} (vs sector avg {sector_pe:.1f})",
                    confidence=0.60,
                    category="Value"
                ))

        # Enhanced P/B Analysis with ROE consideration
        # Ensure values are properly converted to floats
        pb_ratio = float(stock.price_to_book) if not isinstance(stock.price_to_book, (int, float)) else stock.price_to_book
        roe = stock.return_on_equity if hasattr(stock, 'return_on_equity') else 0.10  # Default 10% ROE
        roe = float(roe) if not isinstance(roe, (int, float)) else roe

        if 0 < pb_ratio < 1.0:  # Stricter P/B threshold
            # Calculate justified P/B based on ROE
            justified_pb = roe / 0.15  # Assuming 15% required return (stricter)
            pb_discount = (justified_pb - pb_ratio) / justified_pb if justified_pb > 0 else 0

            if pb_discount > 0.4:  # 40% discount to justified value (stricter)
                strength = min(pb_discount, 1.0)
                signals.append(BuySignal(
                    name="low_pb_ratio",
                    strength=strength,
                    weight=category_weight * 0.09,
                    triggered=True,
                    reasoning=f"Low P/B ratio: {pb_ratio:.2f} (ROE: {roe:.1%}, justified: {justified_pb:.2f})",
                    confidence=0.60,
                    category="Value"
                ))

        # Free Cash Flow Yield Analysis
        # Ensure values are properly converted to floats
        fcf_yield = stock.free_cash_flow_yield if hasattr(stock, 'free_cash_flow_yield') else 0.05
        fcf_yield = float(fcf_yield) if not isinstance(fcf_yield, (int, float)) else fcf_yield

        if fcf_yield > 0.12:  # 12% FCF yield threshold (stricter)
            strength = min(fcf_yield / 0.20, 1.0)  # Scale to 20% max
            signals.append(BuySignal(
                name="high_fcf_yield",
                strength=strength,
                weight=category_weight * 0.11,
                triggered=True,
                reasoning=f"High free cash flow yield: {fcf_yield:.1%}",
                confidence=0.65,
                category="Value"
            ))

        # Debt-to-Equity Health Check
        # Ensure values are properly converted to floats
        debt_equity = stock.debt_to_equity if hasattr(stock, 'debt_to_equity') else 0.5
        debt_equity = float(debt_equity) if not isinstance(debt_equity, (int, float)) else debt_equity

        if debt_equity < 0.1:  # Conservative debt levels (stricter)
            strength = min((0.1 - debt_equity) / 0.1, 1.0)
            signals.append(BuySignal(
                name="healthy_balance_sheet",
                strength=strength,
                weight=category_weight * 0.08,
                triggered=True,
                reasoning=f"Healthy debt-to-equity ratio: {debt_equity:.2f}",
                confidence=0.55,
                category="Value"
            ))

        # Dividend Sustainability (for dividend stocks)
        # Ensure values are properly converted to floats
        dividend_yield = stock.dividend_yield if hasattr(stock, 'dividend_yield') else 0
        dividend_yield = float(dividend_yield) if not isinstance(dividend_yield, (int, float)) else dividend_yield
        payout_ratio = stock.payout_ratio if hasattr(stock, 'payout_ratio') else 0
        payout_ratio = float(payout_ratio) if not isinstance(payout_ratio, (int, float)) else payout_ratio

        if dividend_yield > 0.04 and payout_ratio < 0.4:  # 4% yield, sustainable payout (stricter)
            strength = min((dividend_yield * (1 - payout_ratio / 0.5)) / 0.07, 1.0)
            signals.append(BuySignal(
                name="sustainable_dividend",
                strength=strength,
                weight=category_weight * 0.07,
                triggered=True,
                reasoning=f"Sustainable dividend: {dividend_yield:.1%} yield, {payout_ratio:.1%} payout",
                confidence=0.50,
                category="Value"
            ))

        # Earnings Quality Score
        # Ensure values are properly converted to floats
        earnings_quality = stock.earnings_quality if hasattr(stock, 'earnings_quality') else 0.5
        earnings_quality = float(earnings_quality) if not isinstance(earnings_quality, (int, float)) else earnings_quality

        if earnings_quality > 0.80:  # High quality earnings (stricter)
            strength = min(earnings_quality / 0.95, 1.0)
            signals.append(BuySignal(
                name="high_earnings_quality",
                strength=strength,
                weight=category_weight * 0.09,
                triggered=True,
                reasoning=f"High earnings quality score: {earnings_quality:.2f}",
                confidence=0.60,
                category="Value"
            ))

        # Insider Ownership Signal
        # Ensure values are properly converted to floats
        insider_ownership = stock.insider_ownership if hasattr(stock, 'insider_ownership') else 0
        insider_ownership = float(insider_ownership) if not isinstance(insider_ownership, (int, float)) else insider_ownership

        if insider_ownership > 0.15:  # 15% insider ownership (stricter)
            strength = min(insider_ownership / 0.25, 1.0)  # Scale to 25% max
            signals.append(BuySignal(
                name="significant_insider_ownership",
                strength=strength,
                weight=category_weight * 0.06,
                triggered=True,
                reasoning=f"Significant insider ownership: {insider_ownership:.1%}",
                confidence=0.55,
                category="Value"
            ))

        return signals

    def _generate_sentiment_signals(self, sentiment: Dict, category_weight: float = 0.20) -> List[BuySignal]:
        """Generate enhanced sentiment-based buy signals with impact scoring"""
        signals = []

        # REALISTIC FIX: Handle different sentiment data structures
        # The sentiment data can come in different formats, so we need to check for all possible keys
        
        # Check for comprehensive sentiment data (from fetch_combined_sentiment)
        if "weighted_aggregated" in sentiment:
            sentiment_data = sentiment["weighted_aggregated"]
            positive = sentiment_data.get("positive", 0)
            negative = sentiment_data.get("negative", 0)
            neutral = sentiment_data.get("neutral", 0)
            total = positive + negative + neutral
            
            if total > 0:
                sentiment_score = (positive - negative) / total
            else:
                sentiment_score = 0
        # Check for aggregated sentiment data
        elif "aggregated" in sentiment:
            sentiment_data = sentiment["aggregated"]
            positive = sentiment_data.get("positive", 0)
            negative = sentiment_data.get("negative", 0)
            neutral = sentiment_data.get("neutral", 0)
            total = positive + negative + neutral
            
            if total > 0:
                sentiment_score = (positive - negative) / total
            else:
                sentiment_score = 0
        # Check for direct sentiment scores
        elif "overall_sentiment" in sentiment:
            sentiment_score = sentiment.get("overall_sentiment", 0)
        # Check for compound sentiment score
        elif "compound" in sentiment:
            sentiment_score = sentiment.get("compound", 0)
        # Default case - try to extract from any available keys
        else:
            positive = sentiment.get("positive", 0)
            negative = sentiment.get("negative", 0)
            neutral = sentiment.get("neutral", 0)
            total = positive + negative + neutral
            
            if total > 0:
                sentiment_score = (positive - negative) / total
            else:
                sentiment_score = 0

        # Enhanced Positive Sentiment with momentum
        sentiment_momentum = sentiment.get("sentiment_momentum", 0)

        if sentiment_score > 0.4:  # Stricter positive sentiment threshold
            # Boost signal if sentiment is improving
            momentum_boost = 1.05 if sentiment_momentum > 0.2 else 1.0  # Stricter momentum threshold
            strength = min(sentiment_score / 0.8, 1.0) * momentum_boost

            reasoning = f"Positive sentiment: {sentiment_score:.2f}"
            if sentiment_momentum > 0.2:
                reasoning += f" (improving momentum: {sentiment_momentum:.2f})"

            signals.append(BuySignal(
                name="positive_sentiment",
                strength=strength,
                weight=category_weight * 0.15,
                triggered=True,
                reasoning=reasoning,
                confidence=0.55,
                category="Sentiment"
            ))

        # Enhanced Negative Sentiment Protection (contrarian indicator)
        if sentiment_score < -0.4:  # Stricter negative sentiment threshold
            strength = min(abs(sentiment_score) / 0.8, 1.0)
            signals.append(BuySignal(
                name="negative_sentiment_extreme",
                strength=strength,
                weight=category_weight * 0.10,
                triggered=True,
                reasoning=f"Extreme negative sentiment: {sentiment_score:.2f} (contrarian opportunity)",
                confidence=0.60,
                category="Sentiment"
            ))

        # News Sentiment Trend Analysis
        news_sentiment = sentiment.get("news_sentiment", {})
        if isinstance(news_sentiment, dict):
            news_positive = news_sentiment.get("positive", 0)
            news_negative = news_sentiment.get("negative", 0)
            news_total = news_positive + news_negative + news_sentiment.get("neutral", 0)
            
            if news_total > 0:
                news_score = (news_positive - news_negative) / news_total
                if news_score > 0.5:  # Stricter news sentiment threshold
                    strength = min(news_score / 0.7, 1.0)
                    signals.append(BuySignal(
                        name="positive_news_sentiment",
                        strength=strength,
                        weight=category_weight * 0.12,
                        triggered=True,
                        reasoning=f"Positive news sentiment: {news_score:.2f}",
                        confidence=0.50,
                        category="Sentiment"
                    ))

        # Social Media Sentiment Analysis
        social_sentiment = sentiment.get("social_sentiment", {})
        if isinstance(social_sentiment, dict):
            social_positive = social_sentiment.get("positive", 0)
            social_negative = social_sentiment.get("negative", 0)
            social_total = social_positive + social_negative + social_sentiment.get("neutral", 0)
            
            if social_total > 0:
                social_score = (social_positive - social_negative) / social_total
                if social_score > 0.6:  # Stricter social sentiment threshold
                    strength = min(social_score / 0.8, 1.0)
                    signals.append(BuySignal(
                        name="positive_social_sentiment",
                        strength=strength,
                        weight=category_weight * 0.10,
                        triggered=True,
                        reasoning=f"Positive social sentiment: {social_score:.2f}",
                        confidence=0.45,
                        category="Sentiment"
                    ))

        # Options Flow Sentiment (institutional activity)
        options_flow = sentiment.get("options_flow_sentiment", 0)
        call_put_ratio = sentiment.get("call_put_ratio", 1.0)

        if options_flow > 0.4 and call_put_ratio > 1.3:  # Stricter options flow threshold
            strength = min((options_flow * (call_put_ratio - 1)) / 0.7, 1.0)
            signals.append(BuySignal(
                name="bullish_options_flow",
                strength=strength,
                weight=category_weight * 0.13,
                triggered=True,
                reasoning=f"Bullish options flow: {options_flow:.2f} (call/put: {call_put_ratio:.2f})",
                confidence=0.70,
                category="Sentiment"
            ))

        # Insider Trading Signal Aggregation
        insider_activity = sentiment.get("insider_activity_score", 0)
        recent_insider_buys = sentiment.get("recent_insider_buys", 0)

        if insider_activity > 0.5 and recent_insider_buys > 2:  # Stricter insider activity threshold
            strength = min((insider_activity * recent_insider_buys) / 10, 1.0)
            signals.append(BuySignal(
                name="insider_buying_activity",
                strength=strength,
                weight=category_weight * 0.09,
                triggered=True,
                reasoning=f"Insider buying activity: {insider_activity:.2f} (recent buys: {recent_insider_buys})",
                confidence=0.60,
                category="Sentiment"
            ))

        # Enhanced Put/Call Ratio (contrarian sentiment)
        pc_ratio = sentiment.get("pc_ratio", 1.0)
        pc_trend = sentiment.get("pc_trend", 0)

        if pc_ratio > 1.4:  # High put volume (fear) - stricter threshold
            # Boost if put/call ratio is trending higher (increasing fear)
            trend_boost = 1.05 if pc_trend > 0.1 else 1.0  # Stricter trend threshold
            strength = min((pc_ratio - 1.0) / 0.7, 1.0) * trend_boost

            reasoning = f"High Put/Call ratio ({pc_ratio:.3f}) indicates fear"
            if pc_trend > 0.1:
                reasoning += " (increasing fear trend)"

            signals.append(BuySignal(
                name="high_put_call_ratio",
                strength=strength,
                weight=category_weight * 0.12,
                triggered=True,
                reasoning=reasoning,
                confidence=0.65,
                category="Sentiment"
            ))

        # Short Interest Trend (contrarian indicator)
        short_interest = sentiment.get("short_interest_pct", 0)
        short_trend = sentiment.get("short_trend", 0)

        if short_interest > 0.10 and short_trend < -0.04:  # High short interest, declining - stricter thresholds
            strength = min((short_interest * abs(short_trend)) / 0.004, 1.0)
            signals.append(BuySignal(
                name="declining_short_interest",
                strength=strength,
                weight=category_weight * 0.08,
                triggered=True,
                reasoning=f"High short interest {short_interest:.1%} with declining trend",
                confidence=0.55,
                category="Sentiment"
            ))

        return signals

    def _generate_ml_signals(self, ml_analysis: Dict, category_weight: float = 0.20) -> List[BuySignal]:
        """Generate ML/AI-based buy signals with confidence weighting and validation"""
        signals = []

        # REALISTIC FIX: Handle different ML data structures and missing data gracefully
        
        # Enhanced ML Prediction with ensemble confidence
        # Handle different possible keys for ML prediction
        ml_prediction = (ml_analysis.get("prediction_direction", 0) or 
                        ml_analysis.get("predicted_direction", 0) or
                        ml_analysis.get("direction", 0))
        
        # Handle different possible keys for ML confidence
        ml_confidence = (ml_analysis.get("confidence", 0.5) or
                        ml_analysis.get("model_confidence", 0.5) or
                        ml_analysis.get("prediction_confidence", 0.5))
        
        # Handle different possible keys for model accuracy
        model_accuracy = (ml_analysis.get("model_accuracy", 0.7) or
                         ml_analysis.get("accuracy", 0.7) or
                         ml_analysis.get("r2_score", 0.7))

        # Check if ML analysis was successful
        ml_success = ml_analysis.get("success", True)  # Default to True if not specified
        
        # REALISTIC BUY LOGIC: Normalize ML values to prevent extreme inflation
        # Handle the case where ML values are extremely large
        if isinstance(ml_prediction, (int, float)) and abs(ml_prediction) > 1000:
            # Normalize extremely large values
            ml_prediction = np.sign(ml_prediction) * min(abs(ml_prediction) / 1000000, 1.0)
            logger.warning(f"ML prediction value normalized from {ml_analysis.get('prediction_direction', 0)} to {ml_prediction}")
        
        if isinstance(ml_confidence, (int, float)) and ml_confidence > 1000:
            # Normalize extremely large confidence values
            ml_confidence = min(ml_confidence / 1000000, 1.0)
            logger.warning(f"ML confidence value normalized from {ml_analysis.get('confidence', 0)} to {ml_confidence}")
        
        # Cap confidence and accuracy values to prevent extreme values
        ml_confidence = min(ml_confidence, 1.0)
        model_accuracy = min(model_accuracy, 1.0)
        
        if ml_success and ml_prediction > 0.03:  # Stricter ML prediction threshold
            # Calculate weighted confidence based on model accuracy and prediction strength
            prediction_strength = abs(ml_prediction)
            weighted_confidence = (ml_confidence * 0.6) + (model_accuracy * 0.4)  # Weighted ensemble
            # Cap weighted confidence
            weighted_confidence = min(weighted_confidence, 1.0)

            # Boost for strong predictions from accurate models
            accuracy_boost = 1.0 + (model_accuracy - 0.75) * 0.2 if model_accuracy > 0.75 else 1.0

            strength = min(prediction_strength / 0.06, 1.0) * accuracy_boost * self.signal_sensitivity_multiplier

            signals.append(BuySignal(
                name="ml_bullish_prediction",
                strength=strength,
                weight=category_weight * (0.15 + self.ml_signal_weight_boost),
                triggered=True,
                confidence=weighted_confidence,
                reasoning=f"ML prediction: {ml_prediction:.1%} (confidence: {ml_confidence:.2f}, accuracy: {model_accuracy:.2f})",
                category="ML"
            ))

        # Multi-model ensemble signal
        ensemble_prediction = (ml_analysis.get("ensemble_prediction", 0) or
                              ml_analysis.get("ensemble_direction", 0))
        ensemble_models = (ml_analysis.get("ensemble_models_count", 1) or
                          ml_analysis.get("models_count", 1))

        # REALISTIC BUY LOGIC: Normalize ensemble values
        if isinstance(ensemble_prediction, (int, float)) and abs(ensemble_prediction) > 1000:
            ensemble_prediction = np.sign(ensemble_prediction) * min(abs(ensemble_prediction) / 1000000, 1.0)
            logger.warning(f"Ensemble prediction normalized from {ml_analysis.get('ensemble_prediction', 0)} to {ensemble_prediction}")

        if ensemble_prediction > 0.04 and ensemble_models > 3:  # Stricter ensemble threshold
            # Higher confidence for multi-model agreement
            model_agreement = min(ensemble_models / 5, 1.0)  # Scale based on model count
            strength = min(abs(ensemble_prediction) / 0.07, 1.0) * (1 + model_agreement * 0.15)

            signals.append(BuySignal(
                name="ensemble_bullish_consensus",
                strength=strength,
                weight=category_weight * (0.12 + self.ml_signal_weight_boost),
                triggered=True,
                confidence=min(ml_analysis.get("ensemble_confidence", 0.6), 1.0),  # Cap confidence
                reasoning=f"Multi-model consensus: {ensemble_prediction:.1%} ({ensemble_models} models)",
                category="ML"
            ))

        # Enhanced RL recommendation with action confidence
        rl_recommendation = (ml_analysis.get("rl_recommendation", "HOLD") or
                            ml_analysis.get("reinforcement_learning_recommendation", "HOLD"))
        rl_confidence = (ml_analysis.get("rl_confidence", 0.5) or
                        ml_analysis.get("reinforcement_learning_confidence", 0.5))
        rl_sharpe_ratio = (ml_analysis.get("rl_sharpe_ratio", 0.5) or
                          ml_analysis.get("reinforcement_learning_sharpe", 0.5))

        # Cap RL values
        rl_confidence = min(rl_confidence, 1.0)
        rl_sharpe_ratio = min(rl_sharpe_ratio, 1.0)

        if rl_recommendation in ["BUY", "STRONG_BUY"]:
            # Boost confidence for strong recommendations from high-performing strategies
            performance_boost = 1.0 + (rl_sharpe_ratio - 0.6) * 0.2 if rl_sharpe_ratio > 0.6 else 1.0
            confidence_multiplier = 1.2 if rl_recommendation == "STRONG_BUY" else 1.05

            strength = rl_confidence * performance_boost * self.signal_sensitivity_multiplier

            signals.append(BuySignal(
                name="rl_buy_recommendation",
                strength=strength,
                weight=category_weight * (0.10 + self.ml_signal_weight_boost * 0.5),
                triggered=True,
                confidence=min(rl_confidence * confidence_multiplier, 1.0),  # Cap confidence
                reasoning=f"RL recommendation: {rl_recommendation} (confidence: {rl_confidence:.2f}, Sharpe: {rl_sharpe_ratio:.2f})",
                category="ML"
            ))

        # Feature importance-based signal
        feature_importance = ml_analysis.get("feature_importance_score", 0)
        key_features = ml_analysis.get("key_bullish_features", [])

        # Cap feature importance
        feature_importance = min(feature_importance, 1.0)

        if feature_importance > 0.75 and len(key_features) > 2:  # Stricter feature importance threshold
            # Weight by number of important features
            feature_boost = min(len(key_features) / 8, 1.0)  # Up to 8 key features
            strength = min(feature_importance / 0.95, 1.0) * (1 + feature_boost * 0.1)

            signals.append(BuySignal(
                name="key_feature_alignment",
                strength=strength,
                weight=category_weight * 0.08,
                triggered=True,
                confidence=feature_importance,
                reasoning=f"Key features aligned: {feature_importance:.2f} (features: {len(key_features)})",
                category="ML"
            ))

        # Cross-validation score signal
        cv_score = ml_analysis.get("cross_validation_score", 0)
        backtest_performance = ml_analysis.get("backtest_performance", 0)

        # Cap scores
        cv_score = min(cv_score, 1.0)
        backtest_performance = min(backtest_performance, 1.0)

        if cv_score > 0.75 and backtest_performance > 0.65:  # Stricter validation thresholds
            # Combined validation metric
            validation_score = (cv_score * 0.6) + (backtest_performance * 0.4)
            strength = min(validation_score / 0.8, 1.0)

            signals.append(BuySignal(
                name="validated_ml_signal",
                strength=strength,
                weight=category_weight * 0.09,
                triggered=True,
                confidence=min(validation_score, 1.0),  # Cap confidence
                reasoning=f"Validated ML signal: CV {cv_score:.2f}, Backtest {backtest_performance:.2f}",
                category="ML"
            ))

        # Real-time model performance tracking
        recent_accuracy = ml_analysis.get("recent_accuracy", 0.5)
        prediction_consistency = ml_analysis.get("prediction_consistency", 0.5)

        # Cap performance metrics
        recent_accuracy = min(recent_accuracy, 1.0)
        prediction_consistency = min(prediction_consistency, 1.0)

        if recent_accuracy > 0.80 and prediction_consistency > 0.70:  # Stricter performance thresholds
            strength = min((recent_accuracy * prediction_consistency) / 0.7, 1.0)

            signals.append(BuySignal(
                name="reliable_ml_performance",
                strength=strength,
                weight=category_weight * 0.07,
                triggered=True,
                confidence=min((recent_accuracy + prediction_consistency) / 2, 1.0),  # Cap confidence
                reasoning=f"Reliable ML performance: accuracy {recent_accuracy:.2f}, consistency {prediction_consistency:.2f}",
                category="ML"
            ))

        # Price prediction signal
        current_price = ml_analysis.get("current_price", 0)
        predicted_price = ml_analysis.get("predicted_price", current_price)
        
        if current_price > 0 and predicted_price > 0:
            price_change_pct = (predicted_price - current_price) / current_price
            if price_change_pct > 0.04:  # 4% predicted gain (stricter)
                strength = min(price_change_pct / 0.10, 1.0)  # Scale to 10% max gain
                confidence = ml_analysis.get("prediction_confidence", 0.5)
                # Cap confidence
                confidence = min(confidence, 1.0)
                
                signals.append(BuySignal(
                    name="positive_price_prediction",
                    strength=strength,
                    weight=category_weight * 0.11,
                    triggered=True,
                    confidence=confidence,
                    reasoning=f"Positive price prediction: {price_change_pct:.1%} gain expected",
                    category="ML"
                ))

        return signals

    def _generate_market_signals(self, market_context: MarketContext, category_weight: float = 0.15) -> List[BuySignal]:
        """Generate advanced market structure buy signals with sector rotation analysis"""
        signals = []

        # Enhanced Market Stress Analysis
        if market_context.market_stress < 0.25:  # Stricter low stress threshold
            # Check if stress is declining (improving conditions)
            stress_trend = market_context.market_stress_trend if hasattr(market_context, 'market_stress_trend') else 0
            trend_improvement = stress_trend < -0.04  # Declining stress trend (stricter)

            strength = (0.25 - market_context.market_stress) / 0.25
            if trend_improvement:
                strength *= 1.05  # Boost for improving conditions

            signals.append(BuySignal(
                name="low_market_stress",
                strength=strength,
                weight=category_weight * 0.10,
                triggered=True,
                reasoning=f"Low market stress: {market_context.market_stress:.1%}" +
                         (" (improving)" if trend_improvement else ""),
                confidence=0.65,
                category="Market"
            ))

        # Sector Rotation Analysis (cyclical vs defensive)
        sector_performance = market_context.sector_performance if hasattr(market_context, 'sector_performance') else 0
        sector_rotation_score = market_context.sector_rotation_score if hasattr(market_context, 'sector_rotation_score') else 0

        if sector_performance > 0.04:  # Stricter sector performance threshold
            # Analyze if we're in favorable sector rotation
            rotation_quality = sector_rotation_score if sector_rotation_score > 0 else 0.5
            strength = min(sector_performance / 0.07, 1.0) * (1 + rotation_quality * 0.2)

            signals.append(BuySignal(
                name="favorable_sector_rotation",
                strength=strength,
                weight=category_weight * 0.12,
                triggered=True,
                reasoning=f"Favorable sector rotation: {sector_performance:.1%} (quality: {rotation_quality:.2f})",
                confidence=0.60,
                category="Market"
            ))

        # Intermarket Correlation Analysis
        intermarket_correlation = market_context.intermarket_correlation if hasattr(market_context, 'intermarket_correlation') else 0.5
        commodity_correlation = market_context.commodity_correlation if hasattr(market_context, 'commodity_correlation') else 0

        if intermarket_correlation < 0.2:  # Low correlation indicates diversification opportunity (stricter)
            strength = (0.2 - intermarket_correlation) / 0.2
            signals.append(BuySignal(
                name="diversification_opportunity",
                strength=strength,
                weight=category_weight * 0.08,
                triggered=True,
                reasoning=f"Low intermarket correlation: {intermarket_correlation:.2f}",
                confidence=0.55,
                category="Market"
            ))

        # Market Breadth Indicators
        market_breadth = market_context.market_breadth if hasattr(market_context, 'market_breadth') else 0.5
        advancing_stocks = market_context.advancing_stocks if hasattr(market_context, 'advancing_stocks') else 0

        if market_breadth > 0.65 and advancing_stocks > 0.70:  # Stricter breadth thresholds
            strength = min((market_breadth * advancing_stocks) / 0.5, 1.0)
            signals.append(BuySignal(
                name="strong_market_breadth",
                strength=strength,
                weight=category_weight * 0.09,
                triggered=True,
                reasoning=f"Strong market breadth: {market_breadth:.2f} (advancing: {advancing_stocks:.1%})",
                confidence=0.60,
                category="Market"
            ))

        # Volatility Regime Analysis
        volatility_regime = market_context.volatility_regime if hasattr(market_context, 'volatility_regime') else "normal"
        vix_level = market_context.vix_level if hasattr(market_context, 'vix_level') else 20

        if volatility_regime == "low_volatility" and vix_level < 12:  # Stricter low volatility threshold
            strength = (12 - vix_level) / 12  # Scale VIX inversely
            signals.append(BuySignal(
                name="low_volatility_regime",
                strength=strength,
                weight=category_weight * 0.11,
                triggered=True,
                reasoning=f"Low volatility regime: VIX {vix_level:.1f}",
                confidence=0.65,
                category="Market"
            ))

        # Economic Indicator Alignment
        economic_alignment = market_context.economic_alignment if hasattr(market_context, 'economic_alignment') else 0
        leading_indicators = market_context.leading_indicators if hasattr(market_context, 'leading_indicators') else 0

        if economic_alignment > 0.70 and leading_indicators > 0.60:  # Stricter economic alignment thresholds
            strength = min((economic_alignment * leading_indicators) / 0.5, 1.0)
            signals.append(BuySignal(
                name="positive_economic_alignment",
                strength=strength,
                weight=category_weight * 0.10,
                triggered=True,
                reasoning=f"Positive economic alignment: {economic_alignment:.2f} (leading: {leading_indicators:.2f})",
                confidence=0.55,
                category="Market"
            ))

        # Currency Strength Impact
        currency_impact = market_context.currency_impact if hasattr(market_context, 'currency_impact') else 0
        usd_strength = market_context.usd_strength if hasattr(market_context, 'usd_strength') else 0

        if currency_impact > 0.20 and usd_strength < 0.40:  # Stricter currency impact thresholds
            strength = min(currency_impact / 0.4, 1.0)
            signals.append(BuySignal(
                name="favorable_currency_impact",
                strength=strength,
                weight=category_weight * 0.07,
                triggered=True,
                reasoning=f"Favorable currency impact: {currency_impact:.2f} (USD: {usd_strength:.2f})",
                confidence=0.50,
                category="Market"
            ))

        # Bond Market Signal (risk-on/risk-off indicator)
        bond_yield_trend = market_context.bond_yield_trend if hasattr(market_context, 'bond_yield_trend') else 0
        treasury_spread = market_context.treasury_spread if hasattr(market_context, 'treasury_spread') else 0

        if bond_yield_trend < -0.10 and treasury_spread > 2.0:  # Stricter bond market thresholds
            strength = min(abs(bond_yield_trend) / 0.15, 1.0) * (treasury_spread / 4.0)
            signals.append(BuySignal(
                name="risk_on_environment",
                strength=strength,
                weight=category_weight * 0.08,
                triggered=True,
                reasoning=f"Risk-on environment: yield trend {bond_yield_trend:.3f}, spread {treasury_spread:.2f}%",
                confidence=0.60,
                category="Market"
            ))

        # Global Market Momentum
        global_momentum = market_context.global_momentum if hasattr(market_context, 'global_momentum') else 0
        emerging_markets = market_context.emerging_markets_performance if hasattr(market_context, 'emerging_markets_performance') else 0

        if global_momentum > 0.04 and emerging_markets > 0.03:  # Stricter global momentum thresholds
            strength = min((global_momentum * emerging_markets) / 0.0005, 1.0)
            signals.append(BuySignal(
                name="global_market_momentum",
                strength=strength,
                weight=category_weight * 0.06,
                triggered=True,
                reasoning=f"Global momentum: {global_momentum:.1%} (emerging: {emerging_markets:.1%})",
                confidence=0.50,
                category="Market"
            ))

        return signals

    def _calculate_optimized_entry_levels(self, stock: StockMetrics, market_context: MarketContext) -> Dict:
        """Calculate optimized entry levels for better timing"""
        
        # REALISTIC BUY LOGIC: Earlier entry opportunities
        base_entry = stock.current_price * (1 - self.early_entry_buffer_pct)

        # Volatility-adjusted entry with enhanced sensitivity
        volatility_multiplier = 1 + (stock.volatility / 0.03) * 0.3  # Reduced multiplier for earlier entries
        volatility_entry = stock.current_price * (1 - self.early_entry_buffer_pct * volatility_multiplier)

        # Support-based entry with buffer
        support_entry = stock.support_level * 1.02 if stock.support_level > 0 else stock.current_price  # Reduced buffer

        # Choose the most aggressive (lowest) entry for better timing
        target_entry = min(base_entry, volatility_entry, support_entry)

        # Stop-loss based on dynamic percentage from frontend
        stop_loss = target_entry * (1 - self.stop_loss_pct)

        # Take-profit based on dynamic target price percentage from frontend
        # Simple calculation: entry price + target price percentage
        take_profit = target_entry * (1 + self.target_price_pct)

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
            decision.urgency = min(decision.urgency * 1.05, 1.0)
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
        position_scale = min(weighted_score * 1.0, 1.0)  # Reduced multiplier for better scaling
        position_scale = max(position_scale, 0.1)  # Minimum 10% position
        logger.info(f"Base position scale from weighted score: {position_scale:.3f}")

        # REALISTIC BUY LOGIC: Boost position size for high-confidence ML signals
        ml_signals = [s for s in triggered_signals if s.category == "ML"]
        if ml_signals:
            ml_confidence = np.mean([s.confidence for s in ml_signals])
            # Cap ML confidence to prevent extreme values
            ml_confidence = min(ml_confidence, 1.0)
            logger.info(f"ML signals detected - average confidence: {ml_confidence:.3f}")
            if ml_confidence > 0.80:  # Stricter ML confidence threshold
                original_scale = position_scale
                position_scale *= 1.15  # 15% boost for high-confidence ML signals
                logger.info(f"High-confidence ML boost applied - position scale increased from {original_scale:.3f} to {position_scale:.3f}")
            elif ml_confidence > 0.65:  # Medium confidence threshold
                original_scale = position_scale
                position_scale *= 1.05  # 5% boost for medium-confidence ML signals
                logger.info(f"Medium-confidence ML boost applied - position scale increased from {original_scale:.3f} to {position_scale:.3f}")

        # REALISTIC BUY LOGIC: Adjust for aggressive entry opportunities
        if weighted_score > self.aggressive_entry_threshold:
            original_scale = position_scale
            position_scale *= 1.10  # 10% boost for high-conviction setups
            logger.info(f"Aggressive entry boost applied - position scale increased from {original_scale:.3f} to {position_scale:.3f}")

        # Cap position size
        position_scale = min(position_scale, 1.0)
        logger.info(f"Final position scale (capped): {position_scale:.3f}")

        # Set position scale (the actual quantity will be calculated in the integration layer)
        decision.buy_quantity = 1  # Placeholder - actual quantity calculated in integration layer
        decision.buy_percentage = position_scale
        decision.reasoning += f" | ENHANCED POSITION SIZING ({position_scale:.1%})"

        return decision
