"""
Production-Level Adaptive Threshold Manager
Dynamic thresholds based on market conditions
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    RANGING = "ranging"

@dataclass
class MarketContext:
    """Market context for threshold adaptation"""
    regime: MarketRegime
    volatility_percentile: float  # 0-100
    trend_strength: float  # -1 to 1
    volume_profile: float  # 0-1 (relative to average)
    time_of_day: str  # 'open', 'mid', 'close'
    market_stress_level: float  # 0-1
    confidence: float  # 0-1

@dataclass
class AdaptiveThresholds:
    """Adaptive threshold values"""
    buy_threshold: float
    sell_threshold: float
    confidence_minimum: float
    position_size_multiplier: float
    reasoning: str
    base_thresholds: Dict[str, float]
    adjustments: Dict[str, float]

class AdaptiveThresholdManager:
    """Production-level adaptive threshold management"""
    
    def __init__(self):
        # Base thresholds (conservative defaults)
        self.base_thresholds = {
            'buy_threshold': 0.65,
            'sell_threshold': -0.65,
            'confidence_minimum': 0.60,
            'position_size_multiplier': 1.0
        }
        
        # Market regime adjustments
        self.regime_adjustments = {
            MarketRegime.BULL_MARKET: {
                'buy_threshold': 0.85,    # Easier to buy
                'sell_threshold': 0.95,   # Harder to sell
                'confidence_minimum': 0.90,
                'position_size_multiplier': 1.1
            },
            MarketRegime.BEAR_MARKET: {
                'buy_threshold': 1.15,    # Harder to buy
                'sell_threshold': 0.85,   # Easier to sell
                'confidence_minimum': 1.05,
                'position_size_multiplier': 0.8
            },
            MarketRegime.HIGH_VOLATILITY: {
                'buy_threshold': 1.25,    # Much stricter
                'sell_threshold': 1.25,
                'confidence_minimum': 1.20,
                'position_size_multiplier': 0.7
            },
            MarketRegime.LOW_VOLATILITY: {
                'buy_threshold': 0.90,    # Slightly easier
                'sell_threshold': 0.90,
                'confidence_minimum': 0.95,
                'position_size_multiplier': 1.05
            },
            MarketRegime.SIDEWAYS: {
                'buy_threshold': 1.10,    # More conservative
                'sell_threshold': 1.10,
                'confidence_minimum': 1.10,
                'position_size_multiplier': 0.9
            }
        }
        
        # Time-of-day adjustments
        self.time_adjustments = {
            'market_open': {  # First 30 minutes
                'buy_threshold': 1.15,
                'sell_threshold': 1.15,
                'confidence_minimum': 1.10,
                'position_size_multiplier': 0.8
            },
            'market_close': {  # Last 30 minutes
                'buy_threshold': 1.10,
                'sell_threshold': 1.10,
                'confidence_minimum': 1.05,
                'position_size_multiplier': 0.9
            },
            'mid_day': {  # Normal trading hours
                'buy_threshold': 1.0,
                'sell_threshold': 1.0,
                'confidence_minimum': 1.0,
                'position_size_multiplier': 1.0
            }
        }
        
        # Performance tracking
        self.threshold_performance = {}
        self.adaptation_history = []
        
    async def get_adaptive_thresholds(self, market_context: MarketContext, symbol: str = None) -> AdaptiveThresholds:
        """Get adaptive thresholds based on market context"""
        
        logger.debug(f"Calculating adaptive thresholds for regime: {market_context.regime}")
        
        # Start with base thresholds
        thresholds = self.base_thresholds.copy()
        adjustments = {}
        reasoning_parts = []
        
        # Apply market regime adjustments
        regime_adj = self.regime_adjustments.get(market_context.regime, {})
        for key, multiplier in regime_adj.items():
            if key in thresholds:
                old_value = thresholds[key]
                thresholds[key] *= multiplier
                adjustments[f'regime_{key}'] = multiplier
                reasoning_parts.append(f"{market_context.regime.value}: {key} {old_value:.3f} -> {thresholds[key]:.3f}")
        
        # Apply time-of-day adjustments
        time_adj = self.time_adjustments.get(market_context.time_of_day, {})
        for key, multiplier in time_adj.items():
            if key in thresholds:
                old_value = thresholds[key]
                thresholds[key] *= multiplier
                adjustments[f'time_{key}'] = multiplier
                reasoning_parts.append(f"{market_context.time_of_day}: {key} {old_value:.3f} -> {thresholds[key]:.3f}")
        
        # Apply volatility-based adjustments
        volatility_adj = self._calculate_volatility_adjustments(market_context.volatility_percentile)
        for key, multiplier in volatility_adj.items():
            if key in thresholds:
                old_value = thresholds[key]
                thresholds[key] *= multiplier
                adjustments[f'volatility_{key}'] = multiplier
                reasoning_parts.append(f"Volatility {market_context.volatility_percentile:.1f}%: {key} {old_value:.3f} -> {thresholds[key]:.3f}")
        
        # Apply trend strength adjustments
        trend_adj = self._calculate_trend_adjustments(market_context.trend_strength)
        for key, multiplier in trend_adj.items():
            if key in thresholds:
                old_value = thresholds[key]
                thresholds[key] *= multiplier
                adjustments[f'trend_{key}'] = multiplier
                reasoning_parts.append(f"Trend {market_context.trend_strength:.2f}: {key} {old_value:.3f} -> {thresholds[key]:.3f}")
        
        # Apply market stress adjustments
        stress_adj = self._calculate_stress_adjustments(market_context.market_stress_level)
        for key, multiplier in stress_adj.items():
            if key in thresholds:
                old_value = thresholds[key]
                thresholds[key] *= multiplier
                adjustments[f'stress_{key}'] = multiplier
                reasoning_parts.append(f"Stress {market_context.market_stress_level:.2f}: {key} {old_value:.3f} -> {thresholds[key]:.3f}")
        
        # Ensure thresholds are within reasonable bounds
        thresholds = self._apply_threshold_bounds(thresholds)
        
        # Create adaptive thresholds object
        adaptive_thresholds = AdaptiveThresholds(
            buy_threshold=thresholds['buy_threshold'],
            sell_threshold=thresholds['sell_threshold'],
            confidence_minimum=thresholds['confidence_minimum'],
            position_size_multiplier=thresholds['position_size_multiplier'],
            reasoning=" | ".join(reasoning_parts) if reasoning_parts else "No adjustments applied",
            base_thresholds=self.base_thresholds.copy(),
            adjustments=adjustments
        )
        
        # Store adaptation history
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'market_context': market_context,
            'thresholds': thresholds,
            'symbol': symbol
        })
        
        # Keep only last 100 adaptations
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-100:]
        
        logger.info(f"Adaptive thresholds: BUY={adaptive_thresholds.buy_threshold:.3f}, "
                   f"SELL={adaptive_thresholds.sell_threshold:.3f}, "
                   f"CONF={adaptive_thresholds.confidence_minimum:.3f}")
        
        return adaptive_thresholds
    
    def _calculate_volatility_adjustments(self, volatility_percentile: float) -> Dict[str, float]:
        """Calculate adjustments based on volatility percentile"""
        adjustments = {}
        
        if volatility_percentile > 80:  # Very high volatility
            adjustments['buy_threshold'] = 1.3
            adjustments['sell_threshold'] = 1.3
            adjustments['confidence_minimum'] = 1.25
            adjustments['position_size_multiplier'] = 0.6
        elif volatility_percentile > 60:  # High volatility
            adjustments['buy_threshold'] = 1.15
            adjustments['sell_threshold'] = 1.15
            adjustments['confidence_minimum'] = 1.10
            adjustments['position_size_multiplier'] = 0.8
        elif volatility_percentile < 20:  # Very low volatility
            adjustments['buy_threshold'] = 0.85
            adjustments['sell_threshold'] = 0.85
            adjustments['confidence_minimum'] = 0.90
            adjustments['position_size_multiplier'] = 1.1
        elif volatility_percentile < 40:  # Low volatility
            adjustments['buy_threshold'] = 0.95
            adjustments['sell_threshold'] = 0.95
            adjustments['confidence_minimum'] = 0.95
            adjustments['position_size_multiplier'] = 1.05
        
        return adjustments
    
    def _calculate_trend_adjustments(self, trend_strength: float) -> Dict[str, float]:
        """Calculate adjustments based on trend strength"""
        adjustments = {}
        
        if trend_strength > 0.7:  # Strong uptrend
            adjustments['buy_threshold'] = 0.9   # Easier to buy
            adjustments['sell_threshold'] = 1.2  # Harder to sell
        elif trend_strength < -0.7:  # Strong downtrend
            adjustments['buy_threshold'] = 1.2   # Harder to buy
            adjustments['sell_threshold'] = 0.9  # Easier to sell
        elif abs(trend_strength) < 0.2:  # No clear trend
            adjustments['buy_threshold'] = 1.1   # More conservative
            adjustments['sell_threshold'] = 1.1
            adjustments['confidence_minimum'] = 1.05
        
        return adjustments
    
    def _calculate_stress_adjustments(self, stress_level: float) -> Dict[str, float]:
        """Calculate adjustments based on market stress level"""
        adjustments = {}
        
        if stress_level > 0.8:  # High stress
            adjustments['buy_threshold'] = 1.4
            adjustments['sell_threshold'] = 1.4
            adjustments['confidence_minimum'] = 1.3
            adjustments['position_size_multiplier'] = 0.5
        elif stress_level > 0.6:  # Medium stress
            adjustments['buy_threshold'] = 1.2
            adjustments['sell_threshold'] = 1.2
            adjustments['confidence_minimum'] = 1.15
            adjustments['position_size_multiplier'] = 0.7
        elif stress_level < 0.2:  # Low stress
            adjustments['buy_threshold'] = 0.9
            adjustments['sell_threshold'] = 0.9
            adjustments['confidence_minimum'] = 0.95
            adjustments['position_size_multiplier'] = 1.1
        
        return adjustments
    
    def _apply_threshold_bounds(self, thresholds: Dict[str, float]) -> Dict[str, float]:
        """Apply reasonable bounds to thresholds"""
        bounds = {
            'buy_threshold': (0.3, 2.0),
            'sell_threshold': (-2.0, -0.3),
            'confidence_minimum': (0.3, 1.0),
            'position_size_multiplier': (0.1, 2.0)
        }
        
        bounded_thresholds = {}
        for key, value in thresholds.items():
            if key in bounds:
                min_val, max_val = bounds[key]
                bounded_thresholds[key] = max(min_val, min(max_val, value))
            else:
                bounded_thresholds[key] = value
        
        return bounded_thresholds
    
    def update_threshold_performance(self, thresholds_used: AdaptiveThresholds, outcome: str, profit_loss: float):
        """Update performance tracking for threshold adaptation"""
        threshold_key = f"{thresholds_used.buy_threshold:.2f}_{thresholds_used.sell_threshold:.2f}_{thresholds_used.confidence_minimum:.2f}"
        
        if threshold_key not in self.threshold_performance:
            self.threshold_performance[threshold_key] = {
                'total_trades': 0,
                'profitable_trades': 0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0
            }
        
        perf = self.threshold_performance[threshold_key]
        perf['total_trades'] += 1
        perf['total_pnl'] += profit_loss
        
        if profit_loss > 0:
            perf['profitable_trades'] += 1
        
        perf['avg_pnl'] = perf['total_pnl'] / perf['total_trades']
        perf['win_rate'] = perf['profitable_trades'] / perf['total_trades']
    
    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get adaptation performance metrics"""
        if not self.adaptation_history:
            return {}
        
        recent_adaptations = self.adaptation_history[-50:]  # Last 50 adaptations
        
        regime_counts = {}
        for adaptation in recent_adaptations:
            regime = adaptation['market_context'].regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        return {
            'total_adaptations': len(self.adaptation_history),
            'recent_regime_distribution': regime_counts,
            'threshold_performance': self.threshold_performance,
            'base_thresholds': self.base_thresholds,
            'avg_adjustments': self._calculate_average_adjustments(recent_adaptations)
        }
    
    def _calculate_average_adjustments(self, adaptations: list) -> Dict[str, float]:
        """Calculate average adjustments over recent adaptations"""
        if not adaptations:
            return {}
        
        adjustment_sums = {}
        adjustment_counts = {}
        
        for adaptation in adaptations:
            thresholds = adaptation['thresholds']
            for key, value in thresholds.items():
                base_value = self.base_thresholds.get(key, 1.0)
                adjustment = value / base_value if base_value != 0 else 1.0
                
                adjustment_sums[key] = adjustment_sums.get(key, 0) + adjustment
                adjustment_counts[key] = adjustment_counts.get(key, 0) + 1
        
        return {
            key: adjustment_sums[key] / adjustment_counts[key]
            for key in adjustment_sums
        }

    def set_initial_threshold(self, threshold):
        """PRODUCTION FIX: Set initial confidence threshold"""
        try:
            self.base_thresholds['buy_threshold'] = threshold
            self.base_thresholds['sell_threshold'] = threshold
            self.base_thresholds['confidence_minimum'] = threshold
            logger.info(f"Set initial thresholds to {threshold}")
        except Exception as e:
            logger.error(f"Error setting initial threshold: {e}")
