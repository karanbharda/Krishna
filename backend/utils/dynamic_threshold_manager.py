"""
Phase 1: Dynamic Threshold Manager
Replaces hardcoded indicator thresholds with adaptive ones based on market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DynamicThresholdManager:
    """
    Manages adaptive thresholds for technical indicators based on market volatility
    and historical performance
    """
    
    def __init__(self):
        self.default_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_bullish': 0,
            'macd_bearish': 0,
            'bb_oversold': 0.2,  # Distance from lower band
            'bb_overbought': 0.8,  # Distance from upper band
            'volume_threshold': 1.5,  # Volume multiplier
            'volatility_threshold': 0.02  # Daily volatility threshold
        }
        
        self.adaptive_thresholds = self.default_thresholds.copy()
        self.market_regime = 'normal'  # 'volatile', 'trending', 'sideways', 'normal'
        self.volatility_window = 20
        self.performance_history = []
        
        logger.info("✅ Dynamic Threshold Manager initialized")
    
    def calculate_market_volatility(self, price_data: pd.DataFrame) -> float:
        """Calculate current market volatility using price data"""
        try:
            if len(price_data) < self.volatility_window:
                return 0.02  # Default volatility
                
            returns = price_data['Close'].pct_change().dropna()
            volatility = returns.rolling(window=self.volatility_window).std().iloc[-1]
            
            return volatility if not pd.isna(volatility) else 0.02
            
        except Exception as e:
            logger.warning(f"Error calculating volatility: {e}")
            return 0.02
    
    def detect_market_regime(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> str:
        """Detect current market regime"""
        try:
            if len(price_data) < 50:
                return 'normal'
            
            # Calculate volatility
            volatility = self.calculate_market_volatility(price_data)
            
            # Calculate trend strength
            close_prices = price_data['Close'].iloc[-20:]
            trend_slope = np.polyfit(range(len(close_prices)), close_prices, 1)[0]
            price_range = close_prices.max() - close_prices.min()
            trend_strength = abs(trend_slope) / (price_range / len(close_prices))
            
            # Classify regime
            if volatility > 0.04:  # High volatility
                self.market_regime = 'volatile'
            elif trend_strength > 0.3:  # Strong trend
                self.market_regime = 'trending'
            elif volatility < 0.01 and trend_strength < 0.1:  # Low volatility, weak trend
                self.market_regime = 'sideways'
            else:
                self.market_regime = 'normal'
                
            logger.debug(f"Market regime detected: {self.market_regime} (vol: {volatility:.4f}, trend: {trend_strength:.4f})")
            return self.market_regime
            
        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}")
            return 'normal'
    
    def update_adaptive_thresholds(self, price_data: pd.DataFrame, performance_feedback: Optional[Dict] = None):
        """Update thresholds based on market conditions and performance feedback"""
        try:
            regime = self.detect_market_regime(price_data)
            volatility = self.calculate_market_volatility(price_data)
            
            # Base adjustments on market regime
            if regime == 'volatile':
                # In volatile markets, use wider thresholds to avoid noise
                self.adaptive_thresholds['rsi_oversold'] = max(20, 30 - volatility * 500)
                self.adaptive_thresholds['rsi_overbought'] = min(80, 70 + volatility * 500)
                self.adaptive_thresholds['volume_threshold'] = 2.0
                
            elif regime == 'trending':
                # In trending markets, use tighter thresholds to catch momentum
                self.adaptive_thresholds['rsi_oversold'] = 35
                self.adaptive_thresholds['rsi_overbought'] = 65
                self.adaptive_thresholds['volume_threshold'] = 1.2
                
            elif regime == 'sideways':
                # In sideways markets, use standard thresholds
                self.adaptive_thresholds['rsi_oversold'] = 30
                self.adaptive_thresholds['rsi_overbought'] = 70
                self.adaptive_thresholds['volume_threshold'] = 1.5
                
            else:  # normal
                # Use slightly adjusted thresholds based on volatility
                vol_adjustment = min(10, volatility * 300)
                self.adaptive_thresholds['rsi_oversold'] = 30 - vol_adjustment
                self.adaptive_thresholds['rsi_overbought'] = 70 + vol_adjustment
                self.adaptive_thresholds['volume_threshold'] = 1.5
            
            # Incorporate performance feedback
            if performance_feedback:
                self._adjust_based_on_performance(performance_feedback)
            
            logger.debug(f"Updated thresholds for {regime} market: RSI({self.adaptive_thresholds['rsi_oversold']:.1f}, {self.adaptive_thresholds['rsi_overbought']:.1f})")
            
        except Exception as e:
            logger.error(f"Error updating adaptive thresholds: {e}")
            # Fallback to default thresholds
            self.adaptive_thresholds = self.default_thresholds.copy()
    
    def _adjust_based_on_performance(self, performance_feedback: Dict):
        """Adjust thresholds based on recent trading performance"""
        try:
            success_rate = performance_feedback.get('success_rate', 0.5)
            avg_return = performance_feedback.get('avg_return', 0.0)
            
            # If performance is poor, make thresholds more conservative
            if success_rate < 0.4:
                # Make RSI thresholds more extreme (more conservative)
                self.adaptive_thresholds['rsi_oversold'] *= 0.9
                self.adaptive_thresholds['rsi_overbought'] *= 1.1
                self.adaptive_thresholds['volume_threshold'] *= 1.2
                
            elif success_rate > 0.7:
                # Make RSI thresholds less extreme (more aggressive)
                self.adaptive_thresholds['rsi_oversold'] *= 1.05
                self.adaptive_thresholds['rsi_overbought'] *= 0.95
                self.adaptive_thresholds['volume_threshold'] *= 0.9
            
            # Ensure thresholds stay within reasonable bounds
            self.adaptive_thresholds['rsi_oversold'] = np.clip(self.adaptive_thresholds['rsi_oversold'], 15, 40)
            self.adaptive_thresholds['rsi_overbought'] = np.clip(self.adaptive_thresholds['rsi_overbought'], 60, 85)
            self.adaptive_thresholds['volume_threshold'] = np.clip(self.adaptive_thresholds['volume_threshold'], 1.0, 3.0)
            
        except Exception as e:
            logger.warning(f"Error adjusting thresholds based on performance: {e}")
    
    def get_rsi_thresholds(self) -> Tuple[float, float]:
        """Get current RSI oversold and overbought thresholds"""
        return (
            self.adaptive_thresholds['rsi_oversold'],
            self.adaptive_thresholds['rsi_overbought']
        )
    
    def get_volume_threshold(self) -> float:
        """Get current volume threshold multiplier"""
        return self.adaptive_thresholds['volume_threshold']
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all current adaptive thresholds"""
        return self.adaptive_thresholds.copy()
    
    def get_threshold_explanation(self) -> str:
        """Get explanation of current thresholds and market regime"""
        oversold, overbought = self.get_rsi_thresholds()
        
        explanation = f"""
Dynamic Threshold Status:
- Market Regime: {self.market_regime.upper()}
- RSI Oversold: {oversold:.1f} (vs default 30.0)
- RSI Overbought: {overbought:.1f} (vs default 70.0)
- Volume Threshold: {self.get_volume_threshold():.1f}x
- Adaptation: {'Active' if self.market_regime != 'normal' else 'Minimal'}
        """.strip()
        
        return explanation
    
    def reset_to_defaults(self):
        """Reset all thresholds to default values"""
        self.adaptive_thresholds = self.default_thresholds.copy()
        self.market_regime = 'normal'
        logger.info("🔄 Thresholds reset to defaults")


# Global instance for easy access
_threshold_manager = None

def get_threshold_manager() -> DynamicThresholdManager:
    """Get the global threshold manager instance"""
    global _threshold_manager
    if _threshold_manager is None:
        _threshold_manager = DynamicThresholdManager()
    return _threshold_manager