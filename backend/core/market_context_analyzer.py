"""
Market Context Analyzer
Provides professional market context analysis for trading decisions
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from .professional_sell_logic import MarketTrend, MarketContext

logger = logging.getLogger(__name__)

class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    trend: MarketTrend
    strength: float
    duration_days: int
    momentum: float
    confidence: float

class MarketContextAnalyzer:
    """
    Professional market context analyzer
    Provides institutional-grade market regime detection
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Trend detection parameters
        self.short_ma_period = config.get("short_ma_period", 10)
        self.long_ma_period = config.get("long_ma_period", 50)
        self.trend_strength_threshold = config.get("trend_strength_threshold", 0.02)
        
        # Volatility parameters
        self.volatility_lookback = config.get("volatility_lookback", 20)
        self.low_vol_threshold = config.get("low_vol_threshold", 0.015)
        self.high_vol_threshold = config.get("high_vol_threshold", 0.035)
        
        logger.info("Market Context Analyzer initialized")
    
    def analyze_market_context(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        sector_data: Optional[pd.DataFrame] = None,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> MarketContext:
        """
        Analyze comprehensive market context
        """
        logger.info("Analyzing market context...")
        
        # Trend analysis
        trend_analysis = self._analyze_trend(price_data)
        
        # Volatility regime
        volatility_regime = self._analyze_volatility_regime(price_data)
        
        # Market stress
        market_stress = self._calculate_market_stress(price_data, volume_data)
        
        # Sector performance
        sector_performance = self._analyze_sector_performance(price_data, sector_data, benchmark_data)
        
        # Volume profile
        volume_profile = self._analyze_volume_profile(volume_data) if volume_data is not None else 0.5
        
        context = MarketContext(
            trend=trend_analysis.trend,
            trend_strength=trend_analysis.strength,
            volatility_regime=volatility_regime.value,
            market_stress=market_stress,
            sector_performance=sector_performance,
            volume_profile=volume_profile
        )
        
        logger.info(f"Market Context: {context.trend.value} | "
                   f"Volatility: {context.volatility_regime} | "
                   f"Stress: {context.market_stress:.2f}")
        
        return context
    
    def _analyze_trend(self, price_data: pd.DataFrame) -> TrendAnalysis:
        """Analyze market trend using multiple timeframes"""
        
        if len(price_data) < self.long_ma_period:
            return TrendAnalysis(
                trend=MarketTrend.SIDEWAYS,
                strength=0.0,
                duration_days=0,
                momentum=0.0,
                confidence=0.0
            )
        
        # Calculate moving averages
        short_ma = price_data['Close'].rolling(window=self.short_ma_period).mean()
        long_ma = price_data['Close'].rolling(window=self.long_ma_period).mean()
        
        current_price = price_data['Close'].iloc[-1]
        current_short_ma = short_ma.iloc[-1]
        current_long_ma = long_ma.iloc[-1]
        
        # Trend direction
        ma_spread = (current_short_ma - current_long_ma) / current_long_ma
        price_vs_ma = (current_price - current_long_ma) / current_long_ma
        
        # Momentum calculation
        momentum = (current_price - price_data['Close'].iloc[-10]) / price_data['Close'].iloc[-10]
        
        # Trend strength
        strength = abs(ma_spread)
        
        # Classify trend
        if ma_spread > self.trend_strength_threshold and price_vs_ma > self.trend_strength_threshold:
            if strength > 0.05:
                trend = MarketTrend.STRONG_UPTREND
            else:
                trend = MarketTrend.UPTREND
        elif ma_spread < -self.trend_strength_threshold and price_vs_ma < -self.trend_strength_threshold:
            if strength > 0.05:
                trend = MarketTrend.STRONG_DOWNTREND
            else:
                trend = MarketTrend.DOWNTREND
        else:
            trend = MarketTrend.SIDEWAYS
        
        # Calculate trend duration
        duration_days = self._calculate_trend_duration(short_ma, long_ma)
        
        # Confidence based on consistency
        ma_consistency = self._calculate_ma_consistency(short_ma, long_ma)
        confidence = min(strength * 10, 1.0) * ma_consistency
        
        return TrendAnalysis(
            trend=trend,
            strength=strength,
            duration_days=duration_days,
            momentum=momentum,
            confidence=confidence
        )
    
    def _analyze_volatility_regime(self, price_data: pd.DataFrame) -> VolatilityRegime:
        """Analyze current volatility regime"""
        
        if len(price_data) < self.volatility_lookback:
            return VolatilityRegime.NORMAL
        
        # Calculate rolling volatility
        returns = price_data['Close'].pct_change().dropna()
        current_vol = returns.rolling(window=self.volatility_lookback).std().iloc[-1] * np.sqrt(252)
        
        # Classify volatility regime
        if current_vol < self.low_vol_threshold:
            return VolatilityRegime.LOW
        elif current_vol > self.high_vol_threshold * 2:
            return VolatilityRegime.EXTREME
        elif current_vol > self.high_vol_threshold:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.NORMAL
    
    def _calculate_market_stress(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame]) -> float:
        """Calculate market stress indicator"""
        
        if len(price_data) < 20:
            return 0.0
        
        # Price-based stress indicators
        returns = price_data['Close'].pct_change().dropna()
        
        # Volatility stress
        vol_stress = min(returns.rolling(20).std().iloc[-1] * np.sqrt(252) / 0.20, 1.0)
        
        # Drawdown stress
        rolling_max = price_data['Close'].rolling(window=50, min_periods=1).max()
        drawdown = (price_data['Close'] - rolling_max) / rolling_max
        drawdown_stress = min(abs(drawdown.iloc[-1]) / 0.20, 1.0)
        
        # Combine stress indicators
        stress = (vol_stress * 0.6 + drawdown_stress * 0.4)
        
        return min(stress, 1.0)
    
    def _analyze_sector_performance(
        self, 
        price_data: pd.DataFrame, 
        sector_data: Optional[pd.DataFrame],
        benchmark_data: Optional[pd.DataFrame]
    ) -> float:
        """Analyze sector performance relative to market"""
        
        if sector_data is None or benchmark_data is None or len(price_data) < 20:
            return 0.0
        
        # Calculate 20-day returns
        stock_return = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[-20]) - 1
        benchmark_return = (benchmark_data['Close'].iloc[-1] / benchmark_data['Close'].iloc[-20]) - 1
        
        # Relative performance
        relative_performance = stock_return - benchmark_return
        
        return relative_performance
    
    def _analyze_volume_profile(self, volume_data: pd.DataFrame) -> float:
        """Analyze volume profile"""
        
        if volume_data is None or len(volume_data) < 20:
            return 0.5
        
        # Current volume vs average
        current_volume = volume_data['Volume'].iloc[-1]
        avg_volume = volume_data['Volume'].rolling(20).mean().iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Normalize to 0-1 scale
        return min(volume_ratio / 2.0, 1.0)
    
    def _calculate_trend_duration(self, short_ma: pd.Series, long_ma: pd.Series) -> int:
        """Calculate how long the current trend has been in place"""
        
        # Find where short MA crossed long MA
        ma_diff = short_ma - long_ma
        ma_diff_sign = np.sign(ma_diff)
        
        # Count consecutive days with same sign
        current_sign = ma_diff_sign.iloc[-1]
        duration = 0
        
        for i in range(len(ma_diff_sign) - 1, -1, -1):
            if ma_diff_sign.iloc[i] == current_sign:
                duration += 1
            else:
                break
        
        return duration
    
    def _calculate_ma_consistency(self, short_ma: pd.Series, long_ma: pd.Series) -> float:
        """Calculate moving average consistency"""
        
        if len(short_ma) < 10:
            return 0.5
        
        # Check how consistent the MA relationship has been
        ma_diff = short_ma - long_ma
        ma_diff_sign = np.sign(ma_diff)
        
        # Calculate consistency over last 10 periods
        recent_signs = ma_diff_sign.iloc[-10:]
        consistency = abs(recent_signs.sum()) / len(recent_signs)
        
        return consistency
