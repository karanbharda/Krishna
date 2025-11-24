"""
Phase 3: Market Regime Detection System
Implements Bull/Bear/Sideways market adaptation with regime-specific strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL_MARKET = "bull"
    BEAR_MARKET = "bear"
    SIDEWAYS_MARKET = "sideways"
    VOLATILE_MARKET = "volatile"
    UNKNOWN = "unknown"


class MarketRegimeDetector:
    """
    Detects and adapts to different market regimes using multiple indicators
    """
    
    def __init__(self, lookback_period: int = 60):
        self.lookback_period = lookback_period
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        
        # Detection thresholds
        self.bull_threshold = 0.3
        self.bear_threshold = -0.3
        self.volatility_threshold = 0.025
        
        # Regime-specific parameters
        self.regime_parameters = {
            MarketRegime.BULL_MARKET: {
                'rsi_buy_threshold': 40,
                'rsi_sell_threshold': 75,
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 0.8
            },
            MarketRegime.BEAR_MARKET: {
                'rsi_buy_threshold': 25,
                'rsi_sell_threshold': 65,
                'position_size_multiplier': 0.6,
                'stop_loss_multiplier': 1.5
            },
            MarketRegime.SIDEWAYS_MARKET: {
                'rsi_buy_threshold': 30,
                'rsi_sell_threshold': 70,
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 1.0
            },
            MarketRegime.VOLATILE_MARKET: {
                'rsi_buy_threshold': 20,
                'rsi_sell_threshold': 80,
                'position_size_multiplier': 0.5,
                'stop_loss_multiplier': 2.0
            }
        }
        
        logger.info("✅ Market Regime Detector initialized")
    
    def detect_regime(self, market_data: pd.DataFrame) -> Dict:
        """Detect current market regime"""
        try:
            if len(market_data) < self.lookback_period:
                return self._get_unknown_regime()
            
            # Calculate indicators
            indicators = self._calculate_indicators(market_data)
            
            # Multiple detection methods
            trend_regime = self._detect_trend_regime(indicators)
            volatility_regime = self._detect_volatility_regime(indicators)
            momentum_regime = self._detect_momentum_regime(indicators)
            
            # Calculate consensus
            regime_votes = [trend_regime, volatility_regime, momentum_regime]
            consensus_regime, confidence = self._calculate_consensus(regime_votes)
            
            self.current_regime = consensus_regime
            self.regime_confidence = confidence
            
            return {
                'regime': consensus_regime.value,
                'confidence': confidence,
                'indicators': indicators,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return self._get_unknown_regime()
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate regime detection indicators"""
        try:
            close = data['Close']
            
            # Trend indicators
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            trend_slope = self._calculate_slope(close, 20)
            
            # Momentum
            momentum_20 = (close.iloc[-1] / close.iloc[-20]) - 1 if len(close) >= 20 else 0
            
            # Volatility
            returns = close.pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else 0.02
            
            # RSI
            rsi = self._calculate_rsi(close, 14)
            
            # MACD
            macd_line, signal_line, _ = self._calculate_macd(close)
            macd_histogram = macd_line - signal_line
            
            # Bollinger Bands
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(close)
            bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0
            
            # ADX for trend strength
            adx = self._calculate_adx(data, 14)
            
            return {
                'sma_20': sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else close.iloc[-1],
                'sma_50': sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else close.iloc[-1],
                'trend_slope': trend_slope,
                'momentum_20': momentum_20,
                'volatility': volatility,
                'rsi': rsi,
                'macd_histogram': macd_histogram.iloc[-1] if not pd.isna(macd_histogram.iloc[-1]) else 0,
                'bb_width': bb_width,
                'adx': adx
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def _calculate_slope(self, series: pd.Series, period: int) -> float:
        """Calculate trend slope"""
        try:
            if len(series) < period:
                return 0.0
            
            data = series.tail(period).values
            x = np.arange(len(data))
            slope, _, _, _, _ = stats.linregress(x, data)
            return slope / data[-1] if data[-1] != 0 else 0
            
        except Exception:
            return 0.0
    
    def _calculate_rsi(self, close: pd.Series, period: int) -> float:
        """Calculate RSI"""
        try:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except Exception:
            return 50.0
    
    def _calculate_macd(self, close: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple:
        """Calculate MACD"""
        try:
            ema_fast = close.ewm(span=fast_period).mean()
            ema_slow = close.ewm(span=slow_period).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception:
            return pd.Series(0), pd.Series(0), pd.Series(0)
    
    def _calculate_bollinger_bands(self, close: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple:
        """Calculate Bollinger Bands"""
        try:
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band.iloc[-1], lower_band.iloc[-1], sma.iloc[-1]
        except Exception:
            current_price = close.iloc[-1] if len(close) > 0 else 100
            return current_price * 1.02, current_price * 0.98, current_price
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX for trend strength"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # Calculate True Range
            tr1 = abs(high - low)
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            up_move = high - high.shift()
            down_move = low.shift() - low
            
            # Calculate +DM and -DM
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
            
            # Smooth values
            atr = tr.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).sum() / atr)
            minus_di = 100 * (minus_dm.rolling(period).sum() / atr)
            
            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20.0
        except Exception:
            return 20.0
    
    def _detect_trend_regime(self, indicators: Dict) -> MarketRegime:
        """Detect regime based on trend"""
        trend_slope = indicators.get('trend_slope', 0)
        adx = indicators.get('adx', 20)
        sma_20 = indicators.get('sma_20', 100)
        sma_50 = indicators.get('sma_50', 100)
        
        # Enhanced trend detection with ADX confirmation
        if trend_slope > self.bull_threshold and adx > 25:
            return MarketRegime.BULL_MARKET
        elif trend_slope < self.bear_threshold and adx > 25:
            return MarketRegime.BEAR_MARKET
        elif abs(trend_slope) < 0.1 and adx < 20:
            return MarketRegime.SIDEWAYS_MARKET
        else:
            # Use moving average relationship as fallback
            if sma_20 > sma_50 * 1.05:
                return MarketRegime.BULL_MARKET
            elif sma_20 < sma_50 * 0.95:
                return MarketRegime.BEAR_MARKET
            else:
                return MarketRegime.SIDEWAYS_MARKET
    
    def _detect_volatility_regime(self, indicators: Dict) -> MarketRegime:
        """Detect regime based on volatility"""
        volatility = indicators.get('volatility', 0.02)
        bb_width = indicators.get('bb_width', 0.1)
        momentum = indicators.get('momentum_20', 0)
        
        # Enhanced volatility detection with Bollinger Band width
        if volatility > self.volatility_threshold * 3 or bb_width > 0.1:
            return MarketRegime.VOLATILE_MARKET
        elif volatility > self.volatility_threshold * 1.5:
            return MarketRegime.BEAR_MARKET if momentum < -0.02 else MarketRegime.BULL_MARKET
        else:
            return MarketRegime.SIDEWAYS_MARKET
    
    def _detect_momentum_regime(self, indicators: Dict) -> MarketRegime:
        """Detect regime based on momentum"""
        rsi = indicators.get('rsi', 50)
        momentum = indicators.get('momentum_20', 0)
        macd_hist = indicators.get('macd_histogram', 0)
        
        # Enhanced momentum detection with MACD confirmation
        if rsi > 60 and momentum > 0.05 and macd_hist > 0:
            return MarketRegime.BULL_MARKET
        elif rsi < 40 and momentum < -0.05 and macd_hist < 0:
            return MarketRegime.BEAR_MARKET
        else:
            return MarketRegime.SIDEWAYS_MARKET
    
    def _calculate_consensus(self, votes: List[MarketRegime]) -> Tuple[MarketRegime, float]:
        """Calculate consensus from votes"""
        vote_counts = {}
        for regime in votes:
            vote_counts[regime] = vote_counts.get(regime, 0) + 1
        
        if not vote_counts:
            return MarketRegime.UNKNOWN, 0.0
        
        consensus = max(vote_counts, key=vote_counts.get)
        confidence = vote_counts[consensus] / len(votes)
        return consensus, confidence
    
    def _get_unknown_regime(self) -> Dict:
        """Return unknown regime"""
        return {
            'regime': MarketRegime.UNKNOWN.value,
            'confidence': 0.0,
            'indicators': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def get_regime_parameters(self) -> Dict:
        """Get parameters for current regime"""
        return self.regime_parameters.get(self.current_regime, {
            'rsi_buy_threshold': 30,
            'rsi_sell_threshold': 70,
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0
        })
    
    def adapt_strategy(self, base_strategy: Dict) -> Dict:
        """Adapt strategy to current regime"""
        adapted = base_strategy.copy()
        params = self.get_regime_parameters()
        
        # Apply regime adjustments
        if 'position_size' in adapted:
            adapted['position_size'] *= params.get('position_size_multiplier', 1.0)
        
        if 'stop_loss_pct' in adapted:
            adapted['stop_loss_pct'] *= params.get('stop_loss_multiplier', 1.0)
        
        adapted['rsi_buy_threshold'] = params.get('rsi_buy_threshold', 30)
        adapted['rsi_sell_threshold'] = params.get('rsi_sell_threshold', 70)
        
        return adapted
    
    def get_regime_summary(self) -> str:
        """Get regime summary"""
        regime_names = {
            MarketRegime.BULL_MARKET: "📈 Bull Market",
            MarketRegime.BEAR_MARKET: "📉 Bear Market", 
            MarketRegime.SIDEWAYS_MARKET: "➡️ Sideways Market",
            MarketRegime.VOLATILE_MARKET: "⚡ Volatile Market",
            MarketRegime.UNKNOWN: "❓ Unknown"
        }
        
        name = regime_names.get(self.current_regime, "Unknown")
        return f"{name} (Confidence: {self.regime_confidence:.1%})"


# Global instance
_regime_detector = None

def get_regime_detector() -> MarketRegimeDetector:
    """Get global regime detector instance"""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = MarketRegimeDetector()
    return _regime_detector