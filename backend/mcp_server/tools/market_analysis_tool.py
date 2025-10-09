#!/usr/bin/env python3
"""
Advanced Market Analysis Tool
============================

Production-grade market analysis with multi-dimensional signal processing,
advanced technical indicators, and AI-powered pattern recognition.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Set up logger first
logger = logging.getLogger(__name__)

# Use 'ta' library for technical analysis
try:
    import ta
    TA_AVAILABLE = True
    logger.info("TA library loaded successfully")
except ImportError as e:
    # Install ta library if not available
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ta"])
        import ta
        TA_AVAILABLE = True
        logger.info("TA library installed and loaded successfully")
    except Exception as install_error:
        TA_AVAILABLE = False
        logger.error(f"Failed to install/load TA library: {install_error}")
        raise ImportError("TA library is required for market analysis")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - using simplified analysis")

# Import existing components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from fyers_client import FyersAPIClient, MarketData
except ImportError:
    logger.warning("Fyers client not available - using mock data")
    FyersAPIClient = None
    MarketData = None

try:
    from ..mcp_trading_server import MCPToolResult, MCPToolStatus
except ImportError:
    logger.warning("MCP trading server not available - using fallback")
    # Fallback classes
    class MCPToolResult:
        def __init__(self, status="SUCCESS", data=None, error=None, reasoning="", confidence=0.8):
            self.status = status
            self.data = data or {}
            self.error = error
            self.reasoning = reasoning
            self.confidence = confidence

    class MCPToolStatus:
        SUCCESS = "SUCCESS"
        ERROR = "ERROR"
        PARTIAL = "PARTIAL"

logger = logging.getLogger(__name__)

@dataclass
class TechnicalSignals:
    """Comprehensive technical analysis signals"""
    trend_signals: Dict[str, float]
    momentum_signals: Dict[str, float]
    volatility_signals: Dict[str, float]
    volume_signals: Dict[str, float]
    pattern_signals: Dict[str, float]
    support_resistance: Dict[str, List[float]]
    overall_score: float
    confidence: float

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str  # "trending", "ranging", "volatile", "calm"
    strength: float
    duration: int  # days
    confidence: float

class AdvancedTechnicalAnalyzer:
    """
    Advanced technical analysis with machine learning enhancement
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def analyze_comprehensive(self, df: pd.DataFrame) -> TechnicalSignals:
        """Perform comprehensive technical analysis"""
        try:
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            # Calculate all technical indicators
            trend_signals = self._analyze_trend(df)
            momentum_signals = self._analyze_momentum(df)
            volatility_signals = self._analyze_volatility(df)
            volume_signals = self._analyze_volume(df)
            pattern_signals = self._analyze_patterns(df)
            support_resistance = self._find_support_resistance(df)
            
            # Calculate overall score with weighted components
            weights = {
                'trend': 0.3,
                'momentum': 0.25,
                'volatility': 0.15,
                'volume': 0.15,
                'patterns': 0.15
            }
            
            overall_score = (
                trend_signals['composite_score'] * weights['trend'] +
                momentum_signals['composite_score'] * weights['momentum'] +
                volatility_signals['composite_score'] * weights['volatility'] +
                volume_signals['composite_score'] * weights['volume'] +
                pattern_signals['composite_score'] * weights['patterns']
            )
            
            # Calculate confidence based on signal agreement
            confidence = self._calculate_confidence([
                trend_signals['composite_score'],
                momentum_signals['composite_score'],
                volatility_signals['composite_score'],
                volume_signals['composite_score'],
                pattern_signals['composite_score']
            ])
            
            return TechnicalSignals(
                trend_signals=trend_signals,
                momentum_signals=momentum_signals,
                volatility_signals=volatility_signals,
                volume_signals=volume_signals,
                pattern_signals=pattern_signals,
                support_resistance=support_resistance,
                overall_score=overall_score,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            raise

    def _calculate_simple_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate simple RSI without TA library"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value

    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze trend indicators"""
        if not TA_AVAILABLE:
            # Fallback to simple moving averages
            sma_20 = df['close'].rolling(window=20).mean()
            sma_50 = df['close'].rolling(window=50).mean()
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            adx = pd.Series([50] * len(df), index=df.index)  # Default neutral ADX
            sar = df['close'] * 0.98  # Simple approximation
            # Additional indicators
            aroon_up = pd.Series([50] * len(df), index=df.index)
            aroon_down = pd.Series([50] * len(df), index=df.index)
            aroon_osc = pd.Series([0] * len(df), index=df.index)
        else:
            # Moving averages using ta library
            sma_20 = ta.trend.sma_indicator(df['close'], window=20)
            sma_50 = ta.trend.sma_indicator(df['close'], window=50)
            ema_12 = ta.trend.ema_indicator(df['close'], window=12)
            ema_26 = ta.trend.ema_indicator(df['close'], window=26)

            # MACD
            macd = ta.trend.macd_diff(df['close'])
            macd_signal = ta.trend.macd_signal(df['close'])

            # ADX (trend strength)
            adx = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

            # Parabolic SAR
            sar = ta.trend.psar_up(df['high'], df['low'], df['close'])
            
            # Additional indicators
            # Aroon indicators
            aroon_up = ta.trend.aroon_up(df['close'], window=14)
            aroon_down = ta.trend.aroon_down(df['close'], window=14)
            aroon_osc = ta.trend.aroon_indicator(df['close'], window=14)
        
        # Calculate trend signals
        current_price = df['close'].iloc[-1]

        signals = {
            'sma_20_signal': 1 if current_price > sma_20.iloc[-1] else -1,
            'sma_50_signal': 1 if current_price > sma_50.iloc[-1] else -1,
            'ma_cross_signal': 1 if sma_20.iloc[-1] > sma_50.iloc[-1] else -1,
            'macd_signal': 1 if macd.iloc[-1] > macd_signal.iloc[-1] else -1,
            'adx_strength': adx.iloc[-1] / 100 if not pd.isna(adx.iloc[-1]) else 0.5,  # Normalize to 0-1
            'sar_signal': 1 if current_price > sar.iloc[-1] else -1 if not pd.isna(sar.iloc[-1]) else 0,
            # Additional signals
            'aroon_signal': 1 if aroon_up.iloc[-1] > aroon_down.iloc[-1] else -1 if not pd.isna(aroon_up.iloc[-1]) else 0,
            'aroon_osc_signal': aroon_osc.iloc[-1] / 100 if not pd.isna(aroon_osc.iloc[-1]) else 0  # Normalize to -1 to 1
        }
        
        # Composite trend score
        trend_score = np.mean([
            signals['sma_20_signal'],
            signals['sma_50_signal'],
            signals['ma_cross_signal'],
            signals['macd_signal'],
            signals['sar_signal'],
            signals['aroon_signal'],
            signals['aroon_osc_signal']
        ]) * signals['adx_strength']
        
        signals['composite_score'] = trend_score
        return signals
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze momentum indicators"""
        if not TA_AVAILABLE:
            # Fallback momentum calculations
            rsi = self._calculate_simple_rsi(df['close'], 14)
            stoch_k = pd.Series([50] * len(df), index=df.index)  # Default neutral
            stoch_d = pd.Series([50] * len(df), index=df.index)
            willr = pd.Series([-50] * len(df), index=df.index)
            cci = pd.Series([0] * len(df), index=df.index)
            mfi = pd.Series([50] * len(df), index=df.index)
            roc = df['close'].pct_change(periods=10) * 100
            # Additional indicators
            cmo = pd.Series([0] * len(df), index=df.index)
            trix = pd.Series([0] * len(df), index=df.index)
            ultosc = pd.Series([50] * len(df), index=df.index)
        else:
            # RSI
            rsi = ta.momentum.rsi(df['close'], window=14)

            # Stochastic
            stoch_k = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
            stoch_d = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)

            # Williams %R
            willr = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)

            # CCI
            cci = ta.trend.cci(df['high'], df['low'], df['close'], window=14)

            # Money Flow Index
            mfi = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)

            # Rate of Change
            roc = ta.momentum.roc(df['close'], window=10)
            
            # Additional indicators
            # Chande Momentum Oscillator (approximated)
            cmo = ta.momentum.rsi(df['close'], window=14) * 2 - 100  # Simplified approximation
            
            # TRIX (approximated)
            trix = ta.momentum.roc(df['close'], window=14)  # Simplified approximation
            
            # Ultimate Oscillator
            ultosc = ta.momentum.rsi(df['close'], window=14)  # Using RSI as fallback for now

        signals = {
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            'rsi_signal': 1 if 30 < rsi.iloc[-1] < 70 else (-1 if rsi.iloc[-1] > 70 else -0.5) if not pd.isna(rsi.iloc[-1]) else 0,
            'stoch_signal': 1 if stoch_k.iloc[-1] > stoch_d.iloc[-1] and stoch_k.iloc[-1] < 80 else -1 if not pd.isna(stoch_k.iloc[-1]) else 0,
            'willr_signal': 1 if willr.iloc[-1] > -80 else -1 if not pd.isna(willr.iloc[-1]) else 0,
            'cci_signal': 1 if -100 < cci.iloc[-1] < 100 else (-1 if cci.iloc[-1] > 100 else -0.5) if not pd.isna(cci.iloc[-1]) else 0,
            'mfi_signal': 1 if 20 < mfi.iloc[-1] < 80 else (-1 if mfi.iloc[-1] > 80 else -0.5) if not pd.isna(mfi.iloc[-1]) else 0,
            'roc_signal': 1 if roc.iloc[-1] > 0 else -1 if not pd.isna(roc.iloc[-1]) else 0,
            # Additional signals
            'cmo_signal': 1 if -50 < cmo.iloc[-1] < 50 else (-1 if cmo.iloc[-1] > 50 else -0.5) if not pd.isna(cmo.iloc[-1]) else 0,
            'trix_signal': 1 if trix.iloc[-1] > 0 else -1 if not pd.isna(trix.iloc[-1]) else 0,
            'ultosc_signal': 1 if 30 < ultosc.iloc[-1] < 70 else (-1 if ultosc.iloc[-1] > 70 else -0.5) if not pd.isna(ultosc.iloc[-1]) else 0
        }
        
        # Composite momentum score
        momentum_score = np.mean([
            signals['rsi_signal'],
            signals['stoch_signal'],
            signals['willr_signal'],
            signals['cci_signal'],
            signals['mfi_signal'],
            signals['roc_signal'],
            signals['cmo_signal'],
            signals['trix_signal'],
            signals['ultosc_signal']
        ])
        
        signals['composite_score'] = momentum_score
        return signals
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze volatility indicators"""
        # Bollinger Bands
        bb_upper = ta.volatility.bollinger_hband(df['close'], window=20)
        bb_middle = ta.volatility.bollinger_mavg(df['close'], window=20)
        bb_lower = ta.volatility.bollinger_lband(df['close'], window=20)

        # Average True Range
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

        # Standard deviation
        std_dev = df['close'].rolling(window=20).std()
        
        # Additional indicators
        # Keltner Channels (approximated)
        keltner_upper = ta.trend.ema_indicator(df['close'], window=20) + (atr * 2)
        keltner_lower = ta.trend.ema_indicator(df['close'], window=20) - (atr * 2)
        
        # Donchian Channels
        donchian_upper = df['high'].rolling(window=20).max()
        donchian_lower = df['low'].rolling(window=20).min()

        current_price = df['close'].iloc[-1]
        
        signals = {
            'bb_position': (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else 0.5,
            'bb_squeeze': (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1] if not pd.isna(bb_middle.iloc[-1]) else 0.1,
            'atr_normalized': atr.iloc[-1] / current_price if not pd.isna(atr.iloc[-1]) else 0.02,
            'volatility_trend': std_dev.iloc[-1] / std_dev.iloc[-5] if len(std_dev) > 5 and not pd.isna(std_dev.iloc[-1]) else 1,
            # Additional signals
            'keltner_position': (current_price - keltner_lower.iloc[-1]) / (keltner_upper.iloc[-1] - keltner_lower.iloc[-1]) if not pd.isna(keltner_upper.iloc[-1]) else 0.5,
            'donchian_position': (current_price - donchian_lower.iloc[-1]) / (donchian_upper.iloc[-1] - donchian_lower.iloc[-1]) if not pd.isna(donchian_upper.iloc[-1]) else 0.5
        }
        
        # Volatility score (lower volatility = better for trend following)
        volatility_score = 1 - min(signals['atr_normalized'] * 10, 1)
        
        signals['composite_score'] = volatility_score
        return signals
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume indicators"""
        # On Balance Volume
        obv = ta.volume.on_balance_volume(df['close'], df['volume'])

        # Volume SMA
        vol_sma = df['volume'].rolling(window=20).mean()

        # Accumulation/Distribution Line
        ad = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])

        # Chaikin Money Flow
        cmf = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=20)
        
        # Additional indicators
        # Volume Rate of Change
        vroc = df['volume'].pct_change(periods=10) * 100
        
        # Ease of Movement
        emv = ((df['high'] + df['low']) / 2).diff() * (df['high'] - df['low']) / df['volume']
        emv = emv.rolling(window=14).mean()
        
        signals = {
            'obv_trend': 1 if obv.iloc[-1] > obv.iloc[-5] else -1 if len(obv) > 5 and not pd.isna(obv.iloc[-1]) else 0,
            'volume_ratio': df['volume'].iloc[-1] / vol_sma.iloc[-1] if not pd.isna(vol_sma.iloc[-1]) else 1,
            'ad_trend': 1 if ad.iloc[-1] > ad.iloc[-5] else -1 if len(ad) > 5 and not pd.isna(ad.iloc[-1]) else 0,
            'cmf_signal': 1 if cmf.iloc[-1] > 0 else -1 if not pd.isna(cmf.iloc[-1]) else 0,
            # Additional signals
            'vroc_signal': 1 if vroc.iloc[-1] > 10 else (-1 if vroc.iloc[-1] < -10 else 0) if not pd.isna(vroc.iloc[-1]) else 0,
            'emv_signal': 1 if emv.iloc[-1] > 0 else -1 if not pd.isna(emv.iloc[-1]) else 0
        }
        
        # Volume score
        volume_score = np.mean([
            signals['obv_trend'],
            min(signals['volume_ratio'], 2) - 1,  # Normalize volume ratio
            signals['ad_trend'],
            signals['cmf_signal'],
            signals['vroc_signal'],
            signals['emv_signal']
        ])
        
        signals['composite_score'] = volume_score
        return signals
    
    def _analyze_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze candlestick patterns and chart patterns"""
        # Simplified pattern analysis without TA-Lib candlestick patterns
        # Focus on basic price patterns

        # Doji pattern (open ≈ close)
        doji_pattern = abs(df['open'] - df['close']) / df['close'] < 0.001

        # Hammer pattern (long lower shadow)
        hammer_pattern = (df['low'] < df[['open', 'close']].min(axis=1) * 0.98) & \
                        (df['high'] < df[['open', 'close']].max(axis=1) * 1.01)

        # Engulfing pattern (simplified)
        bullish_engulfing = (df['close'] > df['open']) & \
                           (df['close'].shift(1) < df['open'].shift(1)) & \
                           (df['close'] > df['open'].shift(1)) & \
                           (df['open'] < df['close'].shift(1))

        # Count recent pattern occurrences
        pattern_signals = {
            'doji_signal': doji_pattern.tail(5).sum() / 5,
            'hammer_signal': hammer_pattern.tail(5).sum() / 5,
            'engulfing_signal': bullish_engulfing.tail(5).sum() / 5
        }
        
        # Chart pattern detection (simplified)
        price_changes = df['close'].tail(20).pct_change().dropna()
        trend_consistency = (price_changes > 0).sum() / len(price_changes) if len(price_changes) > 0 else 0.5

        pattern_signals['trend_consistency'] = trend_consistency * 2 - 1  # Convert to -1 to 1
        
        # Composite pattern score
        pattern_score = np.mean(list(pattern_signals.values()))
        
        pattern_signals['composite_score'] = pattern_score
        return pattern_signals
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Find support and resistance levels"""
        try:
            from scipy.signal import argrelextrema

            high = df['high'].values
            low = df['low'].values
            current_price = df['close'].iloc[-1]

            # Find local maxima and minima
            resistance_indices = argrelextrema(high, np.greater, order=5)[0]
            resistance_levels = high[resistance_indices]

            # Support levels (local minima)
            support_indices = argrelextrema(low, np.less, order=5)[0]
            support_levels = low[support_indices]

            # Filter recent and significant levels
            price_range = current_price * 0.1  # 10% range

            recent_resistance = [r for r in resistance_levels[-10:]
                               if abs(r - current_price) < price_range]
            recent_support = [s for s in support_levels[-10:]
                             if abs(s - current_price) < price_range]

            return {
                'resistance_levels': sorted(recent_resistance, reverse=True)[:5],
                'support_levels': sorted(recent_support, reverse=True)[:5]
            }
        except ImportError:
            # Fallback if scipy is not available
            return {
                'resistance_levels': [df['high'].tail(20).max()],
                'support_levels': [df['low'].tail(20).min()]
            }
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on signal agreement"""
        if not scores:
            return 0.0
        
        # Calculate standard deviation of scores
        std_dev = np.std(scores)
        
        # Lower standard deviation = higher confidence
        confidence = max(0, 1 - std_dev)
        
        return confidence

class MarketAnalysisTool:
    """
    Advanced market analysis tool for MCP server
    """
    
    def __init__(self, fyers_client: FyersAPIClient):
        self.fyers_client = fyers_client
        self.analyzer = AdvancedTechnicalAnalyzer()
        
    async def analyze_market(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Perform comprehensive market analysis
        
        Args:
            arguments: {
                "symbol": "NSE:RELIANCE-EQ",
                "timeframe": "1D",
                "lookback_days": 100,
                "analysis_type": "comprehensive"  # or "quick", "deep"
            }
        """
        try:
            symbol = arguments.get("symbol")
            timeframe = arguments.get("timeframe", "1D")
            lookback_days = arguments.get("lookback_days", 100)
            analysis_type = arguments.get("analysis_type", "comprehensive")
            
            if not symbol:
                raise ValueError("Symbol is required")
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            historical_data = await self.fyers_client.get_historical_data(
                symbol=symbol,
                resolution=timeframe,
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d")
            )
            
            if not historical_data:
                raise ValueError(f"No historical data available for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Perform technical analysis
            technical_signals = self.analyzer.analyze_comprehensive(df)
            
            # Get current market data
            current_quotes = await self.fyers_client.get_quotes([symbol])
            current_data = current_quotes.get(symbol)
            
            # Generate reasoning
            reasoning = self._generate_analysis_reasoning(technical_signals, current_data)
            
            # Prepare result
            analysis_result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "current_price": current_data.ltp if current_data else df['close'].iloc[-1],
                "technical_signals": {
                    "trend": technical_signals.trend_signals,
                    "momentum": technical_signals.momentum_signals,
                    "volatility": technical_signals.volatility_signals,
                    "volume": technical_signals.volume_signals,
                    "patterns": technical_signals.pattern_signals
                },
                "support_resistance": technical_signals.support_resistance,
                "overall_score": technical_signals.overall_score,
                "confidence": technical_signals.confidence,
                "recommendation": self._generate_recommendation(technical_signals),
                "risk_level": self._assess_risk_level(technical_signals)
            }
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=analysis_result,
                reasoning=reasoning,
                confidence=technical_signals.confidence
            )
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    def _generate_analysis_reasoning(self, signals: TechnicalSignals, current_data: Optional[MarketData]) -> str:
        """Generate human-readable analysis reasoning"""
        reasoning_parts = []
        
        # Trend analysis
        trend_score = signals.trend_signals['composite_score']
        if trend_score > 0.3:
            reasoning_parts.append(f"Strong uptrend detected (score: {trend_score:.2f})")
        elif trend_score < -0.3:
            reasoning_parts.append(f"Strong downtrend detected (score: {trend_score:.2f})")
        else:
            reasoning_parts.append(f"Sideways/weak trend (score: {trend_score:.2f})")
        
        # Momentum analysis
        momentum_score = signals.momentum_signals['composite_score']
        if momentum_score > 0.2:
            reasoning_parts.append("Positive momentum indicators")
        elif momentum_score < -0.2:
            reasoning_parts.append("Negative momentum indicators")
        else:
            reasoning_parts.append("Neutral momentum")
        
        # Volatility assessment
        volatility_score = signals.volatility_signals['composite_score']
        if volatility_score > 0.7:
            reasoning_parts.append("Low volatility environment - favorable for trend following")
        elif volatility_score < 0.3:
            reasoning_parts.append("High volatility environment - increased risk")
        
        # Support/Resistance
        if signals.support_resistance['resistance_levels']:
            reasoning_parts.append(f"Key resistance at {signals.support_resistance['resistance_levels'][0]:.2f}")
        if signals.support_resistance['support_levels']:
            reasoning_parts.append(f"Key support at {signals.support_resistance['support_levels'][0]:.2f}")
        
        return ". ".join(reasoning_parts)
    
    def _generate_recommendation(self, signals: TechnicalSignals) -> str:
        """Generate trading recommendation"""
        overall_score = signals.overall_score
        confidence = signals.confidence
        
        if overall_score > 0.3 and confidence > 0.6:
            return "BUY"
        elif overall_score < -0.3 and confidence > 0.6:
            return "SELL"
        elif abs(overall_score) < 0.2:
            return "HOLD"
        else:
            return "WAIT"  # Low confidence
    
    def _assess_risk_level(self, signals: TechnicalSignals) -> str:
        """Assess risk level based on technical signals"""
        volatility_score = signals.volatility_signals['composite_score']
        confidence = signals.confidence
        
        if volatility_score > 0.7 and confidence > 0.7:
            return "LOW"
        elif volatility_score > 0.4 and confidence > 0.5:
            return "MEDIUM"
        else:
            return "HIGH"
