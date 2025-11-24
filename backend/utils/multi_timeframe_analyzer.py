"""
Phase 2: Multi-Timeframe Confirmation System
Implements 1m, 5m, 15m, 1h, 1D analysis for robust signal confirmation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Supported timeframes for analysis"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    D1 = "1d"


class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1


class MultiTimeframeAnalyzer:
    """
    Analyzes multiple timeframes to provide robust signal confirmation
    """
    
    def __init__(self):
        self.timeframes = [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.D1]
        self.weights = {
            Timeframe.M1: 0.05,   # Noise-prone, lowest weight
            Timeframe.M5: 0.15,   # Short-term signals
            Timeframe.M15: 0.25,  # Medium-term signals
            Timeframe.H1: 0.35,   # Strong intermediate signals
            Timeframe.D1: 0.20    # Long-term trend confirmation
        }
        
        self.signal_cache = {}
        self.trend_cache = {}
        
        logger.info("✅ Multi-Timeframe Analyzer initialized")
    
    def analyze_all_timeframes(self, symbol: str, data_provider) -> Dict:
        """
        Analyze signal across all timeframes and provide weighted consensus
        """
        try:
            results = {}
            weighted_signals = []
            
            for timeframe in self.timeframes:
                try:
                    # Get data for this timeframe
                    data = self._get_timeframe_data(symbol, timeframe, data_provider)
                    
                    if data is None or len(data) < 50:
                        logger.warning(f"Insufficient data for {timeframe.value} analysis")
                        continue
                    
                    # Analyze this timeframe
                    analysis = self._analyze_single_timeframe(data, timeframe)
                    results[timeframe.value] = analysis
                    
                    # Calculate weighted signal
                    signal_score = analysis['signal_score']
                    weight = self.weights[timeframe]
                    weighted_signals.append(signal_score * weight)
                    
                    logger.debug(f"{timeframe.value}: Signal={signal_score:.3f}, Weight={weight}")
                    
                except Exception as e:
                    logger.warning(f"Error analyzing {timeframe.value}: {e}")
                    continue
            
            # Calculate consensus
            if weighted_signals:
                consensus_score = sum(weighted_signals)
                consensus_signal = self._interpret_consensus_score(consensus_score)
                
                results['consensus'] = {
                    'score': consensus_score,
                    'signal': consensus_signal,
                    'strength': self._calculate_signal_strength(results),
                    'confirmation_level': self._calculate_confirmation_level(results)
                }
            else:
                results['consensus'] = {
                    'score': 0.0,
                    'signal': 'HOLD',
                    'strength': SignalStrength.VERY_WEAK,
                    'confirmation_level': 0.0
                }
            
            logger.info(f"Multi-timeframe analysis completed: {results['consensus']['signal']} (strength: {results['consensus']['strength'].name})")
            return results
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return {'consensus': {'score': 0.0, 'signal': 'HOLD', 'strength': SignalStrength.VERY_WEAK}}
    
    def _get_timeframe_data(self, symbol: str, timeframe: Timeframe, data_provider) -> Optional[pd.DataFrame]:
        """
        Get data for specific timeframe
        """
        try:
            # Calculate periods needed based on timeframe
            periods_map = {
                Timeframe.M1: 1440,   # 1 day of 1m data
                Timeframe.M5: 288,    # 1 day of 5m data  
                Timeframe.M15: 96,    # 1 day of 15m data
                Timeframe.H1: 168,    # 1 week of 1h data
                Timeframe.D1: 252     # 1 year of daily data
            }
            
            periods = periods_map.get(timeframe, 100)
            
            # Try to get data from provider
            if hasattr(data_provider, 'get_historical_data'):
                data = data_provider.get_historical_data(symbol, timeframe.value, periods)
            else:
                # Fallback: resample daily data
                daily_data = data_provider.get_data(symbol)
                if daily_data is not None:
                    data = self._resample_data(daily_data, timeframe)
                else:
                    return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting {timeframe.value} data for {symbol}: {e}")
            return None
    
    def _resample_data(self, daily_data: pd.DataFrame, target_timeframe: Timeframe) -> pd.DataFrame:
        """
        Resample daily data to target timeframe (limited functionality)
        """
        try:
            if target_timeframe == Timeframe.D1:
                return daily_data
            
            # For intraday timeframes, we can't accurately resample from daily data
            # Return the daily data as approximation
            logger.warning(f"Cannot accurately resample daily data to {target_timeframe.value}")
            return daily_data
            
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return daily_data
    
    def _analyze_single_timeframe(self, data: pd.DataFrame, timeframe: Timeframe) -> Dict:
        """
        Analyze signals for a single timeframe
        """
        try:
            # Calculate indicators for this timeframe
            indicators = self._calculate_timeframe_indicators(data)
            
            # Analyze trend
            trend_analysis = self._analyze_trend(data, indicators)
            
            # Analyze momentum
            momentum_analysis = self._analyze_momentum(indicators)
            
            # Analyze volume
            volume_analysis = self._analyze_volume(data)
            
            # Calculate support/resistance
            sr_analysis = self._analyze_support_resistance(data)
            
            # Combine signals
            signals = [
                trend_analysis['signal_score'],
                momentum_analysis['signal_score'], 
                volume_analysis['signal_score'],
                sr_analysis['signal_score']
            ]
            
            signal_score = np.mean(signals)
            
            return {
                'timeframe': timeframe.value,
                'signal_score': signal_score,
                'trend': trend_analysis,
                'momentum': momentum_analysis,
                'volume': volume_analysis,
                'support_resistance': sr_analysis,
                'indicators': indicators,
                'recommendation': self._get_recommendation(signal_score)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {timeframe.value}: {e}")
            return {
                'timeframe': timeframe.value,
                'signal_score': 0.0,
                'recommendation': 'HOLD'
            }
    
    def _calculate_timeframe_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Calculate technical indicators for timeframe analysis
        """
        try:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data.get('Volume', pd.Series([1] * len(data)))
            
            indicators = {}
            
            # Moving averages
            indicators['sma_10'] = close.rolling(10).mean()
            indicators['sma_20'] = close.rolling(20).mean()
            indicators['sma_50'] = close.rolling(50).mean()
            indicators['ema_12'] = close.ewm(span=12).mean()
            indicators['ema_26'] = close.ewm(span=26).mean()
            
            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            indicators['bb_upper'] = bb_middle + (bb_std * 2)
            indicators['bb_lower'] = bb_middle - (bb_std * 2)
            indicators['bb_position'] = (close - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            
            # Volume indicators
            indicators['volume_sma'] = volume.rolling(20).mean()
            indicators['volume_ratio'] = volume / indicators['volume_sma']
            
            # Average True Range
            high_low = high - low
            high_close = (high - close.shift()).abs()
            low_close = (low - close.shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr'] = true_range.rolling(14).mean()
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def _analyze_trend(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """
        Analyze trend strength and direction
        """
        try:
            close = data['Close'].iloc[-1]
            
            # Moving average analysis
            ma_signals = []
            if 'sma_10' in indicators and not pd.isna(indicators['sma_10'].iloc[-1]):
                ma_signals.append(1 if close > indicators['sma_10'].iloc[-1] else -1)
            if 'sma_20' in indicators and not pd.isna(indicators['sma_20'].iloc[-1]):
                ma_signals.append(1 if close > indicators['sma_20'].iloc[-1] else -1)
            if 'sma_50' in indicators and not pd.isna(indicators['sma_50'].iloc[-1]):
                ma_signals.append(1 if close > indicators['sma_50'].iloc[-1] else -1)
            
            # MACD trend
            macd_signal = 0
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'].iloc[-1]
                macd_sig = indicators['macd_signal'].iloc[-1]
                if not (pd.isna(macd) or pd.isna(macd_sig)):
                    macd_signal = 1 if macd > macd_sig else -1
            
            # Combine trend signals
            all_signals = ma_signals + [macd_signal]
            trend_score = np.mean(all_signals) if all_signals else 0
            
            return {
                'signal_score': trend_score,
                'direction': 'BULLISH' if trend_score > 0.2 else 'BEARISH' if trend_score < -0.2 else 'NEUTRAL',
                'strength': abs(trend_score),
                'ma_alignment': np.mean(ma_signals) if ma_signals else 0,
                'macd_signal': macd_signal
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {'signal_score': 0, 'direction': 'NEUTRAL'}
    
    def _analyze_momentum(self, indicators: Dict) -> Dict:
        """
        Analyze momentum indicators
        """
        try:
            momentum_signals = []
            
            # RSI analysis
            if 'rsi' in indicators and not pd.isna(indicators['rsi'].iloc[-1]):
                rsi = indicators['rsi'].iloc[-1]
                if rsi < 30:
                    momentum_signals.append(1)  # Oversold - bullish
                elif rsi > 70:
                    momentum_signals.append(-1)  # Overbought - bearish
                else:
                    momentum_signals.append(0)  # Neutral
            
            # MACD histogram
            if 'macd_histogram' in indicators and not pd.isna(indicators['macd_histogram'].iloc[-1]):
                macd_hist = indicators['macd_histogram'].iloc[-1]
                momentum_signals.append(1 if macd_hist > 0 else -1)
            
            momentum_score = np.mean(momentum_signals) if momentum_signals else 0
            
            return {
                'signal_score': momentum_score,
                'rsi_level': indicators.get('rsi', pd.Series([50])).iloc[-1],
                'momentum_strength': abs(momentum_score)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return {'signal_score': 0}
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """
        Analyze volume patterns
        """
        try:
            if 'Volume' not in data.columns:
                return {'signal_score': 0, 'volume_trend': 'NEUTRAL'}
            
            volume = data['Volume']
            volume_sma = volume.rolling(20).mean()
            
            current_volume = volume.iloc[-1]
            avg_volume = volume_sma.iloc[-1]
            
            if pd.isna(avg_volume) or avg_volume == 0:
                return {'signal_score': 0, 'volume_trend': 'NEUTRAL'}
            
            volume_ratio = current_volume / avg_volume
            
            # Volume signal interpretation
            if volume_ratio > 1.5:
                volume_signal = 0.5  # High volume can support moves
            elif volume_ratio < 0.5:
                volume_signal = -0.2  # Low volume reduces conviction
            else:
                volume_signal = 0
            
            return {
                'signal_score': volume_signal,
                'volume_ratio': volume_ratio,
                'volume_trend': 'HIGH' if volume_ratio > 1.5 else 'LOW' if volume_ratio < 0.5 else 'NORMAL'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return {'signal_score': 0, 'volume_trend': 'NEUTRAL'}
    
    def _analyze_support_resistance(self, data: pd.DataFrame) -> Dict:
        """
        Analyze support and resistance levels
        """
        try:
            high = data['High']
            low = data['Low']
            close = data['Close'].iloc[-1]
            
            # Calculate recent highs and lows
            resistance = high.rolling(20).max().iloc[-1]
            support = low.rolling(20).min().iloc[-1]
            
            # Distance from support/resistance
            if resistance != support:
                position = (close - support) / (resistance - support)
            else:
                position = 0.5
            
            # Signal based on position
            if position < 0.2:
                sr_signal = 0.3  # Near support - potential bounce
            elif position > 0.8:
                sr_signal = -0.3  # Near resistance - potential rejection
            else:
                sr_signal = 0  # In the middle
            
            return {
                'signal_score': sr_signal,
                'support_level': support,
                'resistance_level': resistance,
                'position_in_range': position
            }
            
        except Exception as e:
            logger.error(f"Error analyzing support/resistance: {e}")
            return {'signal_score': 0}
    
    def _get_recommendation(self, signal_score: float) -> str:
        """
        Convert signal score to recommendation
        """
        if signal_score > 0.3:
            return 'STRONG_BUY'
        elif signal_score > 0.1:
            return 'BUY'
        elif signal_score > -0.1:
            return 'HOLD'
        elif signal_score > -0.3:
            return 'SELL'
        else:
            return 'STRONG_SELL'
    
    def _interpret_consensus_score(self, score: float) -> str:
        """
        Interpret consensus score into trading signal
        """
        if score > 0.4:
            return 'STRONG_BUY'
        elif score > 0.2:
            return 'BUY'
        elif score > -0.2:
            return 'HOLD'
        elif score > -0.4:
            return 'SELL'
        else:
            return 'STRONG_SELL'
    
    def _calculate_signal_strength(self, results: Dict) -> SignalStrength:
        """
        Calculate overall signal strength based on timeframe agreement
        """
        try:
            # Count agreeing timeframes
            consensus_signal = results.get('consensus', {}).get('signal', 'HOLD')
            agreeing_timeframes = 0
            total_timeframes = 0
            
            for tf_key, tf_data in results.items():
                if tf_key == 'consensus':
                    continue
                    
                total_timeframes += 1
                tf_signal = tf_data.get('recommendation', 'HOLD')
                
                # Check if signals align
                if self._signals_align(consensus_signal, tf_signal):
                    agreeing_timeframes += 1
            
            if total_timeframes == 0:
                return SignalStrength.VERY_WEAK
            
            agreement_ratio = agreeing_timeframes / total_timeframes
            
            if agreement_ratio >= 0.8:
                return SignalStrength.VERY_STRONG
            elif agreement_ratio >= 0.6:
                return SignalStrength.STRONG
            elif agreement_ratio >= 0.4:
                return SignalStrength.MODERATE
            elif agreement_ratio >= 0.2:
                return SignalStrength.WEAK
            else:
                return SignalStrength.VERY_WEAK
                
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return SignalStrength.VERY_WEAK
    
    def _signals_align(self, signal1: str, signal2: str) -> bool:
        """
        Check if two signals are aligned
        """
        buy_signals = ['BUY', 'STRONG_BUY']
        sell_signals = ['SELL', 'STRONG_SELL']
        hold_signals = ['HOLD']
        
        if signal1 in buy_signals and signal2 in buy_signals:
            return True
        elif signal1 in sell_signals and signal2 in sell_signals:
            return True
        elif signal1 in hold_signals and signal2 in hold_signals:
            return True
        else:
            return False
    
    def _calculate_confirmation_level(self, results: Dict) -> float:
        """
        Calculate confirmation level (0.0 to 1.0)
        """
        try:
            signal_scores = []
            
            for tf_key, tf_data in results.items():
                if tf_key == 'consensus':
                    continue
                    
                score = tf_data.get('signal_score', 0)
                signal_scores.append(abs(score))
            
            if not signal_scores:
                return 0.0
            
            # Average absolute signal strength
            avg_strength = np.mean(signal_scores)
            return min(1.0, avg_strength)
            
        except Exception as e:
            logger.error(f"Error calculating confirmation level: {e}")
            return 0.0
    
    def get_timeframe_summary(self, results: Dict) -> str:
        """
        Get human-readable summary of multi-timeframe analysis
        """
        try:
            consensus = results.get('consensus', {})
            signal = consensus.get('signal', 'HOLD')
            strength = consensus.get('strength', SignalStrength.VERY_WEAK)
            confirmation = consensus.get('confirmation_level', 0.0)
            
            summary = f"Multi-Timeframe Analysis:\n"
            summary += f"Consensus Signal: {signal}\n"
            summary += f"Signal Strength: {strength.name}\n"
            summary += f"Confirmation Level: {confirmation:.1%}\n\n"
            
            summary += "Timeframe Breakdown:\n"
            for tf_key, tf_data in results.items():
                if tf_key == 'consensus':
                    continue
                    
                tf_signal = tf_data.get('recommendation', 'HOLD')
                tf_score = tf_data.get('signal_score', 0)
                summary += f"  {tf_key}: {tf_signal} (score: {tf_score:+.2f})\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return "Multi-timeframe analysis unavailable"


# Global instance
_mtf_analyzer = None

def get_mtf_analyzer() -> MultiTimeframeAnalyzer:
    """Get the global multi-timeframe analyzer instance"""
    global _mtf_analyzer
    if _mtf_analyzer is None:
        _mtf_analyzer = MultiTimeframeAnalyzer()
    return _mtf_analyzer