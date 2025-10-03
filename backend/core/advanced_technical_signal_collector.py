"""
Advanced Technical Signal Collectors
Integrates advanced indicators with the AsyncSignalCollector framework
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from .advanced_technical_indicators import AdvancedTechnicalIndicators, AdvancedTechnicalData

logger = logging.getLogger(__name__)

class AdvancedTechnicalSignalCollector:
    """
    Signal collector for advanced technical indicators
    """

    def __init__(self):
        self.indicators_calculator = AdvancedTechnicalIndicators()
        self.price_history = []
        self.volume_history = []
        self.high_history = []
        self.low_history = []

    async def collect_mfi_signal(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect Money Flow Index signal
        """
        try:
            # Get market data from context
            market_data = context.get('market_data', {})

            # Extract price and volume data
            prices = market_data.get('prices', [])
            volumes = market_data.get('volumes', [])
            highs = market_data.get('highs', prices)
            lows = market_data.get('lows', prices)

            if len(prices) < 14:
                return {
                    'strength': 0.0,
                    'confidence': 0.3,
                    'data': {'mfi': 50.0, 'signal': 'NEUTRAL', 'error': 'Insufficient data'}
                }

            # Update calculator data
            self.indicators_calculator.update_price_data(prices, volumes, highs, lows)

            # Calculate MFI
            mfi, mfi_signal = self.indicators_calculator.calculate_money_flow_index(
                prices, volumes, highs, lows)

            # Convert signal to strength (-1 to 1)
            if mfi_signal == "BULLISH":
                strength = min(mfi / 100, 1.0) if mfi < 50 else 0.0
            elif mfi_signal == "BEARISH":
                strength = max((100 - mfi) / 100 * -1, -1.0) if mfi > 50 else 0.0
            else:
                strength = 0.0

            confidence = 0.7 if 20 <= mfi <= 80 else 0.5

            return {
                'strength': strength,
                'confidence': confidence,
                'data': {
                    'mfi': mfi,
                    'signal': mfi_signal,
                    'interpretation': f"MFI at {mfi:.1f} indicating {mfi_signal.lower()} conditions"
                }
            }

        except Exception as e:
            logger.error(f"Error collecting MFI signal for {symbol}: {e}")
            return {
                'strength': 0.0,
                'confidence': 0.0,
                'data': {'error': str(e)}
            }

    async def collect_pc_ratio_signal(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect Put/Call Ratio signal
        """
        try:
            # Get options data from context
            options_data = context.get('options_data', {})

            put_volume = options_data.get('put_volume', 0)
            call_volume = options_data.get('call_volume', 0)

            if call_volume == 0:
                return {
                    'strength': 0.0,
                    'confidence': 0.3,
                    'data': {'pc_ratio': 1.0, 'signal': 'NEUTRAL', 'error': 'No call volume data'}
                }

            # Calculate PC ratio
            pc_ratio, pc_signal = self.indicators_calculator.calculate_put_call_ratio(put_volume, call_volume)

            # Convert signal to strength (-1 to 1)
            if pc_signal == "BULLISH":  # High put volume (fear) = bullish contrarian signal
                strength = min(pc_ratio / 2, 1.0) if pc_ratio > 1.0 else 0.0
            elif pc_signal == "BEARISH":  # Low put volume (greed) = bearish contrarian signal
                strength = max((1 - pc_ratio) / 1 * -1, -1.0) if pc_ratio < 1.0 else 0.0
            else:
                strength = 0.0

            confidence = 0.8 if pc_ratio > 0.5 and pc_ratio < 1.5 else 0.4

            return {
                'strength': strength,
                'confidence': confidence,
                'data': {
                    'pc_ratio': pc_ratio,
                    'signal': pc_signal,
                    'interpretation': f"Put/Call ratio at {pc_ratio:.3f} indicating {pc_signal.lower()} sentiment"
                }
            }

        except Exception as e:
            logger.error(f"Error collecting PC ratio signal for {symbol}: {e}")
            return {
                'strength': 0.0,
                'confidence': 0.0,
                'data': {'error': str(e)}
            }

    async def collect_order_book_signal(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect order book analysis signal
        """
        try:
            # Get order book data from context
            order_book = context.get('order_book', {})

            bid_volume = order_book.get('bid_volume', 0)
            ask_volume = order_book.get('ask_volume', 0)
            bid_prices = order_book.get('bid_prices', [])
            ask_prices = order_book.get('ask_prices', [])

            if bid_volume == 0 and ask_volume == 0:
                return {
                    'strength': 0.0,
                    'confidence': 0.3,
                    'data': {'imbalance': 0.0, 'pressure': 0.0, 'flow': 'NEUTRAL', 'error': 'No order book data'}
                }

            # Calculate order book metrics
            imbalance, pressure, order_flow = self.indicators_calculator.calculate_order_book_analysis(
                bid_volume, ask_volume, bid_prices, ask_prices)

            # Convert to strength (-1 to 1)
            strength = imbalance  # Already in -1 to 1 range
            confidence = 0.7 if abs(imbalance) > 0.1 else 0.4

            return {
                'strength': strength,
                'confidence': confidence,
                'data': {
                    'imbalance': imbalance,
                    'pressure': pressure,
                    'flow': order_flow,
                    'interpretation': f"Order book showing {order_flow.lower()} pressure with {imbalance:.3f} imbalance"
                }
            }

        except Exception as e:
            logger.error(f"Error collecting order book signal for {symbol}: {e}")
            return {
                'strength': 0.0,
                'confidence': 0.0,
                'data': {'error': str(e)}
            }

    async def collect_stochastic_signal(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect Stochastic Oscillator signal
        """
        try:
            # Get market data from context
            market_data = context.get('market_data', {})

            prices = market_data.get('prices', [])
            highs = market_data.get('highs', prices)
            lows = market_data.get('lows', prices)

            if len(prices) < 14:
                return {
                    'strength': 0.0,
                    'confidence': 0.3,
                    'data': {'stoch_k': 50.0, 'stoch_d': 50.0, 'signal': 'NEUTRAL', 'error': 'Insufficient data'}
                }

            # Calculate stochastic
            stoch_k, stoch_d, stoch_signal = self.indicators_calculator.calculate_stochastic_oscillator(
                prices, highs, lows)

            # Convert signal to strength (-1 to 1)
            if stoch_signal == "BULLISH":
                strength = min((stoch_k - 20) / 80, 1.0) if stoch_k < 50 else 0.0
            elif stoch_signal == "BEARISH":
                strength = max((80 - stoch_k) / 80 * -1, -1.0) if stoch_k > 50 else 0.0
            else:
                strength = 0.0

            confidence = 0.8 if (stoch_k < 20 or stoch_k > 80) else 0.5

            return {
                'strength': strength,
                'confidence': confidence,
                'data': {
                    'stoch_k': stoch_k,
                    'stoch_d': stoch_d,
                    'signal': stoch_signal,
                    'interpretation': f"Stochastic at {stoch_k:.1f}% indicating {stoch_signal.lower()} conditions"
                }
            }

        except Exception as e:
            logger.error(f"Error collecting stochastic signal for {symbol}: {e}")
            return {
                'strength': 0.0,
                'confidence': 0.0,
                'data': {'error': str(e)}
            }

    async def collect_bollinger_bands_signal(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect Bollinger Bands signal
        """
        try:
            # Get market data from context
            market_data = context.get('market_data', {})

            prices = market_data.get('prices', [])

            if len(prices) < 20:
                return {
                    'strength': 0.0,
                    'confidence': 0.3,
                    'data': {'position': 0.5, 'signal': 'NEUTRAL', 'error': 'Insufficient data'}
                }

            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_position, bb_signal = self.indicators_calculator.calculate_bollinger_bands(prices)

            # Convert position to strength (-1 to 1)
            # Lower position = bullish (near lower band), higher position = bearish (near upper band)
            strength = (0.5 - bb_position) * 2  # Convert 0-1 scale to -1 to 1

            confidence = 0.8 if bb_position < 0.1 or bb_position > 0.9 else 0.5

            return {
                'strength': strength,
                'confidence': confidence,
                'data': {
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'bb_lower': bb_lower,
                    'position': bb_position,
                    'signal': bb_signal,
                    'interpretation': f"Bollinger Band position at {bb_position:.2f} indicating {bb_signal.lower()} conditions"
                }
            }

        except Exception as e:
            logger.error(f"Error collecting Bollinger Bands signal for {symbol}: {e}")
            return {
                'strength': 0.0,
                'confidence': 0.0,
                'data': {'error': str(e)}
            }

    async def collect_volume_signal(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect volume-based signal
        """
        try:
            # Get market data from context
            market_data = context.get('market_data', {})

            prices = market_data.get('prices', [])
            volumes = market_data.get('volumes', [1000000] * len(prices))

            if len(prices) < 2 or len(volumes) < 2:
                return {
                    'strength': 0.0,
                    'confidence': 0.3,
                    'data': {'volume_roc': 0.0, 'obv': 0.0, 'vpt': 0.0, 'error': 'Insufficient data'}
                }

            # Calculate volume indicators
            volume_roc, obv, vpt = self.indicators_calculator.calculate_volume_indicators(prices, volumes)

            # Volume ROC as primary signal
            # Positive ROC = bullish, negative ROC = bearish
            strength = min(abs(volume_roc) / 50, 1.0) * (1 if volume_roc > 0 else -1) if volume_roc != 0 else 0.0

            confidence = 0.6 if abs(volume_roc) > 10 else 0.4

            return {
                'strength': strength,
                'confidence': confidence,
                'data': {
                    'volume_roc': volume_roc,
                    'obv': obv,
                    'vpt': vpt,
                    'interpretation': f"Volume ROC at {volume_roc:.1f}% indicating {'bullish' if volume_roc > 0 else 'bearish' if volume_roc < 0 else 'neutral'} volume trend"
                }
            }

        except Exception as e:
            logger.error(f"Error collecting volume signal for {symbol}: {e}")
            return {
                'strength': 0.0,
                'confidence': 0.0,
                'data': {'error': str(e)}
            }
