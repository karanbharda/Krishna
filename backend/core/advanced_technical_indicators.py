"""
Advanced Technical Indicators Module
Professional-grade technical analysis with institutional indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AdvancedTechnicalData:
    """Container for advanced technical indicator data"""
    # Money Flow Index
    mfi: float = 0.0
    mfi_signal: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL

    # Put/Call Ratio
    pc_ratio: float = 0.0
    pc_ratio_signal: str = "NEUTRAL"

    # Order Book Data
    order_book_imbalance: float = 0.0
    bid_ask_pressure: float = 0.0
    order_flow: str = "NEUTRAL"

    # Advanced Oscillators
    stochastic_k: float = 50.0
    stochastic_d: float = 50.0
    stochastic_signal: str = "NEUTRAL"

    williams_r: float = -50.0
    williams_r_signal: str = "NEUTRAL"

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_position: float = 0.5  # 0-1 scale
    bb_signal: str = "NEUTRAL"

    # Volume Indicators
    volume_rate_of_change: float = 0.0
    on_balance_volume: float = 0.0
    volume_price_trend: float = 0.0

    # Advanced Moving Averages
    hull_moving_average: float = 0.0
    weighted_moving_average: float = 0.0

    # Market Breadth
    advance_decline_ratio: float = 1.0
    new_highs_new_lows_ratio: float = 1.0

class AdvancedTechnicalIndicators:
    """
    Advanced Technical Indicators Calculator
    Implements institutional-grade technical indicators for professional trading
    """

    def __init__(self, lookback_period: int = 14):
        self.lookback_period = lookback_period
        self.price_history = []
        self.volume_history = []
        self.high_history = []
        self.low_history = []

    def update_price_data(self, prices: List[float], volumes: List[float] = None,
                         highs: List[float] = None, lows: List[float] = None):
        """Update historical price data for calculations"""
        self.price_history = prices[-100:]  # Keep last 100 periods
        if volumes:
            self.volume_history = volumes[-100:]
        if highs:
            self.high_history = highs[-100:]
        if lows:
            self.low_history = lows[-100:]

    def calculate_money_flow_index(self, prices: List[float], volumes: List[float],
                                  highs: List[float], lows: List[float]) -> Tuple[float, str]:
        """
        Calculate Money Flow Index (MFI)
        MFI = 100 - (100 / (1 + Money Flow Ratio))
        """
        if len(prices) < self.lookback_period + 1:
            return 50.0, "NEUTRAL"

        # Calculate typical price
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, prices)]

        # Calculate raw money flow
        money_flows = []
        for i in range(1, len(typical_prices)):
            if typical_prices[i] > typical_prices[i-1]:
                money_flows.append(typical_prices[i] * volumes[i])
            elif typical_prices[i] < typical_prices[i-1]:
                money_flows.append(-typical_prices[i] * volumes[i])
            else:
                money_flows.append(0)

        # Calculate positive and negative money flows
        positive_flow = sum(mf for mf in money_flows[-self.lookback_period:] if mf > 0)
        negative_flow = abs(sum(mf for mf in money_flows[-self.lookback_period:] if mf < 0))

        if negative_flow == 0:
            mfi = 100.0
        else:
            money_ratio = positive_flow / negative_flow
            mfi = 100 - (100 / (1 + money_ratio))

        # Generate signal
        if mfi > 80:
            signal = "BEARISH"  # Overbought
        elif mfi < 20:
            signal = "BULLISH"  # Oversold
        else:
            signal = "NEUTRAL"

        return round(mfi, 2), signal

    def calculate_put_call_ratio(self, put_volume: float, call_volume: float) -> Tuple[float, str]:
        """
        Calculate Put/Call Ratio and generate signal
        PCR > 1.0 indicates bearish sentiment, PCR < 1.0 indicates bullish sentiment
        """
        if call_volume == 0:
            return 1.0, "NEUTRAL"

        pc_ratio = put_volume / call_volume

        # Generate signal based on extreme readings
        if pc_ratio > 1.5:
            signal = "BULLISH"  # Extreme fear (contrarian bullish)
        elif pc_ratio < 0.7:
            signal = "BEARISH"  # Extreme greed (contrarian bearish)
        else:
            signal = "NEUTRAL"

        return round(pc_ratio, 3), signal

    def calculate_order_book_analysis(self, bid_volume: float, ask_volume: float,
                                    bid_prices: List[float], ask_prices: List[float]) -> Tuple[float, float, str]:
        """
        Analyze order book for pressure and imbalance
        Returns: (order_book_imbalance, bid_ask_pressure, order_flow_signal)
        """
        if bid_volume == 0 and ask_volume == 0:
            return 0.0, 0.0, "NEUTRAL"

        # Order book imbalance (positive = bullish pressure, negative = bearish pressure)
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            imbalance = 0.0
        else:
            imbalance = (bid_volume - ask_volume) / total_volume

        # Bid-ask pressure based on price levels
        avg_bid_price = np.mean(bid_prices) if bid_prices else 0
        avg_ask_price = np.mean(ask_prices) if ask_prices else 0

        if avg_bid_price == 0 and avg_ask_price == 0:
            pressure = 0.0
        else:
            pressure = (avg_bid_price - avg_ask_price) / max(avg_bid_price, avg_ask_price)

        # Generate order flow signal
        if imbalance > 0.3 and pressure > 0.1:
            order_flow = "BULLISH"
        elif imbalance < -0.3 and pressure < -0.1:
            order_flow = "BEARISH"
        else:
            order_flow = "NEUTRAL"

        return round(imbalance, 3), round(pressure, 3), order_flow

    def calculate_stochastic_oscillator(self, prices: List[float], highs: List[float],
                                      lows: List[float]) -> Tuple[float, float, str]:
        """
        Calculate Stochastic Oscillator (%K and %D)
        %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = 3-period SMA of %K
        """
        if len(prices) < self.lookback_period:
            return 50.0, 50.0, "NEUTRAL"

        # Calculate %K
        current_price = prices[-1]
        lowest_low = min(lows[-self.lookback_period:])
        highest_high = max(highs[-self.lookback_period:])

        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100

        # Calculate %D (3-period SMA of %K)
        if len(prices) < self.lookback_period + 3:
            d_percent = k_percent
        else:
            k_history = []
            for i in range(3):
                if i < len(prices):
                    low = min(lows[-self.lookback_period-i:-i] if i > 0 else lows[-self.lookback_period:])
                    high = max(highs[-self.lookback_period-i:-i] if i > 0 else highs[-self.lookback_period:])
                    price = prices[-(i+1)]
                    if high != low:
                        k_val = ((price - low) / (high - low)) * 100
                    else:
                        k_val = 50.0
                    k_history.append(k_val)

            d_percent = np.mean(k_history) if k_history else k_percent

        # Generate signal
        if k_percent > 80 and d_percent > 80:
            signal = "BEARISH"  # Overbought
        elif k_percent < 20 and d_percent < 20:
            signal = "BULLISH"  # Oversold
        else:
            signal = "NEUTRAL"

        return round(k_percent, 2), round(d_percent, 2), signal

    def calculate_williams_r(self, prices: List[float], highs: List[float],
                           lows: List[float]) -> Tuple[float, str]:
        """
        Calculate Williams %R
        %R = (Highest High - Current Close) / (Highest High - Lowest Low) * -100
        """
        if len(prices) < self.lookback_period:
            return -50.0, "NEUTRAL"

        current_price = prices[-1]
        highest_high = max(highs[-self.lookback_period:])
        lowest_low = min(lows[-self.lookback_period:])

        if highest_high == lowest_low:
            williams_r = -50.0
        else:
            williams_r = ((highest_high - current_price) / (highest_high - lowest_low)) * -100

        # Generate signal
        if williams_r > -20:
            signal = "BEARISH"  # Overbought
        elif williams_r < -80:
            signal = "BULLISH"  # Oversold
        else:
            signal = "NEUTRAL"

        return round(williams_r, 2), signal

    def calculate_bollinger_bands(self, prices: List[float], period: int = 20,
                                std_dev: float = 2.0) -> Tuple[float, float, float, float, str]:
        """
        Calculate Bollinger Bands
        Middle Band = SMA(20)
        Upper Band = SMA(20) + (StdDev * 2)
        Lower Band = SMA(20) - (StdDev * 2)
        """
        if len(prices) < period:
            current_price = prices[-1] if prices else 0
            return current_price, current_price, current_price, 0.5, "NEUTRAL"

        # Calculate middle band (SMA)
        middle_band = np.mean(prices[-period:])

        # Calculate standard deviation
        if len(prices[-period:]) < 2:
            std = 0
        else:
            std = np.std(prices[-period:])

        # Calculate bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        current_price = prices[-1]

        # Calculate position within bands (0-1 scale)
        if upper_band != lower_band:
            position = (current_price - lower_band) / (upper_band - lower_band)
        else:
            position = 0.5

        # Generate signal
        if position > 0.9:
            signal = "BEARISH"  # Near upper band
        elif position < 0.1:
            signal = "BULLISH"  # Near lower band
        else:
            signal = "NEUTRAL"

        return round(upper_band, 2), round(middle_band, 2), round(lower_band, 2), round(position, 2), signal

    def calculate_volume_indicators(self, prices: List[float], volumes: List[float]) -> Tuple[float, float, float]:
        """
        Calculate volume-based indicators
        Returns: (volume_roc, on_balance_volume, volume_price_trend)
        """
        if len(prices) < 2 or len(volumes) < 2:
            return 0.0, 0.0, 0.0

        # Volume Rate of Change
        current_volume = volumes[-1]
        previous_volume = volumes[-2] if len(volumes) > 1 else current_volume
        volume_roc = ((current_volume - previous_volume) / previous_volume) * 100 if previous_volume != 0 else 0

        # On Balance Volume (simplified calculation)
        obv = 0
        for i in range(1, min(len(prices), len(volumes))):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]

        # Volume Price Trend
        vpt = 0
        for i in range(1, min(len(prices), len(volumes))):
            if prices[i] != prices[i-1]:
                vpt += ((prices[i] - prices[i-1]) / prices[i-1]) * volumes[i]

        return round(volume_roc, 2), round(obv, 2), round(vpt, 2)

    def calculate_advanced_moving_averages(self, prices: List[float]) -> Tuple[float, float]:
        """
        Calculate advanced moving averages
        Returns: (hull_ma, weighted_ma)
        """
        if len(prices) < 10:
            current_price = prices[-1] if prices else 0
            return current_price, current_price

        # Hull Moving Average (HMA) - faster and smoother than traditional MA
        period = min(16, len(prices) // 2)  # Use half of available data, max 16

        # Calculate WMA for half period
        half_period = period // 2
        wma_half = self._calculate_wma(prices[-period:], half_period)

        # Calculate WMA for full period
        wma_full = self._calculate_wma(prices[-period:], period)

        # Calculate raw HMA
        raw_hma = 2 * wma_half - wma_full

        # Calculate final HMA (WMA of raw HMA)
        hma = self._calculate_wma([raw_hma] * min(period, len(raw_hma)), min(half_period, len([raw_hma])))

        # Weighted Moving Average (WMA)
        wma = self._calculate_wma(prices[-period:], period)

        return round(hma, 2), round(wma, 2)

    def _calculate_wma(self, prices: List[float], period: int) -> float:
        """Calculate Weighted Moving Average"""
        if len(prices) < period:
            return np.mean(prices) if prices else 0

        weights = np.arange(1, period + 1)
        wma_values = []

        for i in range(len(prices) - period + 1):
            weighted_sum = sum(price * weight for price, weight in zip(prices[i:i+period], weights))
            wma_values.append(weighted_sum / sum(weights))

        return wma_values[-1] if wma_values else (np.mean(prices) if prices else 0)

    def calculate_all_indicators(self, prices: List[float], volumes: List[float] = None,
                               highs: List[float] = None, lows: List[float] = None,
                               put_volume: float = 0, call_volume: float = 0,
                               bid_volume: float = 0, ask_volume: float = 0,
                               bid_prices: List[float] = None, ask_prices: List[float] = None) -> AdvancedTechnicalData:
        """
        Calculate all advanced technical indicators
        """
        # Ensure we have price data
        if not prices:
            return AdvancedTechnicalData()

        # Set defaults for optional data
        if volumes is None:
            volumes = [1000000] * len(prices)  # Default volume
        if highs is None:
            highs = prices
        if lows is None:
            lows = prices
        if bid_prices is None:
            bid_prices = []
        if ask_prices is None:
            ask_prices = []

        # Update internal data
        self.update_price_data(prices, volumes, highs, lows)

        # Calculate all indicators
        mfi, mfi_signal = self.calculate_money_flow_index(prices, volumes, highs, lows)
        pc_ratio, pc_ratio_signal = self.calculate_put_call_ratio(put_volume, call_volume)
        order_imbalance, pressure, order_flow = self.calculate_order_book_analysis(
            bid_volume, ask_volume, bid_prices, ask_prices)
        stoch_k, stoch_d, stoch_signal = self.calculate_stochastic_oscillator(prices, highs, lows)
        williams_r, williams_r_signal = self.calculate_williams_r(prices, highs, lows)
        bb_upper, bb_middle, bb_lower, bb_position, bb_signal = self.calculate_bollinger_bands(prices)
        vol_roc, obv, vpt = self.calculate_volume_indicators(prices, volumes)
        hull_ma, wma = self.calculate_advanced_moving_averages(prices)

        return AdvancedTechnicalData(
            mfi=mfi,
            mfi_signal=mfi_signal,
            pc_ratio=pc_ratio,
            pc_ratio_signal=pc_ratio_signal,
            order_book_imbalance=order_imbalance,
            bid_ask_pressure=pressure,
            order_flow=order_flow,
            stochastic_k=stoch_k,
            stochastic_d=stoch_d,
            stochastic_signal=stoch_signal,
            williams_r=williams_r,
            williams_r_signal=williams_r_signal,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_position=bb_position,
            bb_signal=bb_signal,
            volume_rate_of_change=vol_roc,
            on_balance_volume=obv,
            volume_price_trend=vpt,
            hull_moving_average=hull_ma,
            weighted_moving_average=wma,
            advance_decline_ratio=1.0,  # Placeholder
            new_highs_new_lows_ratio=1.0  # Placeholder
        )
