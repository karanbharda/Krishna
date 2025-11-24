#!/usr/bin/env python3
"""
Dynamic Position Sizer with Risk Management
Implements adaptive position sizing based on market conditions, volatility, and risk parameters
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Global instance for singleton pattern
_position_sizer_instance = None


@dataclass
class PositionResult:
    """Data class for position sizing results"""
    quantity: int
    position_value: float
    base_size: float
    constrained_size: float
    method_used: str
    constraints_applied: list
    stop_loss: float
    take_profit: float
    risk_metrics: Dict[str, Any]


class DynamicPositionSizer:
    """Dynamic position sizing with multiple methods and risk constraints"""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = 0.1  # 10% of capital
        self.risk_per_trade = 0.02     # 2% risk per trade
        self.volatility_threshold = 0.3  # 30% volatility threshold
        self.sentiment_sensitivity = 0.5   # Sentiment sensitivity multiplier
        self.news_confidence_weight = 0.3  # Weight for news confidence in position sizing

    def update_capital(self, new_capital: float):
        """Update current capital for position sizing calculations"""
        self.current_capital = new_capital

    def calculate_position_size(self, symbol: str, signal_strength: float,
                                current_price: float, volatility: float,
                                historical_data: pd.DataFrame,
                                portfolio_data: Dict[str, Any],
                                market_regime: str = "NORMAL",
                                sentiment_score: float = None,
                                news_confidence: float = None) -> PositionResult:
        """
        Calculate position size using multiple adaptive methods

        Args:
            symbol: Stock symbol
            signal_strength: Confidence level of the signal (0-1)
            current_price: Current market price
            volatility: Current volatility measure
            historical_data: Historical price data
            portfolio_data: Current portfolio information
            market_regime: Market condition ("BULL", "BEAR", "NORMAL", "VOLATILE")
            sentiment_score: News sentiment score (-1 to 1)
            news_confidence: Confidence in news analysis (0-1)
        """
        try:
            # Validate inputs
            if current_price <= 0:
                raise ValueError("Current price must be positive")

            if signal_strength < 0 or signal_strength > 1:
                logger.warning(
                    f"Signal strength {signal_strength} outside valid range [0,1]")
                signal_strength = max(0, min(signal_strength, 1))

            # Base position sizing using Kelly Criterion
            base_size = self._kelly_criterion_position(
                signal_strength, current_price, volatility)

            # Apply volatility-based sizing
            volatility_adjusted_size = self._volatility_adjusted_position(
                base_size, volatility, market_regime)

            # Apply adaptive sizing based on market conditions
            adaptive_size = self._adaptive_position_sizing(
                volatility_adjusted_size, market_regime, portfolio_data)

            # Apply sentiment and news confidence adjustments
            final_size = self._sentiment_adjusted_position(
                adaptive_size, sentiment_score, news_confidence)

            # Apply risk constraints
            constrained_size, constraints_applied = self._apply_risk_constraints(
                final_size, current_price, volatility, portfolio_data, sentiment_score, news_confidence)

            # Calculate quantity
            quantity = int(constrained_size / current_price)

            # Ensure minimum quantity
            if quantity < 1:
                quantity = 0
                constrained_size = 0

            # Calculate stop loss and take profit levels
            stop_loss, take_profit = self._calculate_risk_levels(
                current_price, volatility, market_regime)

            # Calculate risk metrics
            position_value = quantity * current_price
            risk_metrics = self._calculate_risk_metrics(
                position_value, current_price, stop_loss, volatility)

            return PositionResult(
                quantity=quantity,
                position_value=position_value,
                base_size=base_size,
                constrained_size=constrained_size,
                method_used="adaptive",
                constraints_applied=constraints_applied,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_metrics=risk_metrics
            )

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            # Return conservative position
            return PositionResult(
                quantity=0,
                position_value=0,
                base_size=0,
                constrained_size=0,
                method_used="error",
                constraints_applied=["error"],
                stop_loss=current_price * 0.95,
                take_profit=current_price * 1.10,
                risk_metrics={}
            )

    def _kelly_criterion_position(self, signal_strength: float,
                                  current_price: float, volatility: float) -> float:
        """Calculate position size using modified Kelly Criterion"""
        try:
            # Modified Kelly formula for stock trading
            win_rate = signal_strength
            # Assume average win/loss ratio based on signal strength
            win_loss_ratio = 1.5 + (signal_strength * 1.0)  # 1.5 to 2.5

            # Kelly formula: f* = p - (1-p)/b
            kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio

            # Conservative approach: use fraction of Kelly
            kelly_fraction = max(0, min(kelly_fraction, 1.0)) * 0.25

            # Position size based on capital and Kelly fraction
            position_size = self.current_capital * kelly_fraction

            return position_size

        except Exception as e:
            logger.error(f"Error in Kelly criterion calculation: {e}")
            return self.current_capital * 0.01  # 1% fallback

    def _volatility_adjusted_position(self, base_size: float,
                                      volatility: float, market_regime: str) -> float:
        """Adjust position size based on volatility and market regime"""
        try:
            # Base volatility adjustment
            vol_adjustment = max(
                0.1, 1.0 - (volatility / self.volatility_threshold))

            # Market regime adjustments
            regime_multiplier = 1.0
            if market_regime == "BULL":
                regime_multiplier = 1.2
            elif market_regime == "BEAR":
                regime_multiplier = 0.8
            elif market_regime == "VOLATILE":
                regime_multiplier = 0.6

            adjusted_size = base_size * vol_adjustment * regime_multiplier

            return adjusted_size

        except Exception as e:
            logger.error(f"Error in volatility adjustment: {e}")
            return base_size * 0.8

    def _adaptive_position_sizing(self, base_size: float, market_regime: str,
                                  portfolio_data: Dict[str, Any]) -> float:
        """Apply adaptive sizing based on portfolio and market conditions"""
        try:
            # Portfolio-based adjustments
            total_value = portfolio_data.get(
                'total_value', self.current_capital)
            cash = portfolio_data.get('cash', self.current_capital)
            holdings = portfolio_data.get('holdings', {})

            # Concentration risk: reduce position if already heavily invested
            num_positions = len(holdings)
            concentration_factor = max(0.5, 1.0 - (num_positions * 0.05))

            # Cash availability adjustment
            cash_ratio = cash / total_value if total_value > 0 else 1.0
            # Normalized to 20% cash target
            cash_factor = min(1.0, cash_ratio / 0.2)

            # Combine factors
            adaptive_factor = concentration_factor * cash_factor
            adaptive_size = base_size * adaptive_factor

            return adaptive_size

        except Exception as e:
            logger.error(f"Error in adaptive sizing: {e}")
            return base_size

    def _sentiment_adjusted_position(self, base_size: float,
                                     sentiment_score: float,
                                     news_confidence: float) -> float:
        """Adjust position size based on sentiment and news confidence"""
        try:
            # Start with base size
            adjusted_size = base_size

            # Apply sentiment adjustment if provided
            if sentiment_score is not None:
                # Enhanced sentiment multiplier with configurable sensitivity
                sentiment_multiplier = 1.0 + \
                    (sentiment_score * self.sentiment_sensitivity)
                sentiment_multiplier = max(0.5, min(sentiment_multiplier, 2.0))
                adjusted_size = adjusted_size * sentiment_multiplier

            # Apply news confidence adjustment if provided
            if news_confidence is not None:
                # Enhanced news confidence multiplier (0.5 to 2.0)
                confidence_multiplier = 0.5 + (news_confidence * 1.5)
                confidence_multiplier = max(
                    0.5, min(confidence_multiplier, 2.0))
                adjusted_size = adjusted_size * confidence_multiplier

            return adjusted_size

        except Exception as e:
            logger.error(f"Error in sentiment adjustment: {e}")
            return base_size

    def _apply_risk_constraints(self, proposed_size: float, current_price: float,
                                volatility: float, portfolio_data: Dict[str, Any],
                                sentiment_score: float = None, news_confidence: float = None) -> tuple:
        """Apply risk management constraints to position size"""
        try:
            constraints_applied = []

            # 1. Max position size constraint (percentage of capital)
            max_allowed = self.current_capital * self.max_position_size
            if proposed_size > max_allowed:
                proposed_size = max_allowed
                constraints_applied.append("max_position_size")

            # 2. Risk per trade constraint
            max_risk_amount = self.current_capital * self.risk_per_trade
            # Estimate potential loss based on volatility
            estimated_loss_pct = volatility * 2.0  # 2 standard deviations
            estimated_loss_amount = proposed_size * estimated_loss_pct

            if estimated_loss_amount > max_risk_amount:
                # Reduce position size to meet risk constraint
                reduction_factor = max_risk_amount / estimated_loss_amount
                proposed_size = proposed_size * reduction_factor
                constraints_applied.append("risk_per_trade")

            # 3. Portfolio cash constraint
            available_cash = portfolio_data.get('cash', self.current_capital)
            if proposed_size > available_cash:
                proposed_size = available_cash
                constraints_applied.append("available_cash")

            # 4. Minimum position size (at least $100 worth)
            min_position_value = 100.0
            if proposed_size < min_position_value and proposed_size > 0:
                proposed_size = min_position_value
                constraints_applied.append("min_position_size")

            # 5. Enhanced sentiment-based constraint (news-confidence override)
            if news_confidence is not None:
                # High confidence news can increase position size up to 2.0x
                high_confidence_multiplier = 1.0 + (news_confidence * 1.0)
                high_confidence_size = proposed_size * high_confidence_multiplier
                # But still respect max position size constraint
                max_allowed_with_confidence = self.current_capital * self.max_position_size * 2.0
                if high_confidence_size <= max_allowed_with_confidence:
                    proposed_size = high_confidence_size
                    constraints_applied.append("news_confidence_override")

            # 6. Volatility-based constraint
            if volatility is not None and volatility > self.volatility_threshold:
                # Reduce position size for high volatility
                volatility_reduction = max(
                    0.1, 1.0 - (volatility - self.volatility_threshold))
                proposed_size = proposed_size * volatility_reduction
                constraints_applied.append("volatility_filter")

            return proposed_size, constraints_applied

        except Exception as e:
            logger.error(f"Error applying risk constraints: {e}")
            # Conservative fallback
            return min(proposed_size, self.current_capital * 0.01), ["error"]

    def _calculate_risk_levels(self, current_price: float, volatility: float,
                               market_regime: str) -> tuple:
        """Calculate stop loss and take profit levels"""
        try:
            # Base stop loss based on volatility
            vol_multiplier = 2.0
            if market_regime == "VOLATILE":
                vol_multiplier = 2.5
            elif market_regime == "BULL":
                vol_multiplier = 1.5

            stop_loss_distance = current_price * volatility * vol_multiplier
            stop_loss = current_price - stop_loss_distance

            # Take profit based on risk-reward ratio
            risk_reward_ratio = 2.0
            if market_regime == "BULL":
                risk_reward_ratio = 2.5
            elif market_regime == "BEAR":
                risk_reward_ratio = 1.5

            take_profit_distance = stop_loss_distance * risk_reward_ratio
            take_profit = current_price + take_profit_distance

            # Ensure reasonable levels
            stop_loss = max(stop_loss, current_price *
                            0.8)  # Max 20% stop loss
            take_profit = min(take_profit, current_price *
                              1.5)  # Max 50% take profit

            return stop_loss, take_profit

        except Exception as e:
            logger.error(f"Error calculating risk levels: {e}")
            # Conservative fallback
            return current_price * 0.95, current_price * 1.10

    def _calculate_risk_metrics(self, position_value: float, current_price: float,
                                stop_loss: float, volatility: float) -> Dict:
        """
        Calculate comprehensive risk metrics for the position
        """
        try:
            # Validate inputs
            if position_value <= 0 or current_price <= 0 or volatility <= 0:
                # If any input is invalid, return default risk metrics
                return {
                    'position_risk': 0,
                    'portfolio_risk_pct': 0,
                    'max_loss_pct': 0.05,
                    'daily_var_95': 0,
                    'stop_loss_price': stop_loss if stop_loss > 0 else current_price * 0.95,
                    'sharpe_estimate': 0,
                    'volatility_annualized': 0.20
                }

            # Position risk (max loss to stop loss)
            max_loss = (current_price - stop_loss) / \
                current_price if current_price > 0 else 0
            # Ensure max_loss is reasonable (0 to 1)
            max_loss = max(0.0, min(max_loss, 1.0))

            position_risk = position_value * max_loss

            # Portfolio risk (as % of total capital)
            portfolio_risk_pct = position_risk / \
                self.current_capital if self.current_capital > 0 else 0
            # Ensure portfolio risk is reasonable (0 to 1)
            portfolio_risk_pct = max(0.0, min(portfolio_risk_pct, 1.0))

            # Daily VaR (95% confidence)
            daily_var_95 = position_value * volatility * 1.645  # 95% confidence
            # Ensure VaR is reasonable
            daily_var_95 = max(0.0, min(daily_var_95, position_value))

            # Sharpe ratio estimate (simplified)
            expected_return = 0.10  # Assume 10% expected annual return
            risk_free_rate = 0.03   # Assume 3% risk-free rate
            # Ensure non-negative
            excess_return = max(0, expected_return - risk_free_rate)
            annualized_vol = volatility * np.sqrt(252)
            sharpe_estimate = excess_return / annualized_vol if annualized_vol > 0 else 0
            # Ensure Sharpe ratio is reasonable (-5 to 5)
            sharpe_estimate = max(-5.0, min(sharpe_estimate, 5.0))

            return {
                'position_risk': position_risk,
                'portfolio_risk_pct': portfolio_risk_pct,
                'max_loss_pct': max_loss,
                'daily_var_95': daily_var_95,
                'stop_loss_price': stop_loss,
                'sharpe_estimate': sharpe_estimate,
                'volatility_annualized': annualized_vol
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                'position_risk': 0,
                'portfolio_risk_pct': 0,
                'max_loss_pct': 0.05,
                'daily_var_95': 0,
                'stop_loss_price': stop_loss,
                'sharpe_estimate': 0,
                'volatility_annualized': 0.20
            }


def get_position_sizer(initial_capital: float = 100000.0) -> DynamicPositionSizer:
    """Get singleton instance of DynamicPositionSizer"""
    global _position_sizer_instance
    if _position_sizer_instance is None:
        _position_sizer_instance = DynamicPositionSizer(initial_capital)
    return _position_sizer_instance
