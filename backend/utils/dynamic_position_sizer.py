"""
Phase 2: Dynamic Position Sizing System
Implements Kelly Criterion and volatility-based position sizing for optimal risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methods"""
    KELLY_CRITERION = "kelly"
    VOLATILITY_BASED = "volatility"
    FIXED_PERCENT = "fixed"
    RISK_PARITY = "risk_parity"
    ADAPTIVE = "adaptive"


class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = 1
    MODERATE = 2
    AGGRESSIVE = 3
    VERY_AGGRESSIVE = 4


class DynamicPositionSizer:
    """
    Advanced position sizing system using multiple algorithms
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = 0.25  # 25% max per position
        self.max_total_exposure = 0.95  # 95% max total exposure
        
        # Kelly Criterion parameters
        self.kelly_lookback_period = 50
        self.kelly_max_position = 0.20  # Cap Kelly at 20%
        self.kelly_min_position = 0.01  # Minimum 1%
        
        # Volatility parameters
        self.volatility_target = 0.15  # Target 15% portfolio volatility
        self.volatility_lookback = 30
        self.volatility_regime_adjustments = {
            "NORMAL": 1.0,
            "VOLATILE": 0.8,
            "TRENDING": 1.2
        }
        
        # Risk management
        self.stop_loss_atr_multiplier = 2.0
        self.position_correlation_limit = 0.7
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {}
        
        logger.info("✅ Dynamic Position Sizer initialized")
    
    def calculate_position_size(self, 
                              symbol: str, 
                              signal_strength: float,
                              current_price: float,
                              volatility: float,
                              historical_data: pd.DataFrame,
                              portfolio_data: Dict,
                              method: SizingMethod = SizingMethod.ADAPTIVE,
                              market_regime: str = "NORMAL") -> Dict:
        """
        Calculate optimal position size based on multiple factors
        """
        try:
            # Update current capital
            self.current_capital = portfolio_data.get('total_value', self.initial_capital)
            
            # Calculate base position sizes using different methods
            kelly_size = self._calculate_kelly_size(symbol, historical_data, signal_strength)
            volatility_size = self._calculate_volatility_size(current_price, volatility, market_regime)
            risk_parity_size = self._calculate_risk_parity_size(volatility, portfolio_data)
            
            # Select sizing method
            if method == SizingMethod.KELLY_CRITERION:
                base_size = kelly_size
            elif method == SizingMethod.VOLATILITY_BASED:
                base_size = volatility_size
            elif method == SizingMethod.RISK_PARITY:
                base_size = risk_parity_size
            elif method == SizingMethod.ADAPTIVE:
                # Combine multiple methods with weights
                base_size = self._adaptive_sizing(kelly_size, volatility_size, risk_parity_size, signal_strength)
            else:
                base_size = self._calculate_fixed_size()
            
            # ENHANCEMENT: Adjust Kelly limits based on market conditions
            # Make Kelly fraction more responsive to market regime
            if market_regime == "HIGH_VOLATILITY":
                self.kelly_max_position = 0.10  # More conservative in high volatility
            elif market_regime == "LOW_VOLATILITY":
                self.kelly_max_position = 0.25  # More aggressive in low volatility
            else:
                self.kelly_max_position = 0.20  # Default
            
            # Apply risk management constraints
            constrained_size = self._apply_risk_constraints(
                base_size, symbol, current_price, portfolio_data
            )
            
            # Calculate actual quantities and risk metrics
            position_value = constrained_size * self.current_capital
            quantity = int(position_value / current_price)
            actual_value = quantity * current_price
            actual_size = actual_value / self.current_capital
            
            # Calculate stop loss
            stop_loss = self._calculate_stop_loss(current_price, volatility)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                actual_value, current_price, stop_loss, volatility
            )
            
            result = {
                'symbol': symbol,
                'method_used': method.value,
                'base_size': base_size,
                'constrained_size': constrained_size,
                'actual_size': actual_size,
                'position_value': actual_value,
                'quantity': quantity,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'risk_metrics': risk_metrics,
                'sizing_components': {
                    'kelly_size': kelly_size,
                    'volatility_size': volatility_size,
                    'risk_parity_size': risk_parity_size
                },
                'constraints_applied': actual_size < base_size,
                'market_regime': market_regime,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Position size calculated for {symbol}: {actual_size:.2%} (${actual_value:,.0f})")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return self._fallback_position_size(symbol, current_price)
    
    def _calculate_kelly_size(self, symbol: str, historical_data: pd.DataFrame, signal_strength: float) -> float:
        """
        Calculate position size using Kelly Criterion
        """
        try:
            if len(historical_data) < self.kelly_lookback_period:
                logger.warning(f"Insufficient data for Kelly calculation: {len(historical_data)}")
                return self.kelly_min_position
            
            # Calculate returns
            returns = historical_data['Close'].pct_change().dropna()
            
            if len(returns) < 20:
                return self.kelly_min_position
            
            # Estimate win rate and average win/loss
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(positive_returns) == 0 or len(negative_returns) == 0:
                return self.kelly_min_position
            
            win_rate = len(positive_returns) / len(returns)
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            
            # Kelly Criterion: f = (bp - q) / b
            # where b = odds received on the wager, p = win probability, q = loss probability
            if avg_loss == 0:
                return self.kelly_min_position
            
            odds_ratio = avg_win / avg_loss
            kelly_fraction = (win_rate * odds_ratio - (1 - win_rate)) / odds_ratio
            
            # Adjust for signal strength
            kelly_fraction *= signal_strength
            
            # Apply bounds
            kelly_fraction = max(self.kelly_min_position, min(self.kelly_max_position, kelly_fraction))
            
            logger.debug(f"Kelly calculation: win_rate={win_rate:.3f}, avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f}, kelly={kelly_fraction:.3f}")
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error in Kelly calculation: {e}")
            return self.kelly_min_position
    
    def _calculate_volatility_size(self, current_price: float, volatility: float, market_regime: str = "NORMAL") -> float:
        """
        Calculate position size based on volatility targeting
        Enhanced to be more responsive to market regime
        """
        try:
            if volatility <= 0:
                volatility = 0.20  # Default 20% volatility assumption
            
            # ENHANCEMENT: Improve Risk Parameter Tuning
            # Make volatility targets more responsive to market regime
            regime_adjustment = self.volatility_regime_adjustments.get(market_regime, 1.0)
            adjusted_volatility_target = self.volatility_target * regime_adjustment
            
            # Target portfolio volatility approach
            target_position_vol = adjusted_volatility_target
            volatility_size = target_position_vol / volatility
            
            # Bounds
            volatility_size = max(0.01, min(0.30, volatility_size))
            
            logger.debug(f"Volatility size: {volatility_size:.3f} (vol: {volatility:.3f}, regime: {market_regime}, adj: {regime_adjustment})")
            return volatility_size
            
        except Exception as e:
            logger.error(f"Error in volatility sizing: {e}")
            return 0.10
    
    def _calculate_risk_parity_size(self, volatility: float, portfolio_data: Dict) -> float:
        """
        Calculate position size for risk parity approach
        """
        try:
            current_holdings = portfolio_data.get('holdings', {})
            
            if not current_holdings:
                # No existing positions, use default
                return 0.10
            
            # Calculate risk contribution of existing positions
            total_risk_contribution = 0
            for holding_symbol, holding_data in current_holdings.items():
                holding_value = holding_data.get('current_value', 0)
                holding_weight = holding_value / self.current_capital
                # Assume average volatility for existing holdings if not available
                holding_vol = 0.25  # Default assumption
                risk_contribution = holding_weight * holding_vol
                total_risk_contribution += risk_contribution
            
            # Target equal risk contribution
            target_risk_per_position = self.volatility_target / (len(current_holdings) + 1)
            
            if volatility <= 0:
                volatility = 0.20
            
            risk_parity_size = target_risk_per_position / volatility
            risk_parity_size = max(0.01, min(0.20, risk_parity_size))
            
            logger.debug(f"Risk parity size: {risk_parity_size:.3f}")
            return risk_parity_size
            
        except Exception as e:
            logger.error(f"Error in risk parity calculation: {e}")
            return 0.10
    
    def _adaptive_sizing(self, kelly_size: float, volatility_size: float, 
                        risk_parity_size: float, signal_strength: float) -> float:
        """
        Combine multiple sizing methods adaptively
        """
        try:
            # Weight based on signal strength and market conditions
            kelly_weight = 0.4 * signal_strength
            volatility_weight = 0.3
            risk_parity_weight = 0.3 * (1 - signal_strength)
            
            # Normalize weights
            total_weight = kelly_weight + volatility_weight + risk_parity_weight
            if total_weight > 0:
                kelly_weight /= total_weight
                volatility_weight /= total_weight
                risk_parity_weight /= total_weight
            else:
                kelly_weight = volatility_weight = risk_parity_weight = 1/3
            
            adaptive_size = (kelly_weight * kelly_size + 
                           volatility_weight * volatility_size + 
                           risk_parity_weight * risk_parity_size)
            
            logger.debug(f"Adaptive sizing: {adaptive_size:.3f} (weights: K:{kelly_weight:.2f}, V:{volatility_weight:.2f}, R:{risk_parity_weight:.2f})")
            return adaptive_size
            
        except Exception as e:
            logger.error(f"Error in adaptive sizing: {e}")
            return np.mean([kelly_size, volatility_size, risk_parity_size])
    
    def _calculate_fixed_size(self) -> float:
        """
        Calculate fixed percentage position size
        """
        return 0.10  # Fixed 10%
    
    def _apply_risk_constraints(self, base_size: float, symbol: str, 
                              current_price: float, portfolio_data: Dict) -> float:
        """
        Apply risk management constraints to position size
        """
        try:
            constrained_size = base_size
            
            # 1. Maximum position size constraint
            constrained_size = min(constrained_size, self.max_position_size)
            
            # 2. Portfolio exposure constraint
            current_exposure = self._calculate_current_exposure(portfolio_data)
            available_exposure = self.max_total_exposure - current_exposure
            max_size_by_exposure = available_exposure
            constrained_size = min(constrained_size, max_size_by_exposure)
            
            # 3. Correlation constraint (simplified)
            max_size_by_correlation = self._apply_correlation_constraint(
                symbol, constrained_size, portfolio_data
            )
            constrained_size = min(constrained_size, max_size_by_correlation)
            
            # 4. Minimum position size
            if constrained_size < 0.005:  # Less than 0.5%
                constrained_size = 0.0  # Don't trade
            
            # 5. Cash constraint
            available_cash = portfolio_data.get('cash', 0)
            max_position_value = constrained_size * self.current_capital
            if max_position_value > available_cash:
                constrained_size = available_cash / self.current_capital
            
            logger.debug(f"Size constraints applied: {base_size:.3f} → {constrained_size:.3f}")
            return max(0.0, constrained_size)
            
        except Exception as e:
            logger.error(f"Error applying risk constraints: {e}")
            return min(base_size, 0.05)  # Conservative fallback
    
    def _calculate_current_exposure(self, portfolio_data: Dict) -> float:
        """
        Calculate current portfolio exposure
        """
        try:
            holdings = portfolio_data.get('holdings', {})
            total_holdings_value = sum(
                holding.get('current_value', 0) for holding in holdings.values()
            )
            exposure = total_holdings_value / self.current_capital
            return exposure
            
        except Exception as e:
            logger.error(f"Error calculating exposure: {e}")
            return 0.0
    
    def _apply_correlation_constraint(self, symbol: str, proposed_size: float, 
                                    portfolio_data: Dict) -> float:
        """
        Apply correlation constraint (simplified implementation)
        """
        try:
            # Simplified: limit concentration in similar sectors
            # In a full implementation, this would use actual correlation data
            
            holdings = portfolio_data.get('holdings', {})
            sector_exposure = {}
            
            # Group by simplified sector (first letter assumption)
            for holding_symbol in holdings.keys():
                sector = holding_symbol[0] if holding_symbol else 'OTHER'
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0
                holding_value = holdings[holding_symbol].get('current_value', 0)
                sector_exposure[sector] += holding_value / self.current_capital
            
            # Check if new position would exceed sector concentration
            new_sector = symbol[0] if symbol else 'OTHER'
            current_sector_exposure = sector_exposure.get(new_sector, 0)
            max_sector_exposure = 0.40  # 40% max per sector
            
            if current_sector_exposure + proposed_size > max_sector_exposure:
                max_additional = max_sector_exposure - current_sector_exposure
                return max(0.0, max_additional)
            
            return proposed_size
            
        except Exception as e:
            logger.error(f"Error applying correlation constraint: {e}")
            return proposed_size
    
    def _calculate_stop_loss(self, current_price: float, volatility: float) -> float:
        """
        Calculate stop loss level based on volatility
        """
        try:
            # Use ATR-based stop loss
            atr_estimate = volatility * current_price
            stop_loss = current_price - (self.stop_loss_atr_multiplier * atr_estimate)
            
            # Ensure stop loss is reasonable (not more than 15% down)
            max_stop_loss = current_price * 0.85
            stop_loss = max(stop_loss, max_stop_loss)
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return current_price * 0.95  # 5% stop loss as fallback
    
    def _calculate_risk_metrics(self, position_value: float, current_price: float, 
                              stop_loss: float, volatility: float) -> Dict:
        """
        Calculate comprehensive risk metrics for the position
        """
        try:
            # Position risk (max loss to stop loss)
            max_loss = (current_price - stop_loss) / current_price
            position_risk = position_value * max_loss
            
            # Portfolio risk (as % of total capital)
            portfolio_risk_pct = position_risk / self.current_capital
            
            # Daily VaR (95% confidence)
            daily_var_95 = position_value * volatility * 1.645  # 95% confidence
            
            # Sharpe ratio estimate (simplified)
            expected_return = 0.10  # Assume 10% expected annual return
            risk_free_rate = 0.03   # Assume 3% risk-free rate
            excess_return = expected_return - risk_free_rate
            annualized_vol = volatility * np.sqrt(252)
            sharpe_estimate = excess_return / annualized_vol if annualized_vol > 0 else 0
            
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
    
    def _fallback_position_size(self, symbol: str, current_price: float) -> Dict:
        """
        Fallback position sizing when calculations fail
        """
        fallback_size = 0.05  # 5% fallback
        position_value = fallback_size * self.current_capital
        quantity = int(position_value / current_price)
        
        return {
            'symbol': symbol,
            'method_used': 'fallback',
            'base_size': fallback_size,
            'constrained_size': fallback_size,
            'actual_size': fallback_size,
            'position_value': position_value,
            'quantity': quantity,
            'current_price': current_price,
            'stop_loss': current_price * 0.95,
            'risk_metrics': {'portfolio_risk_pct': 0.05},
            'constraints_applied': False,
            'timestamp': datetime.now().isoformat()
        }
    
    def update_performance(self, trade_result: Dict):
        """
        Update performance tracking for sizing algorithm improvement
        """
        try:
            self.trade_history.append(trade_result)
            
            # Keep only recent trades for performance calculation
            if len(self.trade_history) > 100:
                self.trade_history = self.trade_history[-100:]
            
            # Update performance metrics
            self._calculate_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def _calculate_performance_metrics(self):
        """
        Calculate performance metrics for sizing optimization
        """
        try:
            if not self.trade_history:
                return
            
            returns = [trade.get('return_pct', 0) for trade in self.trade_history]
            
            self.performance_metrics = {
                'total_trades': len(self.trade_history),
                'avg_return': np.mean(returns),
                'return_std': np.std(returns),
                'win_rate': len([r for r in returns if r > 0]) / len(returns),
                'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculate maximum drawdown from returns
        """
        try:
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return float(np.min(drawdown))
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def get_sizing_summary(self) -> Dict:
        """
        Get summary of position sizing system
        """
        return {
            'current_capital': self.current_capital,
            'max_position_size': self.max_position_size,
            'max_total_exposure': self.max_total_exposure,
            'performance_metrics': self.performance_metrics,
            'total_trades_tracked': len(self.trade_history),
            'sizing_methods': [method.value for method in SizingMethod]
        }
    
    def optimize_parameters(self):
        """
        Optimize sizing parameters based on performance
        """
        try:
            if len(self.trade_history) < 20:
                return  # Need more data
            
            # Simple parameter optimization based on Sharpe ratio
            current_sharpe = self.performance_metrics.get('sharpe_ratio', 0)
            
            # Adjust Kelly fraction based on performance
            if current_sharpe < 0.5:  # Poor performance
                self.kelly_max_position *= 0.9  # Reduce Kelly
            elif current_sharpe > 1.5:  # Good performance  
                self.kelly_max_position *= 1.05  # Slightly increase Kelly
            
            # Keep within bounds
            self.kelly_max_position = max(0.05, min(0.25, self.kelly_max_position))
            
            logger.info(f"Parameters optimized: Kelly max = {self.kelly_max_position:.3f}")
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")


# Global instance
_position_sizer = None

def get_position_sizer(initial_capital: float = 100000) -> DynamicPositionSizer:
    """Get the global position sizer instance"""
    global _position_sizer
    if _position_sizer is None:
        _position_sizer = DynamicPositionSizer(initial_capital)
    return _position_sizer