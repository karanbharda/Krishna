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
        
        # Load dynamic configuration from live_config.json
        self._load_dynamic_config()
        
        # Kelly Criterion parameters
        self.kelly_lookback_period = 50
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
    
    def _load_dynamic_config(self):
        """Load dynamic configuration from live_config.json"""
        try:
            import json
            import os
            
            # FIXED: Correct path to project root data directory
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'live_config.json')
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    live_config = json.load(f)
                
                # Get dynamic values from frontend config
                self.max_position_size = live_config.get("max_capital_per_trade", 0.25)  # Use frontend value or 25% default
                self.max_total_exposure = live_config.get("max_total_exposure", 0.95)   # Use frontend value or 95% default
                self.kelly_max_position = live_config.get("kelly_max_position", 0.20)   # Use frontend value or 20% default
                
                logger.info(f"📊 Loaded dynamic config - Max Position: {self.max_position_size:.1%}, Total Exposure: {self.max_total_exposure:.1%}, Kelly Max: {self.kelly_max_position:.1%}")
            else:
                logger.warning(f"live_config.json not found at {config_path}, using hardcoded defaults")
                self.max_position_size = 0.25  # 25% max per position
                self.max_total_exposure = 0.95  # 95% max total exposure
                self.kelly_max_position = 0.20  # Cap Kelly at 20%
                
        except Exception as e:
            logger.error(f"Failed to load dynamic config: {e}")
            self.max_position_size = 0.25  # 25% max per position
            self.max_total_exposure = 0.95  # 95% max total exposure
            self.kelly_max_position = 0.20  # Cap Kelly at 20%
    
    def refresh_dynamic_config(self):
        """Refresh dynamic configuration from live_config.json (call this periodically)"""
        self._load_dynamic_config()

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
            # Validate inputs
            validated_inputs = self._validate_inputs(
                symbol, signal_strength, current_price, volatility, 
                historical_data, portfolio_data
            )
            
            if not validated_inputs['valid']:
                logger.warning(f"Invalid inputs for {symbol}: {validated_inputs['errors']}")
                return self._fallback_position_size(symbol, current_price)
            
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
                # Use enhanced adaptive sizing that considers volatility and correlation
                base_size = self._enhanced_adaptive_sizing(kelly_size, volatility_size, risk_parity_size, 
                                                         signal_strength, volatility, portfolio_data, market_regime)
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
    
    def _validate_inputs(self, symbol: str, signal_strength: float, current_price: float, 
                        volatility: float, historical_data: pd.DataFrame, portfolio_data: Dict) -> Dict:
        """
        Validate all inputs to prevent extreme values
        """
        errors = []
        valid = True
        
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            errors.append("Invalid symbol")
            valid = False
        
        # Validate signal strength (0.0 to 1.0)
        if not isinstance(signal_strength, (int, float)) or signal_strength < 0 or signal_strength > 1:
            errors.append(f"Invalid signal strength: {signal_strength}")
            valid = False
        
        # Validate current price (must be positive)
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            errors.append(f"Invalid current price: {current_price}")
            valid = False
        elif current_price > 1000000:  # Extremely high price
            errors.append(f"Extremely high price: {current_price}")
            valid = False
        
        # Validate volatility (0.0 to 1.0 is reasonable range)
        if not isinstance(volatility, (int, float)) or volatility < 0:
            errors.append(f"Invalid volatility: {volatility}")
            valid = False
        elif volatility > 1.0:  # Extremely high volatility
            errors.append(f"Extremely high volatility: {volatility}")
            valid = False
        
        # Validate historical data
        if not isinstance(historical_data, pd.DataFrame):
            errors.append("Invalid historical data type")
            valid = False
        elif len(historical_data) < 10:  # Need minimum data
            errors.append(f"Insufficient historical data: {len(historical_data)} rows")
            valid = False
        
        # Validate portfolio data
        if not isinstance(portfolio_data, dict):
            errors.append("Invalid portfolio data type")
            valid = False
        else:
            total_value = portfolio_data.get('total_value', 0)
            if not isinstance(total_value, (int, float)) or total_value < 0:
                errors.append(f"Invalid portfolio total value: {total_value}")
                valid = False
            elif total_value > 1000000000:  # Extremely high portfolio value
                errors.append(f"Extremely high portfolio value: {total_value}")
                valid = False
        
        return {'valid': valid, 'errors': errors}
    
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
            if avg_loss <= 0 or avg_win <= 0:
                return self.kelly_min_position
            
            odds_ratio = avg_win / avg_loss
            # Prevent extreme odds ratios
            odds_ratio = max(0.1, min(odds_ratio, 10.0))
            
            kelly_fraction = (win_rate * odds_ratio - (1 - win_rate)) / odds_ratio
            
            # Prevent negative Kelly fractions
            kelly_fraction = max(0.0, kelly_fraction)
            
            # Adjust for signal strength
            kelly_fraction *= signal_strength
            
            # Apply bounds with more conservative maximum
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
        Calculate position size for risk parity approach with enhanced correlation handling
        """
        try:
            current_holdings = portfolio_data.get('holdings', {})
            
            if not current_holdings:
                # No existing positions, use default
                return 0.10
            
            # Calculate risk contribution of existing positions with better correlation handling
            total_risk_contribution = 0
            portfolio_volatility = 0
            
            # Get correlation data if available
            correlations = portfolio_data.get('correlations', {})
            
            for holding_symbol, holding_data in current_holdings.items():
                holding_value = holding_data.get('current_value', 0)
                holding_weight = holding_value / self.current_capital if self.current_capital > 0 else 0
                
                # Use actual volatility if available, otherwise default
                holding_vol = holding_data.get('volatility', 0.25)
                
                # Calculate risk contribution with correlation adjustment
                if correlations and holding_symbol in correlations:
                    # Use correlation-adjusted risk contribution
                    correlation_adjusted_risk = 0
                    for other_symbol, other_data in current_holdings.items():
                        if other_symbol != holding_symbol:
                            other_vol = other_data.get('volatility', 0.25)
                            correlation = correlations.get(holding_symbol, {}).get(other_symbol, 0.5)
                            correlation_adjusted_risk += holding_weight * holding_vol * other_vol * correlation
                    risk_contribution = holding_weight * holding_vol + correlation_adjusted_risk
                else:
                    # Fallback to simple risk contribution
                    risk_contribution = holding_weight * holding_vol
                
                total_risk_contribution += risk_contribution
                portfolio_volatility += (holding_weight ** 2) * (holding_vol ** 2)
            
            # Add cross-correlation terms
            if correlations:
                for i, (symbol1, data1) in enumerate(current_holdings.items()):
                    for j, (symbol2, data2) in enumerate(current_holdings.items()):
                        if i < j:  # Avoid double counting
                            weight1 = data1.get('current_value', 0) / self.current_capital if self.current_capital > 0 else 0
                            weight2 = data2.get('current_value', 0) / self.current_capital if self.current_capital > 0 else 0
                            vol1 = data1.get('volatility', 0.25)
                            vol2 = data2.get('volatility', 0.25)
                            correlation = correlations.get(symbol1, {}).get(symbol2, 0.5)
                            portfolio_volatility += 2 * weight1 * weight2 * vol1 * vol2 * correlation
            
            portfolio_volatility = np.sqrt(portfolio_volatility) if portfolio_volatility > 0 else 0.25
            
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
                        risk_parity_size: float, signal_strength: float, 
                        market_regime: str = "NORMAL") -> float:
        """
        Combine multiple sizing methods adaptively with enhanced correlation and volatility consideration
        """
        try:
            # Adjust weights based on market regime
            if market_regime == "HIGH_VOLATILITY":
                # In high volatility, rely more on risk parity and less on Kelly
                kelly_weight = 0.2 * signal_strength
                volatility_weight = 0.4
                risk_parity_weight = 0.4 * (1 - signal_strength)
            elif market_regime == "LOW_VOLATILITY":
                # In low volatility, can be more aggressive with Kelly
                kelly_weight = 0.5 * signal_strength
                volatility_weight = 0.2
                risk_parity_weight = 0.3 * (1 - signal_strength)
            else:
                # Normal market conditions
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
            
            logger.debug(f"Adaptive sizing: {adaptive_size:.3f} (weights: K:{kelly_weight:.2f}, V:{volatility_weight:.2f}, R:{risk_parity_weight:.2f}, regime: {market_regime})")
            return adaptive_size
            
        except Exception as e:
            logger.error(f"Error in adaptive sizing: {e}")
            return np.mean([kelly_size, volatility_size, risk_parity_size])
    
    def _enhanced_adaptive_sizing(self, kelly_size: float, volatility_size: float, 
                                risk_parity_size: float, signal_strength: float,
                                volatility: float, portfolio_data: Dict,
                                market_regime: str = "NORMAL") -> float:
        """
        Enhanced adaptive sizing that considers volatility clustering and portfolio correlation
        """
        try:
            # Get current portfolio correlation data
            correlations = portfolio_data.get('correlations', {})
            current_holdings = portfolio_data.get('holdings', {})
            
            # Calculate portfolio correlation risk factor
            portfolio_corr_risk = self._calculate_portfolio_correlation_risk(correlations, current_holdings)
            
            # Adjust sizing based on volatility regime
            volatility_adjustment = self._calculate_volatility_adjustment(volatility, market_regime)
            
            # Calculate base adaptive size using existing method
            base_size = self._adaptive_sizing(kelly_size, volatility_size, risk_parity_size, 
                                            signal_strength, market_regime)
            
            # Apply correlation and volatility adjustments
            adjusted_size = base_size * volatility_adjustment * portfolio_corr_risk
            
            # Ensure reasonable bounds
            adjusted_size = max(0.005, min(0.30, adjusted_size))
            
            logger.debug(f"Enhanced adaptive sizing: base={base_size:.3f}, vol_adj={volatility_adjustment:.3f}, corr_risk={portfolio_corr_risk:.3f}, final={adjusted_size:.3f}")
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error in enhanced adaptive sizing: {e}")
            # Fallback to standard adaptive sizing
            return self._adaptive_sizing(kelly_size, volatility_size, risk_parity_size, signal_strength, market_regime)
    
    def _calculate_portfolio_correlation_risk(self, correlations: Dict, holdings: Dict) -> float:
        """
        Calculate correlation risk factor for the portfolio (0.5 to 1.5)
        Lower factor when portfolio is well diversified, higher when concentrated
        """
        try:
            if not correlations or len(holdings) < 2:
                return 1.0  # Neutral factor
            
            # Calculate average correlation
            corr_values = []
            for symbol1, symbol_corrs in correlations.items():
                for symbol2, corr in symbol_corrs.items():
                    if symbol1 != symbol2 and symbol2 in correlations:
                        corr_values.append(abs(corr))
            
            if not corr_values:
                return 1.0
            
            avg_correlation = np.mean(corr_values)
            
            # Convert to risk factor (lower correlation = lower risk factor)
            # Range: 0.5 (well diversified) to 1.5 (highly correlated)
            correlation_risk_factor = 0.5 + avg_correlation
            
            return max(0.5, min(1.5, correlation_risk_factor))
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 1.0
    
    def _calculate_volatility_adjustment(self, volatility: float, market_regime: str) -> float:
        """
        Calculate volatility adjustment factor based on current market volatility
        """
        try:
            # Base adjustment based on volatility relative to target
            vol_ratio = volatility / self.volatility_target if self.volatility_target > 0 else 1.0
            
            # Market regime adjustments
            regime_multipliers = {
                "HIGH_VOLATILITY": 0.7,
                "LOW_VOLATILITY": 1.2,
                "NORMAL": 1.0
            }
            regime_mult = regime_multipliers.get(market_regime, 1.0)
            
            # Volatility adjustment (higher volatility = lower position size)
            if vol_ratio > 2.0:  # Very high volatility
                vol_adjustment = 0.5
            elif vol_ratio > 1.5:  # High volatility
                vol_adjustment = 0.7
            elif vol_ratio > 1.0:  # Moderate-high volatility
                vol_adjustment = 0.85
            elif vol_ratio < 0.5:  # Very low volatility
                vol_adjustment = 1.3
            elif vol_ratio < 0.75:  # Low volatility
                vol_adjustment = 1.15
            else:  # Normal volatility
                vol_adjustment = 1.0
            
            final_adjustment = vol_adjustment * regime_mult
            
            return max(0.3, min(1.5, final_adjustment))
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {e}")
            return 1.0
    
    def _calculate_volatility_adjustment(self, volatility: float, market_regime: str) -> float:
        """
        Calculate volatility adjustment factor based on current market volatility
        """
        try:
            # Base adjustment based on volatility relative to target
            vol_ratio = volatility / self.volatility_target if self.volatility_target > 0 else 1.0
            
            # Market regime adjustments
            regime_multipliers = {
                "HIGH_VOLATILITY": 0.7,
                "LOW_VOLATILITY": 1.2,
                "NORMAL": 1.0
            }
            regime_mult = regime_multipliers.get(market_regime, 1.0)
            
            # Volatility adjustment (higher volatility = lower position size)
            if vol_ratio > 2.0:  # Very high volatility
                vol_adjustment = 0.5
            elif vol_ratio > 1.5:  # High volatility
                vol_adjustment = 0.7
            elif vol_ratio > 1.0:  # Moderate-high volatility
                vol_adjustment = 0.85
            elif vol_ratio < 0.5:  # Very low volatility
                vol_adjustment = 1.3
            elif vol_ratio < 0.75:  # Low volatility
                vol_adjustment = 1.15
            else:  # Normal volatility
                vol_adjustment = 1.0
            
            final_adjustment = vol_adjustment * regime_mult
            
            return max(0.3, min(1.5, final_adjustment))
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {e}")
            return 1.0
    
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
            
            # Ensure stop loss is reasonable (not less than 15% down)
            min_stop_loss = current_price * 0.85
            stop_loss = max(stop_loss, min_stop_loss)
            
            # Ensure stop loss is always below current price for long positions
            stop_loss = min(stop_loss, current_price * 0.99)  # At least 1% buffer
            
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
            # Validate inputs
            if position_value <= 0 or current_price <= 0 or volatility <= 0:
                raise ValueError("Invalid input values for risk metrics calculation")
            
            # Position risk (max loss to stop loss)
            max_loss = (current_price - stop_loss) / current_price if current_price > 0 else 0
            # Ensure max_loss is reasonable (0 to 1)
            max_loss = max(0.0, min(max_loss, 1.0))
            
            position_risk = position_value * max_loss
            
            # Portfolio risk (as % of total capital)
            portfolio_risk_pct = position_risk / self.current_capital if self.current_capital > 0 else 0
            # Ensure portfolio risk is reasonable (0 to 1)
            portfolio_risk_pct = max(0.0, min(portfolio_risk_pct, 1.0))
            
            # Daily VaR (95% confidence)
            daily_var_95 = position_value * volatility * 1.645  # 95% confidence
            # Ensure VaR is reasonable
            daily_var_95 = max(0.0, min(daily_var_95, position_value))
            
            # Sharpe ratio estimate (simplified)
            expected_return = 0.10  # Assume 10% expected annual return
            risk_free_rate = 0.03   # Assume 3% risk-free rate
            excess_return = max(0, expected_return - risk_free_rate)  # Ensure non-negative
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
    
    def optimize_parameters(self, market_regime: str = "NORMAL"):
        """
        Optimize sizing parameters based on performance
        Enhanced to be more responsive to market conditions
        """
        try:
            if len(self.trade_history) < 20:
                return  # Need more data
            
            # Simple parameter optimization based on Sharpe ratio
            current_sharpe = self.performance_metrics.get('sharpe_ratio', 0)
            
            # Adjust Kelly fraction based on performance and market regime
            if current_sharpe < 0.5:  # Poor performance
                self.kelly_max_position *= 0.9  # Reduce Kelly
            elif current_sharpe > 1.5:  # Good performance  
                self.kelly_max_position *= 1.05  # Slightly increase Kelly
            
            # Adjust based on market regime
            if market_regime == "HIGH_VOLATILITY":
                self.kelly_max_position *= 0.8  # Be more conservative
            elif market_regime == "LOW_VOLATILITY":
                self.kelly_max_position *= 1.1  # Can be more aggressive
            
            # Keep within bounds
            self.kelly_max_position = max(0.05, min(0.25, self.kelly_max_position))
            
            logger.info(f"Parameters optimized: Kelly max = {self.kelly_max_position:.3f}, Market Regime = {market_regime}")
            
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