"""
Market Circuit Breaker System
Implements circuit breakers for extreme market movements to protect trading capital
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CircuitBreakerLevel(Enum):
    """Circuit breaker levels"""
    NORMAL = "normal"
    WARNING = "warning"
    TRADING_HALT = "trading_halt"
    FULL_HALT = "full_halt"


@dataclass
class MarketShock:
    """Represents a market shock event"""
    timestamp: datetime
    symbol: str
    price_change_pct: float
    volume_spike: float
    volatility_spike: float
    description: str


class MarketCircuitBreaker:
    """
    Implements circuit breakers for extreme market movements
    """
    
    def __init__(self):
        # Circuit breaker thresholds
        self.price_change_thresholds = {
            'single_stock_5min': 0.10,  # 10% in 5 minutes
            'single_stock_1hour': 0.15,  # 15% in 1 hour
            'portfolio_1hour': 0.08,    # 8% portfolio loss in 1 hour
            'market_index_1hour': 0.05  # 5% market index drop in 1 hour
        }
        
        self.volatility_thresholds = {
            'extreme_volatility': 0.05,  # 5% intraday volatility
            'ultra_volatility': 0.10     # 10% intraday volatility
        }
        
        self.volume_thresholds = {
            'volume_spike': 5.0,  # 5x average volume
            'ultra_spike': 10.0   # 10x average volume
        }
        
        # Circuit breaker states
        self.current_level = CircuitBreakerLevel.NORMAL
        self.activation_time = None
        self.deactivation_time = None
        self.halt_duration = timedelta(minutes=15)
        
        # Market monitoring data
        self.price_history = {}
        self.volatility_history = {}
        self.volume_history = {}
        self.shock_events = []
        
        # Portfolio tracking
        self.portfolio_value_history = []
        
        # Market index tracking
        self.market_index_history = {}
        
        # Correlation tracking
        self.correlation_matrix = {}
        
        logger.info("✅ Market Circuit Breaker initialized")
    
    def __init__(self):
        # Circuit breaker thresholds
        self.price_change_thresholds = {
            'single_stock_5min': 0.10,  # 10% in 5 minutes
            'single_stock_1hour': 0.15,  # 15% in 1 hour
            'portfolio_1hour': 0.08,    # 8% portfolio loss in 1 hour
            'market_index_1hour': 0.05  # 5% market index drop in 1 hour
        }
        
        self.volatility_thresholds = {
            'extreme_volatility': 0.05,  # 5% intraday volatility
            'ultra_volatility': 0.10     # 10% intraday volatility
        }
        
        self.volume_thresholds = {
            'volume_spike': 5.0,  # 5x average volume
            'ultra_spike': 10.0   # 10x average volume
        }
        
        # Circuit breaker states
        self.current_level = CircuitBreakerLevel.NORMAL
        self.activation_time = None
        self.deactivation_time = None
        self.halt_duration = timedelta(minutes=15)
        
        # Market monitoring data
        self.price_history = {}
        self.volatility_history = {}
        self.volume_history = {}
        self.shock_events = []
        
        # Portfolio tracking
        self.portfolio_value_history = []
        
        logger.info("✅ Market Circuit Breaker initialized")
    
    def monitor_market_conditions(self, 
                                symbol: str, 
                                current_price: float, 
                                volume: float,
                                portfolio_value: float,
                                market_index: float = None,
                                correlation_data: Dict = None) -> CircuitBreakerLevel:
        """
        Monitor market conditions and update circuit breaker level
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            volume: Current volume
            portfolio_value: Current portfolio value
            market_index: Market index value (optional)
            correlation_data: Portfolio correlation data (optional)
            
        Returns:
            Current circuit breaker level
        """
        try:
            # Update price history
            self._update_price_history(symbol, current_price)
            
            # Update portfolio history
            self._update_portfolio_history(portfolio_value)
            
            # Update market index history if provided
            if market_index is not None:
                self._update_market_index_history(market_index)
            
            # Update correlation data if provided
            if correlation_data is not None:
                self._update_correlation_data(correlation_data)
            
            # Check for circuit breaker triggers
            new_level = self._assess_circuit_breaker_level(
                symbol, current_price, volume, portfolio_value, market_index
            )
            
            # Update circuit breaker state if needed
            if new_level != self.current_level:
                self._update_circuit_breaker_state(new_level)
            
            return self.current_level
            
        except Exception as e:
            logger.error(f"Error monitoring market conditions: {e}")
            return self.current_level
    
    def _update_price_history(self, symbol: str, price: float):
        """Update price history for a symbol"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'timestamp': datetime.now(),
            'price': price
        })
        
        # Keep only recent history (last 2 hours)
        cutoff_time = datetime.now() - timedelta(hours=2)
        self.price_history[symbol] = [
            entry for entry in self.price_history[symbol]
            if entry['timestamp'] > cutoff_time
        ]
    
    def _update_market_index_history(self, market_index: float):
        """Update market index history"""
        self.market_index_history[datetime.now()] = market_index
        
        # Keep only recent history (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.market_index_history = {
            timestamp: value for timestamp, value in self.market_index_history.items()
            if timestamp > cutoff_time
        }
    
    def _update_correlation_data(self, correlation_data: Dict):
        """Update correlation data"""
        self.correlation_matrix = correlation_data
    
    def _update_portfolio_history(self, portfolio_value: float):
        """Update portfolio value history"""
        self.portfolio_value_history.append({
            'timestamp': datetime.now(),
            'value': portfolio_value
        })
        
        # Keep only recent history (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.portfolio_value_history = [
            entry for entry in self.portfolio_value_history
            if entry['timestamp'] > cutoff_time
        ]
    
    def _assess_circuit_breaker_level(self, 
                                    symbol: str, 
                                    current_price: float, 
                                    volume: float,
                                    portfolio_value: float,
                                    market_index: float = None) -> CircuitBreakerLevel:
        """
        Assess current circuit breaker level based on market conditions
        """
        try:
            # Check for extreme price movements
            price_shock = self._check_price_shocks(symbol, current_price)
            if price_shock:
                self._record_shock_event(price_shock)
            
            # Check for portfolio drawdowns
            portfolio_shock = self._check_portfolio_shocks(portfolio_value)
            if portfolio_shock:
                self._record_shock_event(portfolio_shock)
            
            # Check for market index shocks
            if market_index:
                market_shock = self._check_market_index_shocks(market_index)
                if market_shock:
                    self._record_shock_event(market_shock)
            
            # Check for volatility spikes
            volatility_shock = self._check_volatility_spikes(symbol)
            if volatility_shock:
                self._record_shock_event(volatility_shock)
            
            # Determine circuit breaker level based on shocks
            return self._determine_circuit_breaker_level()
            
        except Exception as e:
            logger.error(f"Error assessing circuit breaker level: {e}")
            return CircuitBreakerLevel.NORMAL
    
    def _check_price_shocks(self, symbol: str, current_price: float) -> Optional[MarketShock]:
        """Check for extreme price movements"""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
                return None
            
            history = self.price_history[symbol]
            current_time = datetime.now()
            
            # Check 5-minute price change
            five_min_ago = current_time - timedelta(minutes=5)
            recent_prices = [entry for entry in history if entry['timestamp'] >= five_min_ago]
            
            if len(recent_prices) >= 2:
                price_change_pct = (current_price - recent_prices[0]['price']) / recent_prices[0]['price']
                if abs(price_change_pct) > self.price_change_thresholds['single_stock_5min']:
                    return MarketShock(
                        timestamp=current_time,
                        symbol=symbol,
                        price_change_pct=price_change_pct,
                        volume_spike=0.0,
                        volatility_spike=0.0,
                        description=f"Extreme 5-minute price movement: {price_change_pct:.2%}"
                    )
            
            # Check 1-hour price change
            one_hour_ago = current_time - timedelta(hours=1)
            hourly_prices = [entry for entry in history if entry['timestamp'] >= one_hour_ago]
            
            if len(hourly_prices) >= 2:
                price_change_pct = (current_price - hourly_prices[0]['price']) / hourly_prices[0]['price']
                if abs(price_change_pct) > self.price_change_thresholds['single_stock_1hour']:
                    return MarketShock(
                        timestamp=current_time,
                        symbol=symbol,
                        price_change_pct=price_change_pct,
                        volume_spike=0.0,
                        volatility_spike=0.0,
                        description=f"Extreme 1-hour price movement: {price_change_pct:.2%}"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking price shocks: {e}")
            return None
    
    def _check_portfolio_shocks(self, portfolio_value: float) -> Optional[MarketShock]:
        """Check for extreme portfolio drawdowns"""
        try:
            if len(self.portfolio_value_history) < 2:
                return None
            
            current_time = datetime.now()
            one_hour_ago = current_time - timedelta(hours=1)
            hourly_values = [entry for entry in self.portfolio_value_history if entry['timestamp'] >= one_hour_ago]
            
            if len(hourly_values) >= 2:
                value_change_pct = (portfolio_value - hourly_values[0]['value']) / hourly_values[0]['value']
                if value_change_pct < -self.price_change_thresholds['portfolio_1hour']:
                    return MarketShock(
                        timestamp=current_time,
                        symbol="PORTFOLIO",
                        price_change_pct=value_change_pct,
                        volume_spike=0.0,
                        volatility_spike=0.0,
                        description=f"Extreme portfolio drawdown: {value_change_pct:.2%}"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking portfolio shocks: {e}")
            return None
    
    def _check_market_index_shocks(self, market_index: float) -> Optional[MarketShock]:
        """Check for extreme market index movements"""
        try:
            if len(self.market_index_history) < 2:
                return None
            
            current_time = datetime.now()
            one_hour_ago = current_time - timedelta(hours=1)
            
            # Get historical index values from last hour
            recent_values = [
                value for timestamp, value in self.market_index_history.items()
                if timestamp >= one_hour_ago
            ]
            
            if len(recent_values) >= 2:
                index_change_pct = (market_index - recent_values[0]) / recent_values[0]
                if abs(index_change_pct) > self.price_change_thresholds['market_index_1hour']:
                    return MarketShock(
                        timestamp=current_time,
                        symbol="MARKET_INDEX",
                        price_change_pct=index_change_pct,
                        volume_spike=0.0,
                        volatility_spike=0.0,
                        description=f"Extreme market index movement: {index_change_pct:.2%}"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking market index shocks: {e}")
            return None
    
    def _check_volatility_spikes(self, symbol: str) -> Optional[MarketShock]:
        """Check for extreme volatility spikes"""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
                return None
            
            # Calculate recent volatility
            recent_prices = [entry['price'] for entry in self.price_history[symbol][-20:]]
            if len(recent_prices) < 2:
                return None
            
            returns = [((recent_prices[i] / recent_prices[i-1]) - 1) for i in range(1, len(recent_prices))]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Check against thresholds
            if volatility > self.volatility_thresholds['ultra_volatility']:
                return MarketShock(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    price_change_pct=0.0,
                    volume_spike=0.0,
                    volatility_spike=volatility,
                    description=f"Ultra volatility spike: {volatility:.2%}"
                )
            elif volatility > self.volatility_thresholds['extreme_volatility']:
                return MarketShock(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    price_change_pct=0.0,
                    volume_spike=0.0,
                    volatility_spike=volatility,
                    description=f"Extreme volatility spike: {volatility:.2%}"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking volatility spikes: {e}")
            return None
    
    def _record_shock_event(self, shock: MarketShock):
        """Record a market shock event"""
        self.shock_events.append(shock)
        
        # Keep only recent events (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.shock_events = [event for event in self.shock_events if event.timestamp > cutoff_time]
        
        logger.warning(f"Market shock recorded: {shock.description}")
    
    def _determine_circuit_breaker_level(self) -> CircuitBreakerLevel:
        """Determine circuit breaker level based on recent shocks"""
        try:
            recent_shocks = [
                shock for shock in self.shock_events 
                if shock.timestamp > datetime.now() - timedelta(minutes=30)
            ]
            
            if not recent_shocks:
                return CircuitBreakerLevel.NORMAL
            
            # Check for severe shocks
            severe_shocks = [
                shock for shock in recent_shocks
                if (abs(shock.price_change_pct) > 0.15 or 
                    shock.volatility_spike > self.volatility_thresholds['ultra_volatility'])
            ]
            
            if severe_shocks:
                return CircuitBreakerLevel.FULL_HALT
            
            # Check for moderate shocks
            moderate_shocks = [
                shock for shock in recent_shocks
                if (abs(shock.price_change_pct) > 0.10 or 
                    shock.volatility_spike > self.volatility_thresholds['extreme_volatility'])
            ]
            
            if moderate_shocks:
                return CircuitBreakerLevel.TRADING_HALT
            
            # Check for mild shocks
            mild_shocks = [
                shock for shock in recent_shocks
                if abs(shock.price_change_pct) > 0.05
            ]
            
            if mild_shocks:
                return CircuitBreakerLevel.WARNING
            
            return CircuitBreakerLevel.NORMAL
            
        except Exception as e:
            logger.error(f"Error determining circuit breaker level: {e}")
            return CircuitBreakerLevel.NORMAL
    
    def _update_circuit_breaker_state(self, new_level: CircuitBreakerLevel):
        """Update circuit breaker state"""
        old_level = self.current_level
        self.current_level = new_level
        self.activation_time = datetime.now() if new_level != CircuitBreakerLevel.NORMAL else None
        
        logger.info(f"Circuit breaker level changed: {old_level.value} → {new_level.value}")
        
        # Log specific actions based on level
        if new_level == CircuitBreakerLevel.WARNING:
            logger.warning("⚠️ Market circuit breaker: WARNING level activated")
        elif new_level == CircuitBreakerLevel.TRADING_HALT:
            logger.critical("🛑 Market circuit breaker: TRADING HALT activated")
        elif new_level == CircuitBreakerLevel.FULL_HALT:
            logger.critical("🚨 Market circuit breaker: FULL HALT activated")
    
    def should_halt_trading(self) -> bool:
        """Check if trading should be halted"""
        if self.current_level in [CircuitBreakerLevel.TRADING_HALT, CircuitBreakerLevel.FULL_HALT]:
            # Check if halt period has expired
            if self.activation_time:
                if datetime.now() - self.activation_time > self.halt_duration:
                    self.current_level = CircuitBreakerLevel.WARNING
                    return False
                return True
            return True
        return False
    
    def should_reduce_positions(self) -> bool:
        """Check if positions should be reduced"""
        return self.current_level in [CircuitBreakerLevel.WARNING, CircuitBreakerLevel.TRADING_HALT]
    
    def get_circuit_breaker_status(self) -> Dict:
        """Get current circuit breaker status"""
        return {
            'level': self.current_level.value,
            'activation_time': self.activation_time.isoformat() if self.activation_time else None,
            'recent_shocks': len([
                shock for shock in self.shock_events 
                if shock.timestamp > datetime.now() - timedelta(hours=1)
            ]),
            'portfolio_value_history_length': len(self.portfolio_value_history)
        }
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker to normal state"""
        self.current_level = CircuitBreakerLevel.NORMAL
        self.activation_time = None
        self.deactivation_time = datetime.now()
        logger.info("Circuit breaker manually reset to NORMAL")


# Global instance
_market_circuit_breaker = None


def get_market_circuit_breaker() -> MarketCircuitBreaker:
    """Get global market circuit breaker instance"""
    global _market_circuit_breaker
    if _market_circuit_breaker is None:
        _market_circuit_breaker = MarketCircuitBreaker()
    return _market_circuit_breaker