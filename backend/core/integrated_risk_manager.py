"""
Integrated Risk Management System
Comprehensive risk management with portfolio-level controls and dynamic adjustments
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .drawdown_protector import DrawdownProtector
from .correlation_manager import CorrelationManager
from .fee_optimizer import FeeOptimizer
from .professional_sell_logic import MarketContext

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Current risk metrics for portfolio and positions"""
    portfolio_value: float = 0.0
    total_exposure: float = 0.0
    sector_exposures: Dict[str, float] = field(default_factory=dict)
    max_position_size: float = 0.0
    portfolio_volatility: float = 0.0
    correlation_risk: float = 0.0
    current_drawdown: float = 0.0
    fees_paid: float = 0.0

@dataclass
class PositionRisk:
    """Risk metrics for a specific position"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    exposure: float
    stop_loss_price: float
    take_profit_price: float
    risk_amount: float
    reward_potential: float
    risk_reward_ratio: float
    atr: float
    volatility: float
    sector: str = ""

class IntegratedRiskManager:
    """Enhanced risk management with portfolio-level controls"""
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
            
        # ENHANCED RISK MANAGEMENT: Portfolio-level controls
        self.max_portfolio_risk_pct = config.get("max_portfolio_risk_pct", 0.05)  # 5% max portfolio risk
        self.max_single_stock_exposure = config.get("max_single_stock_exposure", 0.10)  # 10% max per stock
        self.max_sector_exposure = config.get("max_sector_exposure", 0.25)  # 25% max per sector
        self.correlation_risk_multiplier = config.get("correlation_risk_multiplier", 0.8)
        
        # Dynamic Stop Loss Configuration
        self.dynamic_stop_loss_enabled = config.get("dynamic_stop_loss_enabled", True)
        self.trailing_stop_enabled = config.get("trailing_stop_enabled", False)
        self.atr_stop_multiplier = config.get("atr_stop_multiplier", 1.5)
        
        # Risk Metrics Tracking
        self.portfolio_drawdown_limit = config.get("portfolio_drawdown_limit", 0.15)  # 15% max drawdown
        self.volatility_adjustment_enabled = config.get("volatility_adjustment_enabled", True)
        
        # Initialize risk components
        # Fix: Pass proper config dictionary to DrawdownProtector instead of float
        drawdown_config = {
            "max_drawdown_limit": self.portfolio_drawdown_limit,
            "daily_loss_limit": config.get("daily_loss_limit", 0.05),
            "recovery_threshold": config.get("recovery_threshold", 0.02),
            "scaling_factor": config.get("scaling_factor", 0.5),
            "monitoring_window": config.get("monitoring_window", 20)
        }
        self.drawdown_protector = DrawdownProtector(drawdown_config)
        self.correlation_manager = CorrelationManager()
        self.fee_optimizer = FeeOptimizer()
        
        logger.info("Integrated Risk Manager initialized with enhanced configuration")
    
    def assess_position_risk(
        self, 
        symbol: str, 
        quantity: int, 
        avg_price: float, 
        current_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        atr: float,
        sector: str = ""
    ) -> PositionRisk:
        """Assess risk for a specific position"""
        
        exposure = quantity * current_price
        risk_amount = abs(avg_price - stop_loss_price) * quantity
        reward_potential = abs(take_profit_price - avg_price) * quantity
        risk_reward_ratio = reward_potential / risk_amount if risk_amount > 0 else 0
        
        position_risk = PositionRisk(
            symbol=symbol,
            quantity=quantity,
            avg_price=avg_price,
            current_price=current_price,
            exposure=exposure,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_amount=risk_amount,
            reward_potential=reward_potential,
            risk_reward_ratio=risk_reward_ratio,
            atr=atr,
            volatility=atr / current_price if current_price > 0 else 0,
            sector=sector
        )
        
        return position_risk
    
    def validate_position(
        self, 
        position_risk: PositionRisk, 
        portfolio_value: float,
        sector_exposures: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Validate if a position meets risk criteria"""
        
        # Check single stock exposure
        position_exposure_pct = position_risk.exposure / portfolio_value
        if position_exposure_pct > self.max_single_stock_exposure:
            return False, f"Position exposure {position_exposure_pct:.2%} exceeds limit {self.max_single_stock_exposure:.2%}"
        
        # Check sector exposure if sector is provided
        if position_risk.sector:
            current_sector_exposure = sector_exposures.get(position_risk.sector, 0.0)
            new_sector_exposure = current_sector_exposure + position_risk.exposure
            sector_exposure_pct = new_sector_exposure / portfolio_value
            if sector_exposure_pct > self.max_sector_exposure:
                return False, f"Sector exposure {sector_exposure_pct:.2%} exceeds limit {self.max_sector_exposure:.2%}"
        
        # Check risk-reward ratio
        if position_risk.risk_reward_ratio < 1.5:
            return False, f"Risk-reward ratio {position_risk.risk_reward_ratio:.2f} below minimum 1.5"
        
        return True, "Position meets risk criteria"
    
    def calculate_dynamic_stop_loss(
        self, 
        current_price: float, 
        atr: float, 
        volatility: float,
        market_context: Optional[MarketContext] = None
    ) -> float:
        """Calculate dynamic stop loss based on market conditions"""
        
        if not self.dynamic_stop_loss_enabled:
            return current_price * 0.95  # Default 5% stop loss
        
        # Base stop loss on ATR
        atr_stop = current_price - (atr * self.atr_stop_multiplier)
        
        # Adjust based on market volatility
        if self.volatility_adjustment_enabled and market_context:
            volatility_factor = 1.0
            if market_context.volatility > 0.03:  # High volatility
                volatility_factor = 1.2
            elif market_context.volatility < 0.01:  # Low volatility
                volatility_factor = 0.8
                
            atr_stop = current_price - (atr * self.atr_stop_multiplier * volatility_factor)
        
        return atr_stop
    
    def get_portfolio_risk_metrics(
        self, 
        positions: List[PositionRisk], 
        portfolio_value: float,
        historical_returns: List[float] = None
    ) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        total_exposure = sum(pos.exposure for pos in positions)
        
        # Calculate sector exposures
        sector_exposures = {}
        for pos in positions:
            if pos.sector:
                sector_exposures[pos.sector] = sector_exposures.get(pos.sector, 0) + pos.exposure
        
        # Calculate portfolio volatility
        portfolio_volatility = 0.0
        if historical_returns:
            portfolio_volatility = np.std(historical_returns) if len(historical_returns) > 1 else 0.0
        
        # Calculate correlation risk
        correlation_risk = self.correlation_manager.calculate_portfolio_correlation_risk(positions)
        
        # Get current drawdown
        current_drawdown = self.drawdown_protector.get_current_drawdown()
        
        # Get fees paid
        fees_paid = self.fee_optimizer.get_total_fees()
        
        metrics = RiskMetrics(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            sector_exposures=sector_exposures,
            max_position_size=max(pos.exposure for pos in positions) if positions else 0,
            portfolio_volatility=portfolio_volatility,
            correlation_risk=correlation_risk,
            current_drawdown=current_drawdown,
            fees_paid=fees_paid
        )
        
        return metrics
    
    def should_reduce_exposure(self, risk_metrics: RiskMetrics) -> bool:
        """Determine if portfolio exposure should be reduced"""
        
        # Check portfolio drawdown
        if risk_metrics.current_drawdown > self.portfolio_drawdown_limit:
            logger.warning(f"Portfolio drawdown {risk_metrics.current_drawdown:.2%} exceeds limit {self.portfolio_drawdown_limit:.2%}")
            return True
        
        # Check if total exposure is too high
        exposure_pct = risk_metrics.total_exposure / risk_metrics.portfolio_value
        if exposure_pct > 0.8:  # 80% exposure limit
            logger.warning(f"Portfolio exposure {exposure_pct:.2%} exceeds 80% limit")
            return True
        
        return False
    
    def get_risk_adjusted_position_size(
        self, 
        signal_strength: float, 
        risk_per_trade: float, 
        stop_loss_distance: float,
        portfolio_value: float
    ) -> int:
        """Calculate risk-adjusted position size"""
        
        # Base position size calculation
        if stop_loss_distance <= 0:
            return 0
            
        position_size = int(risk_per_trade / stop_loss_distance)
        
        # Adjust based on signal strength
        signal_multiplier = min(1.0, max(0.5, signal_strength))  # Range 0.5-1.0
        position_size = int(position_size * signal_multiplier)
        
        # Ensure position doesn't exceed single stock exposure limit
        max_position_value = portfolio_value * self.max_single_stock_exposure
        if position_size * stop_loss_distance > max_position_value:
            position_size = int(max_position_value / stop_loss_distance)
        
        return position_size
