import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DrawdownProtector:
    """Enhanced drawdown protection system"""
    
    def __init__(self, config: Dict):
        self.max_drawdown_limit = config.get("max_drawdown_limit", 0.15)  # 15% max drawdown
        self.daily_loss_limit = config.get("daily_loss_limit", 0.05)      # 5% daily loss limit
        self.recovery_threshold = config.get("recovery_threshold", 0.02)   # 2% recovery needed
        self.scaling_factor = config.get("scaling_factor", 0.5)           # Position scaling
        self.monitoring_window = config.get("monitoring_window", 20)      # Days to monitor
        
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.in_recovery_mode = False
        self.daily_loss = 0.0
        self.last_reset = datetime.now()
        self.value_history = []
        
    def update_portfolio_value(self, current_value: float, timestamp: datetime):
        """Update portfolio value and check drawdown conditions"""
        try:
            # Update peak value
            self.peak_value = max(self.peak_value, current_value)
            
            # Calculate current drawdown
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
            
            # Update value history
            self.value_history.append({
                "timestamp": timestamp,
                "value": current_value,
                "drawdown": self.current_drawdown
            })
            
            # Keep only recent history
            self.value_history = self.value_history[-self.monitoring_window:]
            
            # Reset daily loss at market open
            if (timestamp - self.last_reset).days >= 1:
                self.daily_loss = 0
                self.last_reset = timestamp
            
            # Calculate daily loss
            if len(self.value_history) >= 2:
                daily_return = (current_value - self.value_history[-2]["value"]) / self.value_history[-2]["value"]
                if daily_return < 0:
                    self.daily_loss = abs(daily_return)
            
            # Check recovery conditions
            if self.in_recovery_mode:
                recovery = (current_value - self.value_history[0]["value"]) / self.value_history[0]["value"]
                if recovery >= self.recovery_threshold:
                    self.in_recovery_mode = False
                    logger.info("Exiting recovery mode - Recovery threshold met")
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def get_position_adjustment(self) -> float:
        """Get position size adjustment based on drawdown conditions"""
        try:
            adjustment = 1.0
            
            # Apply drawdown-based scaling
            if self.current_drawdown > self.max_drawdown_limit:
                # Severe drawdown - significant reduction
                adjustment *= 0.25
                self.in_recovery_mode = True
                logger.warning(f"Severe drawdown protection activated: {self.current_drawdown:.1%}")
            
            elif self.current_drawdown > self.max_drawdown_limit * 0.7:
                # Approaching max drawdown - moderate reduction
                adjustment *= 0.5
                logger.warning(f"Moderate drawdown protection: {self.current_drawdown:.1%}")
            
            elif self.current_drawdown > self.max_drawdown_limit * 0.5:
                # Early drawdown - light reduction
                adjustment *= 0.75
                logger.info(f"Light drawdown protection: {self.current_drawdown:.1%}")
            
            # Apply daily loss limits
            if self.daily_loss > self.daily_loss_limit:
                adjustment *= 0.25  # Significant reduction on high daily losses
                logger.warning(f"Daily loss limit protection: {self.daily_loss:.1%}")
            
            # Recovery mode scaling
            if self.in_recovery_mode:
                adjustment *= self.scaling_factor
                logger.info("Recovery mode active - Reduced position sizing")
            
            return max(0.1, adjustment)  # Never go below 10% of original size
            
        except Exception as e:
            logger.error(f"Error calculating position adjustment: {e}")
            return 0.5  # Conservative default
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            "current_drawdown": self.current_drawdown,
            "peak_value": self.peak_value,
            "daily_loss": self.daily_loss,
            "in_recovery": self.in_recovery_mode,
            "adjustment_factor": self.get_position_adjustment()
        }
