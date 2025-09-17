"""
Professional Sell Logic Configuration
Centralized configuration for professional-grade sell logic
"""

from typing import Dict

class ProfessionalSellConfig:
    """
    Professional sell logic configuration with institutional-grade parameters
    """
    
    @staticmethod
    def get_default_config() -> Dict:
        """Get default professional sell configuration"""
        return {
            # Signal Requirements (Professional Standards)
            "min_sell_signals": 2,                    # Minimum 2 signals required (vs 1 in legacy)
            "min_sell_confidence": 0.40,              # REDUCED from 0.65 to 0.40 to balance with buy thresholds
            "min_weighted_sell_score": 0.04,          # REDUCED from 0.40 to 0.04 to balance with buy thresholds
            
            # Stop-Loss Configuration (Dynamic & Trailing)
            "stop_loss_pct": 0.05,                    # Base 5% stop-loss
            "trailing_stop_pct": 0.03,                # 3% trailing stop
            "profit_protection_threshold": 0.05,      # Lock profits after 5% gain
            "volatility_stop_multiplier": 1.5,        # Adjust stops for volatility
            
            # Position Sizing (Partial vs Full Exits) - More granular thresholds
            "conservative_exit_threshold": 0.15,      # 15% confidence for conservative exit
            "partial_exit_threshold": 0.30,           # REDUCED from 0.50 to 0.30 for more granular exits
            "aggressive_exit_threshold": 0.50,        # 50% confidence for aggressive exit
            "full_exit_threshold": 0.70,              # REDUCED from 0.75 to 0.70 for balanced thresholds
            "emergency_exit_threshold": 0.90,         # 90% confidence for emergency exits
            
            # Market Context Filters (Less restrictive in uptrends)
            "uptrend_sell_multiplier": 1.1,           # REDUCED from 1.5 to 1.1 for less restriction in uptrends
            "downtrend_sell_multiplier": 0.9,         # INCREASED from 0.8 to 0.9 for less restriction in downtrends
            "sideways_sell_multiplier": 1.0,          # Normal threshold in sideways
            "strong_trend_threshold": 0.05,           # 5% for strong trend classification
            
            # Signal Weights (Professional Distribution)
            "technical_weight": 0.30,                 # 30% technical analysis
            "risk_management_weight": 0.25,           # 25% risk management
            "sentiment_weight": 0.20,                 # 20% sentiment analysis
            "ml_weight": 0.15,                        # 15% ML/AI signals
            "market_structure_weight": 0.10,          # 10% market structure
            
            # Risk Management (Professional Standards)
            "max_loss_threshold": 0.08,               # 8% maximum loss before forced exit
            "time_decay_threshold": 30,               # 30 days without profit
            "volatility_spike_threshold": 0.03,       # 3% daily volatility threshold
            "unrealized_loss_threshold": 0.02,        # 2% unrealized loss signal
            
            # Market Context Parameters
            "short_ma_period": 10,                    # Short-term moving average
            "long_ma_period": 50,                     # Long-term moving average
            "volatility_lookback": 20,                # Volatility calculation period
            "low_vol_threshold": 0.015,               # 1.5% low volatility
            "high_vol_threshold": 0.035,              # 3.5% high volatility
            
            # Professional Features
            "enable_professional_sell_logic": True,   # Enable professional logic
            "fallback_to_legacy_sell": False,         # Don't fallback to legacy
            "enable_profit_protection": True,         # Enable profit locking
            "enable_trailing_stops": True,            # Enable trailing stops
            "enable_market_context_filters": True,    # Enable trend filters
            "enable_position_sizing": True,           # Enable smart position sizing
            
            # Logging and Audit
            "enable_detailed_logging": True,          # Detailed decision logging
            "log_signal_breakdown": True,             # Log individual signals
            "audit_trail_enabled": True,              # Enable audit trail
            "conservative_exit_percentage": 0.25,     # 25% conservative exit
            "aggressive_exit_percentage": 1.0,        # 100% aggressive exit
        }
    
    @staticmethod
    def get_conservative_config() -> Dict:
        """Get conservative sell configuration for risk-averse trading"""
        config = ProfessionalSellConfig.get_default_config()
        config.update({
            "min_sell_signals": 3,                    # Require 3 signals
            "min_sell_confidence": 0.50,              # 50% confidence (less conservative than before)
            "min_weighted_sell_score": 0.10,          # 10% weighted score (less conservative than before)
            "stop_loss_pct": 0.04,                    # Tighter 4% stop-loss
            "trailing_stop_pct": 0.025,               # Tighter 2.5% trailing
            "uptrend_sell_multiplier": 1.2,           # Less high threshold in uptrends
            "partial_exit_threshold": 0.25,           # Lower threshold for partial exits
        })
        return config
    
    @staticmethod
    def get_aggressive_config() -> Dict:
        """Get aggressive sell configuration for active trading"""
        config = ProfessionalSellConfig.get_default_config()
        config.update({
            "min_sell_signals": 2,                    # Keep 2 signals
            "min_sell_confidence": 0.30,              # Lower 30% confidence
            "min_weighted_sell_score": 0.02,          # Lower 2% weighted score
            "stop_loss_pct": 0.06,                    # Wider 6% stop-loss
            "trailing_stop_pct": 0.04,                # Wider 4% trailing
            "uptrend_sell_multiplier": 1.0,           # Lower threshold in uptrends
            "full_exit_threshold": 0.60,              # Lower threshold for full exits
        })
        return config
    
    @staticmethod
    def get_scalping_config() -> Dict:
        """Get scalping configuration for short-term trading"""
        config = ProfessionalSellConfig.get_default_config()
        config.update({
            "min_sell_signals": 1,                    # Only 1 signal for speed
            "min_sell_confidence": 0.25,              # Lower confidence for speed
            "min_weighted_sell_score": 0.01,          # Lower weighted score
            "stop_loss_pct": 0.02,                    # Tight 2% stop-loss
            "trailing_stop_pct": 0.015,               # Tight 1.5% trailing
            "profit_protection_threshold": 0.02,      # Lock profits after 2% gain
            "partial_exit_threshold": 0.15,           # Quick partial exits
            "time_decay_threshold": 5,                # 5 days max holding
        })
        return config
    
    @staticmethod
    def get_swing_trading_config() -> Dict:
        """Get swing trading configuration for medium-term positions"""
        config = ProfessionalSellConfig.get_default_config()
        config.update({
            "min_sell_signals": 2,                    # 2 signals for swing
            "min_sell_confidence": 0.50,              # Higher confidence
            "stop_loss_pct": 0.08,                    # Wider 8% stop-loss
            "trailing_stop_pct": 0.05,                # Wider 5% trailing
            "profit_protection_threshold": 0.10,      # Lock profits after 10% gain
            "time_decay_threshold": 60,               # 60 days max holding
            "uptrend_sell_multiplier": 1.1,           # Lower threshold in uptrends
        })
        return config
    
    @staticmethod
    def validate_config(config: Dict) -> bool:
        """Validate professional sell configuration"""
        required_keys = [
            "min_sell_signals", "min_sell_confidence", "min_weighted_sell_score",
            "stop_loss_pct", "trailing_stop_pct", "profit_protection_threshold",
            "partial_exit_threshold", "full_exit_threshold"
        ]
        
        # Check required keys
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required config key: {key}")
                return False
        
        # Validate ranges
        if not (0.0 <= config["min_sell_confidence"] <= 1.0):
            print("❌ min_sell_confidence must be between 0.0 and 1.0")
            return False
        
        if not (0.0 <= config["stop_loss_pct"] <= 0.20):
            print("❌ stop_loss_pct must be between 0.0 and 0.20 (20%)")
            return False
        
        if config["partial_exit_threshold"] >= config["full_exit_threshold"]:
            print("❌ partial_exit_threshold must be less than full_exit_threshold")
            return False
        
        print("✅ Professional sell configuration validated successfully")
        return True
    
    @staticmethod
    def get_config_for_risk_level(risk_level: str) -> Dict:
        """Get configuration based on risk level"""
        risk_level = risk_level.upper()
        
        if risk_level == "LOW":
            return ProfessionalSellConfig.get_conservative_config()
        elif risk_level == "HIGH":
            return ProfessionalSellConfig.get_aggressive_config()
        elif risk_level == "SCALPING":
            return ProfessionalSellConfig.get_scalping_config()
        elif risk_level == "SWING":
            return ProfessionalSellConfig.get_swing_trading_config()
        else:  # MEDIUM or default
            return ProfessionalSellConfig.get_default_config()