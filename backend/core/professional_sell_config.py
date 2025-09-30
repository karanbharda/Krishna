"""
Professional Sell Logic Configuration
Centralized configuration for professional-grade sell logic
"""

import json
import os
from typing import Dict

class ProfessionalSellConfig:
    """
    Professional sell logic configuration with institutional parameters
    """
    
    @staticmethod
    def _load_live_config():
        """Load configuration from live_config.json"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'live_config.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load live_config.json: {e}")
            return {}
    
    @staticmethod
    def get_config() -> Dict:
        """Get professional sell configuration - production-ready settings with live_config.json integration"""
        # Load live configuration from frontend
        live_config = ProfessionalSellConfig._load_live_config()
        
        # Extract values from live_config.json (with defaults)
        stop_loss_pct = live_config.get('stop_loss_pct', 0.05)  # Default 5%
        max_capital_per_trade = live_config.get('max_capital_per_trade', 0.25)  # Default 25%
        drawdown_limit_pct = live_config.get('drawdown_limit_pct', 0.15)  # Default 15%
        
        return {
            # Signal Requirements (Professional Standards)
            "min_sell_signals": 2,                    # Minimum 2 signals required
            "min_sell_confidence": 0.40,              # 40% minimum confidence (balanced with buy)
            "min_weighted_sell_score": 0.04,          # 4% minimum weighted score (balanced with buy)
            
            # Stop-Loss Configuration (Dynamic & Trailing) - Integrated with live_config.json
            "stop_loss_pct": stop_loss_pct,           # From live_config.json
            "trailing_stop_pct": stop_loss_pct * 0.6, # 60% of stop-loss for trailing
            "max_capital_per_trade": max_capital_per_trade,  # From live_config.json
            "drawdown_limit_pct": drawdown_limit_pct, # From live_config.json
            "profit_protection_threshold": 0.05,      # Lock profits after 5% gain
            "volatility_stop_multiplier": 1.5,        # Adjust stops for volatility
            
            # Position Sizing (Partial vs Full Exits) - More granular thresholds
            "conservative_exit_threshold": 0.15,      # 15% confidence for conservative exit
            "partial_exit_threshold": 0.30,           # 30% confidence for partial exit
            "aggressive_exit_threshold": 0.50,        # 50% confidence for aggressive exit
            "full_exit_threshold": 0.70,              # 70% confidence for full exit
            "emergency_exit_threshold": 0.90,         # 90% confidence for emergency exits
            
            # Market Context Filters (Less restrictive in uptrends)
            "uptrend_sell_multiplier": 1.1,           # Less restriction in uptrends
            "downtrend_sell_multiplier": 0.9,         # Less restriction in downtrends
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
    def validate_config(config: Dict) -> bool:
        """Validate professional sell configuration"""
        required_keys = [
            "min_sell_signals", "min_sell_confidence", "min_weighted_sell_score",
            "stop_loss_pct", "trailing_stop_pct", "profit_protection_threshold",
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