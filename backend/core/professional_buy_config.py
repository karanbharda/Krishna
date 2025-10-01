import json
import os

class ProfessionalBuyConfig:
    """Configuration for Professional Buy Logic"""
    
    @staticmethod
    def _load_live_config():
        """Load configuration from live_config.json"""
        try:
            # FIXED: Correct path to project root data directory
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'live_config.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load live_config.json: {e}")
            return {}
    
    @staticmethod
    def get_config():
        """Get professional buy configuration - production-ready settings with live_config.json integration"""
        # Load live configuration from frontend
        live_config = ProfessionalBuyConfig._load_live_config()
        
        # Extract values from live_config.json (with defaults)
        stop_loss_pct = live_config.get('stop_loss_pct', 0.05)  # Default 5%
        max_capital_per_trade = live_config.get('max_capital_per_trade', 0.25)  # Default 25%
        
        return {
            "min_buy_signals": 4,          # Minimum 4 signals (moderate-strict)
            "max_buy_signals": 5,          # Maximum 5 signals
            "min_buy_confidence": 0.45,    # 45% minimum confidence (moderate-strict)
            "min_weighted_buy_score": 0.06, # 6% minimum weighted score (moderate-strict)
            "entry_buffer_pct": 0.015,     # 1.5% entry buffer (moderate-strict)
            "buy_stop_loss_pct": stop_loss_pct,  # From live_config.json
            "max_capital_per_trade": max_capital_per_trade,  # From live_config.json
            "take_profit_ratio": 2.0,
            "partial_entry_threshold": 0.45,   # 45% for partial entry (moderate-strict)
            "full_entry_threshold": 0.70,      # 70% for full entry (moderate-strict)
            "downtrend_buy_multiplier": 0.7,
            "uptrend_buy_multiplier": 1.2,
            "enable_professional_buy_logic": True,
            "fallback_to_legacy_buy": False,
            # OPTIMIZED BUY LOGIC: Enhanced parameters
            "signal_sensitivity_multiplier": 1.1,  # Slightly reduced (moderate-strict)
            "early_entry_buffer_pct": 0.008,       # 0.8% early entry buffer
            "aggressive_entry_threshold": 0.80,    # 80% for aggressive entry (moderate-strict)
            "dynamic_signal_thresholds": True,
            "signal_strength_boost": 0.08,         # 8% boost (moderate-strict)
            "ml_signal_weight_boost": 0.12,        # 12% ML boost (moderate-strict)
            "ml_confidence_multiplier": 1.25,      # 1.25x ML multiplier (moderate-strict)
            "momentum_confirmation_window": 3,
            "momentum_strength_threshold": 0.025    # 2.5% momentum threshold (moderate-strict)
        }