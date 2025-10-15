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
        stop_loss_pct = live_config.get('stop_loss_pct', 0.03)  # Default 3%
        target_profit_pct = live_config.get('target_profit_pct', 0.02)  # Default 2%
        max_capital_per_trade = live_config.get('max_capital_per_trade', 0.05)  # Default 5%
        
        return {
            "min_buy_signals": 3,          # CORRECTED: 3-5 categories as per project requirements
            "max_buy_signals": 5,          # Maximum 5 categories (all categories)
            "min_weighted_buy_score": 0.25, # INCREASED: 25% minimum weighted score (was 15%)
            "entry_buffer_pct": 0.015,     # 1.5% entry buffer (moderate)
            "buy_stop_loss_pct": stop_loss_pct,  # From live_config.json
            "buy_target_profit_pct": target_profit_pct,  # From live_config.json
            "take_profit_ratio": target_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 2.0,
            "partial_entry_threshold": 0.55,   # CORRECTED: 55% as per project requirements
            "full_entry_threshold": live_config.get('full_entry_threshold', 0.85),  # CORRECTED: 85% as per project requirements
            "signal_sensitivity_multiplier": 0.9,  # CORRECTED: 0.9 as per project requirements
            "early_entry_buffer_pct": 0.01,       # 1% early entry buffer (more conservative)
            "aggressive_entry_threshold": 0.90,    # CORRECTED: 90% as per project requirements
            "dynamic_signal_thresholds": True,
            "signal_strength_boost": 0.03,         # CORRECTED: 3% as per project requirements
            "ml_signal_weight_boost": 0.05,        # CORRECTED: 5% as per project requirements
            "ml_confidence_multiplier": 1.10,      # CORRECTED: 1.10 as per project requirements
            "momentum_confirmation_window": 5,
            "momentum_strength_threshold": 0.035,    # CORRECTED: 3.5% as per project requirements
            "min_buy_confidence": 0.70,            # INCREASED: 70% minimum confidence (was 60%)
            "enable_professional_buy_logic": True,  # Enable professional logic
            "fallback_to_legacy_buy": False,        # Don't fallback to legacy
            "max_capital_per_trade": max_capital_per_trade,  # From live_config.json
        }