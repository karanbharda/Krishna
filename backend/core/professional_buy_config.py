class ProfessionalBuyConfig:
    """Configuration for Professional Buy Logic"""
    
    @staticmethod
    def get_default_config():
        """Get default professional buy configuration"""
        return {
            "min_buy_signals": 3,          # Increased from 2
            "min_buy_confidence": 0.65,    # Increased from 0.50
            "min_weighted_buy_score": 0.40, # Increased from 0.30
            "entry_buffer_pct": 0.01,
            "buy_stop_loss_pct": 0.05,
            "take_profit_ratio": 2.0,
            "partial_entry_threshold": 0.50,
            "full_entry_threshold": 0.75,
            "downtrend_buy_multiplier": 0.7,
            "uptrend_buy_multiplier": 1.2,
            "enable_professional_buy_logic": True,
            "fallback_to_legacy_buy": True
        }
    
    @staticmethod
    def get_conservative_config():
        """Get conservative professional buy configuration"""
        return {
            "min_buy_signals": 4,          # Increased from 3
            "min_buy_confidence": 0.70,    # Increased from 0.65
            "min_weighted_buy_score": 0.50, # Increased from 0.40
            "entry_buffer_pct": 0.015,
            "buy_stop_loss_pct": 0.04,
            "take_profit_ratio": 2.5,
            "partial_entry_threshold": 0.40,  # Increased from 0.30
            "full_entry_threshold": 0.80,     # Increased from 0.70
            "downtrend_buy_multiplier": 0.5,
            "uptrend_buy_multiplier": 1.1,
            "enable_professional_buy_logic": True,
            "fallback_to_legacy_buy": False   # Changed to False to prevent fallback
        }
    
    @staticmethod
    def get_aggressive_config():
        """Get aggressive professional buy configuration"""
        return {
            "min_buy_signals": 1,
            "min_buy_confidence": 0.40,
            "min_weighted_buy_score": 0.25,
            "entry_buffer_pct": 0.005,
            "buy_stop_loss_pct": 0.06,
            "take_profit_ratio": 1.8,
            "partial_entry_threshold": 0.50,
            "full_entry_threshold": 0.60,
            "downtrend_buy_multiplier": 0.8,
            "uptrend_buy_multiplier": 1.3,
            "enable_professional_buy_logic": True,
            "fallback_to_legacy_buy": True
        }