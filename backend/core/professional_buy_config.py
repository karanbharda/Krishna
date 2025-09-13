class ProfessionalBuyConfig:
    """Configuration for Professional Buy Logic"""
    
    @staticmethod
    def get_default_config():
        """Get default professional buy configuration"""
        return {
            "min_buy_signals": 2,          # Minimum 2 signals
            "max_buy_signals": 4,          # Maximum 4 signals
            "min_buy_confidence": 0.40,    # REDUCED from 0.50 to 0.40
            "min_weighted_buy_score": 0.08, # REDUCED from 0.12 to 0.08
            "entry_buffer_pct": 0.01,
            "buy_stop_loss_pct": 0.05,
            "take_profit_ratio": 2.0,
            "partial_entry_threshold": 0.40, # Reduced from 0.50
            "full_entry_threshold": 0.65,    # Reduced from 0.75
            "downtrend_buy_multiplier": 0.7,
            "uptrend_buy_multiplier": 1.2,
            "enable_professional_buy_logic": True,
            "fallback_to_legacy_buy": True
        }
    
    @staticmethod
    def get_conservative_config():
        """Get conservative professional buy configuration"""
        return {
            "min_buy_signals": 2,          # Minimum 2 signals
            "max_buy_signals": 4,          # Maximum 4 signals
            "min_buy_confidence": 0.40,    # REDUCED from 0.50 to 0.40
            "min_weighted_buy_score": 0.08, # REDUCED from 0.20 to 0.08
            "entry_buffer_pct": 0.015,
            "buy_stop_loss_pct": 0.04,
            "take_profit_ratio": 2.5,
            "partial_entry_threshold": 0.35,  # Reduced from 0.40
            "full_entry_threshold": 0.70,     # Reduced from 0.80
            "downtrend_buy_multiplier": 0.5,
            "uptrend_buy_multiplier": 1.1,
            "enable_professional_buy_logic": True,
            "fallback_to_legacy_buy": False   # Changed to False to prevent fallback
        }