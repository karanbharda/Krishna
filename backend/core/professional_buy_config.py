class ProfessionalBuyConfig:
    """Configuration for Professional Buy Logic"""
    
    @staticmethod
    def get_default_config():
        """Get default professional buy configuration"""
        return {
            "min_buy_signals": 4,          # Minimum 2 signals
            "max_buy_signals": 5,          # Maximum 4 signals
            "min_buy_confidence": 0.40,    # REDUCED from 0.50 to 0.40
            "min_weighted_buy_score": 0.04, # REDUCED from 0.12 to 0.04 (FIXED: Lowered threshold to allow more trades)
            "entry_buffer_pct": 0.01,
            "buy_stop_loss_pct": 0.05,
            "take_profit_ratio": 2.0,
            "partial_entry_threshold": 0.40, # Reduced from 0.50
            "full_entry_threshold": 0.65,    # Reduced from 0.75
            "downtrend_buy_multiplier": 0.7,
            "uptrend_buy_multiplier": 1.2,
            "enable_professional_buy_logic": True,
            "fallback_to_legacy_buy": False,
            # OPTIMIZED BUY LOGIC: Enhanced parameters
            "signal_sensitivity_multiplier": 1.2,
            "early_entry_buffer_pct": 0.005,
            "aggressive_entry_threshold": 0.75,
            "dynamic_signal_thresholds": True,
            "signal_strength_boost": 0.1,
            "ml_signal_weight_boost": 0.15,
            "ml_confidence_multiplier": 1.3,
            "momentum_confirmation_window": 3,
            "momentum_strength_threshold": 0.02
        }
    
    @staticmethod
    def get_conservative_config():
        """Get conservative professional buy configuration"""
        return {
            "min_buy_signals": 4,          # Minimum 2 signals
            "max_buy_signals": 5,         # Maximum 4 signals
            "min_buy_confidence": 0.40,    # REDUCED from 0.50 to 0.40
            "min_weighted_buy_score": 0.04, # REDUCED from 0.20 to 0.04 (FIXED: Lowered threshold to allow more trades)
            "entry_buffer_pct": 0.015,
            "buy_stop_loss_pct": 0.04,
            "take_profit_ratio": 2.5,
            "partial_entry_threshold": 0.35,  # Reduced from 0.40
            "full_entry_threshold": 0.70,     # Reduced from 0.80
            "downtrend_buy_multiplier": 0.5,
            "uptrend_buy_multiplier": 1.1,
            "enable_professional_buy_logic": True,
            "fallback_to_legacy_buy": False,   # Changed to False to prevent fallback
            # OPTIMIZED BUY LOGIC: Enhanced parameters (conservative settings)
            "signal_sensitivity_multiplier": 1.0,  # No boost for conservative approach
            "early_entry_buffer_pct": 0.01,        # Later entries for conservative approach
            "aggressive_entry_threshold": 0.85,    # Higher threshold for aggressive entries
            "dynamic_signal_thresholds": True,
            "signal_strength_boost": 0.05,         # Lower boost for conservative approach
            "ml_signal_weight_boost": 0.1,         # Lower ML weight boost
            "ml_confidence_multiplier": 1.1,       # Lower ML confidence boost
            "momentum_confirmation_window": 5,     # Longer confirmation window
            "momentum_strength_threshold": 0.03    # Higher momentum threshold
        }