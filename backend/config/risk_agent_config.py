class RiskAgentConfig:
    """Configuration for Risk Agent with Enhanced Adaptive Risk Management"""
    
    @staticmethod
    def get_default_config():
        """Get default risk agent configuration"""
        return {
            "agent_id": "risk_agent",
            "max_portfolio_var": 0.05,          # 5% daily VaR
            "max_position_size": 0.25,          # 25% max per position
            "max_sector_concentration": 0.40,   # 40% per sector
            "min_liquidity_score": 0.3,
            "lookback_period": 252,             # 1 year
            "confidence_level": 0.95,
            # ENHANCED RISK MANAGEMENT: Dynamic risk parameters
            "volatility_threshold_low": 0.02,
            "volatility_threshold_high": 0.04,
            "volatility_adjustment_factor": 0.5,
            "enable_adaptive_risk_management": True
        }
    
    @staticmethod
    def get_conservative_config():
        """Get conservative risk agent configuration"""
        return {
            "agent_id": "risk_agent_conservative",
            "max_portfolio_var": 0.03,          # 3% daily VaR (more conservative)
            "max_position_size": 0.15,          # 15% max per position (more conservative)
            "max_sector_concentration": 0.30,   # 30% per sector (more conservative)
            "min_liquidity_score": 0.5,         # Higher liquidity requirement
            "lookback_period": 252,             # 1 year
            "confidence_level": 0.99,           # 99% confidence (more conservative)
            # ENHANCED RISK MANAGEMENT: Dynamic risk parameters
            "volatility_threshold_low": 0.015,  # Lower threshold for conservative approach
            "volatility_threshold_high": 0.03,  # Lower threshold for conservative approach
            "volatility_adjustment_factor": 0.7, # Higher adjustment factor for conservative approach
            "enable_adaptive_risk_management": True
        }