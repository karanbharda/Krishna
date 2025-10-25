import json
import os
import logging
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DynamicRiskEngine:
    def __init__(self, config_path=None):
        # FIXED: Use project root data directory
        if config_path is None:
            backend_dir = Path(__file__).resolve().parents[1]
            project_root = backend_dir.parent
            config_path = str(project_root / 'data' / 'live_config.json')
        self.config_path = config_path
        self.config = self._load_config()
        self.trading_bot = None  # Reference to the trading bot instance

    def _load_config(self) -> Dict[str, Any]:
        """Load risk config from live_config.json"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                # Standardize percentage values to decimal format
                self._standardize_percentages(config)
                return config
        else:
            # Default config if file doesn't exist
            default = {
                "mode": "paper",
                "riskLevel": "MEDIUM",
                "stop_loss_pct": 0.05,          # Standardized as decimal
                "max_capital_per_trade": 0.20,   # Standardized as decimal
                "max_trade_limit": 100,
                "drawdown_limit_pct": 0.15,      # Standardized as decimal
                "created_at": datetime.now().isoformat()
            }
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default, f, indent=2)
            return default

    def _standardize_percentages(self, config: Dict[str, Any]):
        """Standardize percentage values to decimal format"""
        percentage_keys = ["stop_loss_pct", "max_capital_per_trade", "drawdown_limit_pct"]
        
        for key in percentage_keys:
            if key in config:
                value = config[key]
                # If value is greater than 1, it's likely in percentage format
                if isinstance(value, (int, float)) and value > 1:
                    # Convert from percentage to decimal (e.g., 5 -> 0.05)
                    config[key] = value / 100
                # Ensure value is within reasonable bounds
                config[key] = max(0.01, min(config[key], 0.5))  # 1% to 50%

    def set_trading_bot(self, trading_bot):
        """Set reference to the trading bot instance"""
        self.trading_bot = trading_bot

    def update_risk_profile(self, stop_loss_pct: float, capital_risk_pct: float, drawdown_limit_pct: float):
        """Update risk settings dynamically and save to live_config.json"""
        # Ensure all values are in decimal format (not percentages)
        self.config["stop_loss_pct"] = stop_loss_pct / 100 if stop_loss_pct > 1 else stop_loss_pct
        self.config["max_capital_per_trade"] = capital_risk_pct / 100 if capital_risk_pct > 1 else capital_risk_pct
        self.config["drawdown_limit_pct"] = drawdown_limit_pct / 100 if drawdown_limit_pct > 1 else drawdown_limit_pct
        self.config["updated_at"] = datetime.now().isoformat()
        
        # Validate bounds
        self.config["stop_loss_pct"] = max(0.01, min(self.config["stop_loss_pct"], 0.20))  # 1% to 20%
        self.config["max_capital_per_trade"] = max(0.05, min(self.config["max_capital_per_trade"], 0.50))  # 5% to 50%
        self.config["drawdown_limit_pct"] = max(0.05, min(self.config["drawdown_limit_pct"], 0.30))  # 5% to 30%
        
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Updated live_config.json with new risk settings")
        
        # Notify the trading bot to refresh its professional integrations
        if self.trading_bot:
            try:
                self.trading_bot.refresh_professional_integrations()
                logger.info("Notified trading bot to refresh professional integrations")
            except Exception as e:
                logger.error(f"Failed to notify trading bot of config update: {e}")

    def get_risk_settings(self) -> Dict[str, float]:
        """Get current risk settings"""
        # Reload config to get latest settings
        self.config = self._load_config()
        return {
            "stop_loss_pct": self.config.get("stop_loss_pct", 0.05),
            "capital_risk_pct": self.config.get("max_capital_per_trade", 0.20),
            "drawdown_limit_pct": self.config.get("drawdown_limit_pct", 0.15),
            "max_trade_limit": self.config.get("max_trade_limit", 100)
        }

    def apply_risk_to_position(self, position_value: float) -> Dict[str, float]:
        """Apply current risk settings to calculate limits"""
        settings = self.get_risk_settings()
        stop_loss_amount = position_value * settings["stop_loss_pct"]
        capital_at_risk = position_value * settings["capital_risk_pct"]
        max_drawdown = position_value * settings["drawdown_limit_pct"]
        
        return {
            "stop_loss_amount": stop_loss_amount,
            "capital_at_risk": capital_at_risk,
            "max_drawdown": max_drawdown,
            "max_trade_limit": settings["max_trade_limit"]
        }

    def get_risk_level(self) -> str:
        """Get current risk level"""
        return self.config.get("riskLevel", "MEDIUM")

# Global instance
risk_engine = DynamicRiskEngine()