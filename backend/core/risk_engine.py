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
                return json.load(f)
        else:
            # Default config if file doesn't exist
            default = {
                "mode": "live",
                "riskLevel": "MEDIUM",
                "stop_loss_pct": 0.05,
                "max_capital_per_trade": 0.20,
                "max_trade_limit": 100,
                "drawdown_limit_pct": 0.15,
                "created_at": datetime.now().isoformat()
            }
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default, f, indent=2)
            return default

    def set_trading_bot(self, trading_bot):
        """Set reference to the trading bot instance"""
        self.trading_bot = trading_bot

    def update_risk_profile(self, stop_loss_pct: float, capital_risk_pct: float, drawdown_limit_pct: float):
        """Update risk settings dynamically and save to live_config.json"""
        self.config["stop_loss_pct"] = stop_loss_pct / 100  # Convert % to decimal
        self.config["max_capital_per_trade"] = capital_risk_pct / 100
        self.config["drawdown_limit_pct"] = drawdown_limit_pct / 100
        self.config["updated_at"] = datetime.now().isoformat()
        
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