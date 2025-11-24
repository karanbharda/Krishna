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
                # Maximum allowed volatility (30%)
                "max_volatility": 0.30,
                "sentiment_multiplier": 1.0,     # Sentiment-based position sizing multiplier
                # Maximum position size (10% of capital)
                "max_position_size": 0.1,
                "volatility_filter_enabled": True,  # Enable volatility filtering
                # Enable sentiment conflict filtering
                "sentiment_conflict_filter_enabled": True,
                # Sentiment sensitivity thresholds
                # Threshold for sentiment conflict detection
                "sentiment_sensitivity_threshold": 0.5,
                "sentiment_override_threshold": 0.7,     # Threshold for sentiment override
                # Volatility filter settings
                "volatility_filter_threshold": 0.30,     # 30% volatility threshold
                "high_volatility_threshold": 0.50,       # 50% high volatility threshold
                # Factor to reduce position size for high volatility
                "volatility_reduction_factor": 0.5,
                "created_at": datetime.now().isoformat()
            }
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default, f, indent=2)
            return default

    def _standardize_percentages(self, config: Dict[str, Any]):
        """Standardize percentage values to decimal format"""
        percentage_keys = ["stop_loss_pct", "max_capital_per_trade",
                           "drawdown_limit_pct", "max_volatility", "max_position_size"]

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

    def update_risk_profile(self, stop_loss_pct: float, capital_risk_pct: float, drawdown_limit_pct: float,
                            max_volatility: float = None, sentiment_multiplier: float = None,
                            max_position_size: float = None):
        """Update risk settings dynamically and save to live_config.json"""
        # Ensure all values are in decimal format (not percentages)
        self.config["stop_loss_pct"] = stop_loss_pct / \
            100 if stop_loss_pct > 1 else stop_loss_pct
        self.config["max_capital_per_trade"] = capital_risk_pct / \
            100 if capital_risk_pct > 1 else capital_risk_pct
        self.config["drawdown_limit_pct"] = drawdown_limit_pct / \
            100 if drawdown_limit_pct > 1 else drawdown_limit_pct

        # Update volatility and sentiment settings if provided
        if max_volatility is not None:
            self.config["max_volatility"] = max_volatility / \
                100 if max_volatility > 1 else max_volatility
        if sentiment_multiplier is not None:
            self.config["sentiment_multiplier"] = sentiment_multiplier
        if max_position_size is not None:
            self.config["max_position_size"] = max_position_size / \
                100 if max_position_size > 1 else max_position_size

        self.config["updated_at"] = datetime.now().isoformat()

        # Validate bounds
        self.config["stop_loss_pct"] = max(
            0.01, min(self.config["stop_loss_pct"], 0.20))  # 1% to 20%
        self.config["max_capital_per_trade"] = max(
            0.05, min(self.config["max_capital_per_trade"], 0.50))  # 5% to 50%
        self.config["drawdown_limit_pct"] = max(
            0.05, min(self.config["drawdown_limit_pct"], 0.30))  # 5% to 30%
        self.config["max_volatility"] = max(
            0.05, min(self.config.get("max_volatility", 0.30), 0.50))  # 5% to 50%
        self.config["sentiment_multiplier"] = max(
            # 0.1x to 3x
            0.1, min(self.config.get("sentiment_multiplier", 1.0), 3.0))
        self.config["max_position_size"] = max(
            0.01, min(self.config.get("max_position_size", 0.1), 0.50))  # 1% to 50%

        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Updated live_config.json with new risk settings")

        # Notify the trading bot to refresh its professional integrations
        if self.trading_bot:
            try:
                self.trading_bot.refresh_professional_integrations()
                logger.info(
                    "Notified trading bot to refresh professional integrations")
            except Exception as e:
                logger.error(
                    f"Failed to notify trading bot of config update: {e}")

    def get_risk_settings(self) -> Dict[str, float]:
        """Get current risk settings"""
        # Reload config to get latest settings
        self.config = self._load_config()
        return {
            "stop_loss_pct": self.config.get("stop_loss_pct", 0.05),
            "capital_risk_pct": self.config.get("max_capital_per_trade", 0.20),
            "drawdown_limit_pct": self.config.get("drawdown_limit_pct", 0.15),
            "max_trade_limit": self.config.get("max_trade_limit", 100),
            "max_volatility": self.config.get("max_volatility", 0.30),
            "sentiment_multiplier": self.config.get("sentiment_multiplier", 1.0),
            "max_position_size": self.config.get("max_position_size", 0.1),
            "volatility_filter_enabled": self.config.get("volatility_filter_enabled", True),
            "sentiment_conflict_filter_enabled": self.config.get("sentiment_conflict_filter_enabled", True),
            "sentiment_sensitivity_threshold": self.config.get("sentiment_sensitivity_threshold", 0.5),
            "sentiment_override_threshold": self.config.get("sentiment_override_threshold", 0.7),
            "volatility_filter_threshold": self.config.get("volatility_filter_threshold", 0.30),
            "high_volatility_threshold": self.config.get("high_volatility_threshold", 0.50),
            "volatility_reduction_factor": self.config.get("volatility_reduction_factor", 0.5)
        }

    def apply_risk_to_position(self, position_value: float, volatility: float = None, sentiment_score: float = None) -> Dict[str, float]:
        """Apply current risk settings to calculate limits with volatility and sentiment adjustments"""
        settings = self.get_risk_settings()
        stop_loss_amount = position_value * settings["stop_loss_pct"]
        capital_at_risk = position_value * settings["capital_risk_pct"]
        max_drawdown = position_value * settings["drawdown_limit_pct"]

        # Apply volatility filter if provided
        volatility_multiplier = 1.0
        if volatility is not None:
            max_allowed_volatility = settings["max_volatility"]
            if volatility > max_allowed_volatility:
                # Reduce position size proportionally to volatility excess
                volatility_multiplier = max_allowed_volatility / volatility
            # Enhanced volatility filtering with configurable thresholds
            elif volatility > settings["high_volatility_threshold"]:
                # Apply additional reduction for high volatility
                volatility_multiplier *= settings["volatility_reduction_factor"]

        # Apply sentiment multiplier if provided
        sentiment_multiplier = settings["sentiment_multiplier"]
        if sentiment_score is not None:
            # Adjust multiplier based on sentiment (positive sentiment increases position, negative decreases)
            sentiment_multiplier = 1.0 + \
                (sentiment_score * 0.5)  # -0.5 to +0.5 adjustment
            # Cap between 0.5x and 2.0x
            sentiment_multiplier = max(0.5, min(sentiment_multiplier, 2.0))

        # Apply multipliers to risk amounts
        adjusted_stop_loss = stop_loss_amount * \
            volatility_multiplier * sentiment_multiplier
        adjusted_capital_risk = capital_at_risk * \
            volatility_multiplier * sentiment_multiplier
        adjusted_max_drawdown = max_drawdown * \
            volatility_multiplier * sentiment_multiplier

        return {
            "stop_loss_amount": adjusted_stop_loss,
            "capital_at_risk": adjusted_capital_risk,
            "max_drawdown": adjusted_max_drawdown,
            "max_trade_limit": settings["max_trade_limit"],
            "max_position_size": settings["max_position_size"],
            "volatility_multiplier": volatility_multiplier,
            "sentiment_multiplier": sentiment_multiplier,
            "volatility_filter_enabled": settings["volatility_filter_enabled"],
            "sentiment_conflict_filter_enabled": settings["sentiment_conflict_filter_enabled"],
            "sentiment_sensitivity_threshold": settings["sentiment_sensitivity_threshold"],
            "sentiment_override_threshold": settings["sentiment_override_threshold"],
            "volatility_filter_threshold": settings["volatility_filter_threshold"],
            "high_volatility_threshold": settings["high_volatility_threshold"]
        }

    def get_risk_level(self) -> str:
        """Get current risk level"""
        return self.config.get("riskLevel", "MEDIUM")

    def is_volatility_acceptable(self, volatility: float) -> bool:
        """Check if the given volatility is within acceptable limits"""
        settings = self.get_risk_settings()
        return volatility <= settings["max_volatility"]

    def get_sentiment_adjusted_position_size(self, base_size: float, sentiment_score: float) -> float:
        """Get position size adjusted by sentiment score"""
        settings = self.get_risk_settings()
        if sentiment_score is not None:
            # Adjust multiplier based on sentiment
            sentiment_multiplier = 1.0 + \
                (sentiment_score * 0.5)  # -0.5 to +0.5 adjustment
            # Cap between 0.5x and 2.0x
            sentiment_multiplier = max(0.5, min(sentiment_multiplier, 2.0))
            return base_size * sentiment_multiplier
        return base_size * settings["sentiment_multiplier"]

    def get_news_confidence_adjusted_position_size(self, base_size: float, news_confidence: float) -> float:
        """Get position size adjusted by news confidence score"""
        if news_confidence is not None:
            # News confidence multiplier (0.5 to 1.5)
            confidence_multiplier = 0.5 + (news_confidence * 1.0)
            # Cap between 0.5x and 1.5x
            confidence_multiplier = max(0.5, min(confidence_multiplier, 1.5))
            return base_size * confidence_multiplier
        return base_size

    def should_override_with_sentiment(self, confidence: float, sentiment_score: float) -> bool:
        """Check if sentiment should override low confidence trading signal"""
        settings = self.get_risk_settings()
        return (confidence < 0.3 and
                abs(sentiment_score) > settings["sentiment_override_threshold"])

    def is_position_size_acceptable(self, position_value: float, total_capital: float) -> bool:
        """Check if the proposed position size is within acceptable limits"""
        settings = self.get_risk_settings()
        position_ratio = position_value / total_capital if total_capital > 0 else 0
        return position_ratio <= settings["max_position_size"]

    def should_filter_volatility(self, volatility: float) -> bool:
        """Check if volatility filtering is enabled and if the volatility should be filtered"""
        settings = self.get_risk_settings()
        if not settings["volatility_filter_enabled"]:
            return False
        return not self.is_volatility_acceptable(volatility)

    def should_filter_sentiment_conflict(self, trading_signal: Dict[str, Any], sentiment_data: Dict[str, Any]) -> bool:
        """Check if sentiment conflict filtering is enabled and if there's a conflict"""
        settings = self.get_risk_settings()
        if not settings["sentiment_conflict_filter_enabled"]:
            return False

        # Check if there's a sentiment conflict
        signal_action = trading_signal.get("action", "").upper()
        sentiment_score = sentiment_data.get(
            "confidence", 0) if sentiment_data else 0

        # Use configurable threshold
        conflict_threshold = settings["sentiment_sensitivity_threshold"]

        # If signal is BUY but sentiment is strongly negative, or vice versa
        if signal_action == "BUY" and sentiment_score < -conflict_threshold:
            return True
        elif signal_action == "SELL" and sentiment_score > conflict_threshold:
            return True

        return False


# Global instance
risk_engine = DynamicRiskEngine()
