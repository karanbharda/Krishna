#!/usr/bin/env python3
"""
MCP Market Regime Detection Agent
=================================

AI-powered market regime detection agent for the Model Context Protocol server
that identifies different market conditions and recommends appropriate strategies.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    DEFENSIVE = "defensive"
    MOMENTUM = "momentum"
    UNCERTAIN = "uncertain"


@dataclass
class RegimeAnalysis:
    """Structured market regime analysis"""
    regime: MarketRegime
    confidence: float
    indicators: Dict[str, float]
    recommended_strategy: str
    risk_adjustment: str
    metadata: Dict[str, Any]


class MarketRegimeAgent:
    """
    MCP Market Regime Detection Agent
    Identifies market conditions and recommends appropriate trading strategies
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = config.get("agent_id", "default_regime_agent")
        self.lookback_period = config.get("lookback_period", 20)  # Days
        self.confidence_threshold = config.get("confidence_threshold", 0.7)

        # Initialize components
        self.is_initialized = False
        self.groq_engine = None

        logger.info(f"Market Regime Agent {self.agent_id} initialized")

    async def initialize(self):
        """Initialize the market regime agent"""
        try:
            # Import Groq engine if available
            try:
                # Use absolute import instead of relative import
                import sys
                import os
                # Add backend directory to path
                backend_dir = os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))
                if backend_dir not in sys.path:
                    sys.path.insert(0, backend_dir)
                from groq_api import GroqAPIEngine
                if "groq" in self.config:
                    groq_config = self.config["groq"]
                    self.groq_engine = GroqAPIEngine(groq_config)
                    logger.info("Groq engine connected to market regime agent")
            except ImportError as e:
                logger.warning(f"Groq API integration not available: {e}")

            self.is_initialized = True
            logger.info(
                f"Market Regime Agent {self.agent_id} initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize Market Regime Agent {self.agent_id}: {e}")
            raise

    async def detect_regime(
        self,
        market_data: Optional[Dict[str, Any]] = None,
        indicators: Optional[Dict[str, Any]] = None
    ) -> RegimeAnalysis:
        """
        Detect current market regime based on indicators and market data

        Args:
            market_data: Overall market data (indices, sectors, etc.)
            indicators: Technical and fundamental indicators

        Returns:
            RegimeAnalysis with regime classification and recommendations
        """
        if not self.is_initialized:
            raise RuntimeError("Market regime agent not initialized")

        start_time = time.time()

        try:
            # Extract and process market indicators
            processed_indicators = self._process_indicators(indicators or {})

            # Detect market regime
            regime, confidence = self._classify_regime(
                processed_indicators, market_data)

            # Generate strategy recommendations
            strategy = self._recommend_strategy(regime)
            risk_adjustment = self._recommend_risk_adjustment(regime)

            analysis = RegimeAnalysis(
                regime=regime,
                confidence=confidence,
                indicators=processed_indicators,
                recommended_strategy=strategy,
                risk_adjustment=risk_adjustment,
                metadata={
                    "analysis_time": time.time() - start_time,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
            )

            logger.info(
                f"Market regime detected: {regime.value} (confidence: {confidence:.2f})")
            return analysis

        except Exception as e:
            logger.error(f"Error in detect_regime: {e}")
            # Return uncertain regime as fallback
            return RegimeAnalysis(
                regime=MarketRegime.UNCERTAIN,
                confidence=0.0,
                indicators={},
                recommended_strategy="defensive",
                risk_adjustment="reduce_exposure",
                metadata={
                    "analysis_time": time.time() - start_time,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            )

    def _process_indicators(self, indicators: Dict[str, Any]) -> Dict[str, float]:
        """Process and normalize market indicators"""
        processed = {}

        # Price trend indicators
        processed["price_momentum"] = indicators.get("price_momentum", 0.0)
        processed["moving_average"] = indicators.get("moving_average", 0.0)
        processed["price_vs_ma"] = indicators.get("price_vs_ma", 0.0)

        # Volatility indicators
        processed["volatility"] = indicators.get(
            "volatility", 0.2)  # Default 20% annualized
        processed["volatility_trend"] = indicators.get("volatility_trend", 0.0)

        # Volume indicators
        processed["volume_trend"] = indicators.get("volume_trend", 0.0)
        processed["volume_ratio"] = indicators.get("volume_ratio", 1.0)

        # Breadth indicators
        processed["advance_decline"] = indicators.get("advance_decline", 0.5)
        processed["new_highs_lows"] = indicators.get("new_highs_lows", 0.0)

        # Sentiment indicators
        processed["sentiment_score"] = indicators.get("sentiment_score", 0.0)
        processed["put_call_ratio"] = indicators.get(
            "put_call_ratio", 0.7)  # Default

        # Economic indicators
        processed["vix"] = indicators.get("vix", 20.0)  # Default VIX level
        processed["yield_curve"] = indicators.get("yield_curve", 0.0)

        return processed

    def _classify_regime(self, indicators: Dict[str, float], market_data: Optional[Dict[str, Any]]) -> tuple:
        """Classify market regime based on indicators"""
        scores = {
            MarketRegime.BULL: 0.0,
            MarketRegime.BEAR: 0.0,
            MarketRegime.SIDEWAYS: 0.0,
            MarketRegime.VOLATILE: 0.0,
            MarketRegime.DEFENSIVE: 0.0,
            MarketRegime.MOMENTUM: 0.0,
            MarketRegime.UNCERTAIN: 0.0
        }

        # Price momentum analysis
        momentum = indicators.get("price_momentum", 0.0)
        if momentum > 0.1:
            scores[MarketRegime.BULL] += 0.3
            scores[MarketRegime.MOMENTUM] += 0.2
        elif momentum < -0.1:
            scores[MarketRegime.BEAR] += 0.3
        else:
            scores[MarketRegime.SIDEWAYS] += 0.2

        # Moving average position
        price_vs_ma = indicators.get("price_vs_ma", 0.0)
        if price_vs_ma > 0.05:
            scores[MarketRegime.BULL] += 0.2
        elif price_vs_ma < -0.05:
            scores[MarketRegime.BEAR] += 0.2
        else:
            scores[MarketRegime.SIDEWAYS] += 0.1

        # Volatility analysis
        volatility = indicators.get("volatility", 0.2)
        if volatility > 0.3:
            scores[MarketRegime.VOLATILE] += 0.3
        elif volatility < 0.15:
            scores[MarketRegime.SIDEWAYS] += 0.1

        # Volume analysis
        volume_trend = indicators.get("volume_trend", 0.0)
        if volume_trend > 0.1:
            scores[MarketRegime.MOMENTUM] += 0.2
        elif volume_trend < -0.1:
            scores[MarketRegime.DEFENSIVE] += 0.2

        # Breadth analysis
        advance_decline = indicators.get("advance_decline", 0.5)
        if advance_decline > 0.6:
            scores[MarketRegime.BULL] += 0.2
        elif advance_decline < 0.4:
            scores[MarketRegime.BEAR] += 0.2

        # Sentiment analysis
        sentiment = indicators.get("sentiment_score", 0.0)
        if sentiment > 0.5:
            scores[MarketRegime.BULL] += 0.1
        elif sentiment < -0.5:
            scores[MarketRegime.BEAR] += 0.1
        else:
            scores[MarketRegime.UNCERTAIN] += 0.1

        # VIX analysis
        vix = indicators.get("vix", 20.0)
        if vix > 30:
            scores[MarketRegime.VOLATILE] += 0.2
            scores[MarketRegime.DEFENSIVE] += 0.1
        elif vix < 15:
            scores[MarketRegime.BULL] += 0.1

        # Find regime with highest score
        if not scores:
            return (MarketRegime.UNCERTAIN, 0.0)

        best_regime = max(scores, key=scores.get)
        max_score = scores[best_regime]

        # Calculate confidence (normalize score)
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.0

        return (best_regime, confidence)

    def _recommend_strategy(self, regime: MarketRegime) -> str:
        """Recommend trading strategy based on market regime"""
        strategy_map = {
            MarketRegime.BULL: "growth_oriented",
            MarketRegime.BEAR: "defensive_value",
            MarketRegime.SIDEWAYS: "mean_reversion",
            MarketRegime.VOLATILE: "volatility_trading",
            MarketRegime.DEFENSIVE: "defensive_equity",
            MarketRegime.MOMENTUM: "momentum_following",
            MarketRegime.UNCERTAIN: "conservative_balanced"
        }
        return strategy_map.get(regime, "conservative_balanced")

    def _recommend_risk_adjustment(self, regime: MarketRegime) -> str:
        """Recommend risk adjustment based on market regime"""
        risk_map = {
            MarketRegime.BULL: "moderate_exposure",
            MarketRegime.BEAR: "reduce_exposure",
            MarketRegime.SIDEWAYS: "normal_exposure",
            MarketRegime.VOLATILE: "reduce_exposure",
            MarketRegime.DEFENSIVE: "reduce_exposure",
            MarketRegime.MOMENTUM: "moderate_exposure",
            MarketRegime.UNCERTAIN: "reduce_exposure"
        }
        return risk_map.get(regime, "normal_exposure")

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "initialized": self.is_initialized,
            "lookback_period": self.lookback_period,
            "confidence_threshold": self.confidence_threshold,
            "groq_available": self.groq_engine is not None
        }


# Agent availability flag
MARKET_REGIME_AGENT_AVAILABLE = True
