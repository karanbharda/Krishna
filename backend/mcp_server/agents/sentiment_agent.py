#!/usr/bin/env python3
"""
MCP Sentiment Analysis Agent
============================

AI-powered sentiment analysis agent for the Model Context Protocol server
that analyzes market sentiment from news, social media, and other sources.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SentimentType(Enum):
    """Sentiment classifications"""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class SentimentAnalysis:
    """Structured sentiment analysis result"""
    symbol: str
    sentiment: SentimentType
    confidence: float
    compound_score: float
    positive_score: float
    negative_score: float
    neutral_score: float
    sources: List[str]
    key_themes: List[str]
    metadata: Dict[str, Any]


class SentimentAgent:
    """
    MCP Sentiment Analysis Agent
    Analyzes market sentiment from various sources to inform trading decisions
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = config.get("agent_id", "default_sentiment_agent")
        self.min_confidence = config.get("min_confidence", 0.7)
        self.sources = config.get("sources", ["news", "social_media"])

        # Initialize components
        self.is_initialized = False
        self.groq_engine = None

        logger.info(f"Sentiment Agent {self.agent_id} initialized")

    async def initialize(self):
        """Initialize the sentiment agent"""
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
                    logger.info("Groq engine connected to sentiment agent")
            except ImportError as e:
                logger.warning(f"Groq API integration not available: {e}")

            self.is_initialized = True
            logger.info(
                f"Sentiment Agent {self.agent_id} initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize Sentiment Agent {self.agent_id}: {e}")
            raise

    async def analyze_sentiment(
        self,
        symbol: str,
        sources_data: Optional[Dict[str, Any]] = None
    ) -> SentimentAnalysis:
        """
        Analyze market sentiment for a specific symbol

        Args:
            symbol: Stock symbol to analyze
            sources_data: Data from various sentiment sources

        Returns:
            SentimentAnalysis with sentiment classification and scores
        """
        if not self.is_initialized:
            raise RuntimeError("Sentiment agent not initialized")

        start_time = time.time()

        try:
            # Process sentiment sources
            processed_sources = self._process_sources(sources_data or {})

            # Aggregate sentiment scores
            compound_score, positive_score, negative_score, neutral_score = self._aggregate_scores(
                processed_sources)

            # Classify sentiment
            sentiment = self._classify_sentiment(compound_score)

            # Calculate confidence
            confidence = self._calculate_confidence(
                compound_score, processed_sources)

            # Extract key themes
            key_themes = self._extract_key_themes(processed_sources)

            analysis = SentimentAnalysis(
                symbol=symbol,
                sentiment=sentiment,
                confidence=confidence,
                compound_score=compound_score,
                positive_score=positive_score,
                negative_score=negative_score,
                neutral_score=neutral_score,
                sources=list(processed_sources.keys()),
                key_themes=key_themes,
                metadata={
                    "analysis_time": time.time() - start_time,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
            )

            logger.info(
                f"Sentiment analysis completed for {symbol}: {sentiment.value} (confidence: {confidence:.2f})")
            return analysis

        except Exception as e:
            logger.error(f"Error in analyze_sentiment for {symbol}: {e}")
            # Return neutral sentiment as fallback
            return SentimentAnalysis(
                symbol=symbol,
                sentiment=SentimentType.NEUTRAL,
                confidence=0.0,
                compound_score=0.0,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                sources=[],
                key_themes=[],
                metadata={
                    "analysis_time": time.time() - start_time,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            )

    def _process_sources(self, sources_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Process and normalize sentiment data from various sources"""
        processed = {}

        # News sentiment
        if "news" in sources_data:
            news_data = sources_data["news"]
            processed["news"] = {
                "compound": news_data.get("compound", 0.0),
                "positive": news_data.get("positive", 0.0),
                "negative": news_data.get("negative", 0.0),
                "neutral": news_data.get("neutral", 0.0),
                "weight": 0.4  # News carries significant weight
            }

        # Social media sentiment
        if "social_media" in sources_data:
            social_data = sources_data["social_media"]
            processed["social_media"] = {
                "compound": social_data.get("compound", 0.0),
                "positive": social_data.get("positive", 0.0),
                "negative": social_data.get("negative", 0.0),
                "neutral": social_data.get("neutral", 0.0),
                "weight": 0.3  # Social media has moderate weight
            }

        # Analyst ratings
        if "analyst_ratings" in sources_data:
            analyst_data = sources_data["analyst_ratings"]
            processed["analyst_ratings"] = {
                "compound": analyst_data.get("compound", 0.0),
                "positive": analyst_data.get("positive", 0.0),
                "negative": analyst_data.get("negative", 0.0),
                "neutral": analyst_data.get("neutral", 0.0),
                "weight": 0.3  # Analyst ratings carry significant weight
            }

        # If no sources provided, use default neutral sentiment
        if not processed:
            processed["default"] = {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "weight": 1.0
            }

        return processed

    def _aggregate_scores(self, processed_sources: Dict[str, Dict[str, float]]) -> tuple:
        """Aggregate sentiment scores from multiple sources"""
        total_weight = 0.0
        compound_sum = 0.0
        positive_sum = 0.0
        negative_sum = 0.0
        neutral_sum = 0.0

        for source_data in processed_sources.values():
            weight = source_data.get("weight", 0.0)
            compound_sum += source_data.get("compound", 0.0) * weight
            positive_sum += source_data.get("positive", 0.0) * weight
            negative_sum += source_data.get("negative", 0.0) * weight
            neutral_sum += source_data.get("neutral", 0.0) * weight
            total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            compound_score = compound_sum / total_weight
            positive_score = positive_sum / total_weight
            negative_score = negative_sum / total_weight
            neutral_score = neutral_sum / total_weight
        else:
            compound_score = 0.0
            positive_score = 0.0
            negative_score = 0.0
            neutral_score = 1.0

        return compound_score, positive_score, negative_score, neutral_score

    def _classify_sentiment(self, compound_score: float) -> SentimentType:
        """Classify sentiment based on compound score"""
        if compound_score >= 0.5:
            return SentimentType.VERY_BULLISH
        elif compound_score >= 0.1:
            return SentimentType.BULLISH
        elif compound_score <= -0.5:
            return SentimentType.VERY_BEARISH
        elif compound_score <= -0.1:
            return SentimentType.BEARISH
        else:
            return SentimentType.NEUTRAL

    def _calculate_confidence(self, compound_score: float, processed_sources: Dict[str, Dict[str, float]]) -> float:
        """Calculate confidence in sentiment analysis"""
        # Base confidence on magnitude of compound score
        magnitude_confidence = abs(compound_score)

        # Increase confidence with more sources
        source_count = len(processed_sources)
        # Max confidence with 3+ sources
        source_confidence = min(source_count / 3.0, 1.0)

        # Combine factors
        confidence = (magnitude_confidence * 0.7) + (source_confidence * 0.3)

        return min(confidence, 1.0)

    def _extract_key_themes(self, processed_sources: Dict[str, Dict[str, float]]) -> List[str]:
        """Extract key themes from sentiment sources (simplified)"""
        # In a real implementation, this would extract actual themes from text
        themes = []

        # Simple heuristic based on sentiment scores
        compound_avg = sum(src.get("compound", 0.0) for src in processed_sources.values(
        )) / len(processed_sources) if processed_sources else 0.0

        if compound_avg > 0.3:
            themes.extend(
                ["positive_outlook", "growth_potential", "strong_performance"])
        elif compound_avg < -0.3:
            themes.extend(["concerns", "risks", "underperformance"])
        else:
            themes.extend(
                ["mixed_signals", "neutral_outlook", "balanced_view"])

        return themes

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "initialized": self.is_initialized,
            "min_confidence": self.min_confidence,
            "sources": self.sources,
            "groq_available": self.groq_engine is not None
        }


# Agent availability flag
SENTIMENT_AGENT_AVAILABLE = True
