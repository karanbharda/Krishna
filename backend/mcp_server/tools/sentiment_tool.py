#!/usr/bin/env python3
"""
MCP Sentiment Tool
==================

News sentiment analysis tool for the Model Context Protocol server
with integration for Indian stock market news sources.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from ..mcp_trading_server import MCPToolResult

# Import sentiment analysis components
try:
    from backend.testindia import Stock
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    logging.warning("Sentiment analysis components not available")

logger = logging.getLogger(__name__)


class SentimentTool:
    """
    MCP Sentiment Tool
    Analyzes news sentiment for stock symbols with Indian market focus
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "sentiment_tool")
        self.sentiment_sources = config.get(
            "sentiment_sources", ["news", "social", "market"])

        # Initialize sentiment analyzer
        if SENTIMENT_AVAILABLE:
            self.stock_analyzer = Stock()
        else:
            self.stock_analyzer = None

        logger.info(
            f"Sentiment Tool {self.tool_id} initialized with sources: {self.sentiment_sources}")

    async def analyze_sentiment(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Analyze sentiment for specified symbols

        Args:
            arguments: Tool arguments containing symbols and parameters
            session_id: Session identifier

        Returns:
            MCPToolResult with sentiment analysis
        """
        start_time = time.time()

        try:
            # Extract parameters
            symbol = arguments.get("symbol")
            sources = arguments.get("sources", self.sentiment_sources)
            lookback_days = arguments.get("lookback_days", 7)
            include_news_items = arguments.get("include_news_items", False)

            if not symbol:
                return MCPToolResult(
                    status="ERROR",
                    error="No symbol provided",
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            if not self.stock_analyzer:
                return MCPToolResult(
                    status="ERROR",
                    error="Sentiment analysis not available",
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            # Get comprehensive sentiment analysis
            try:
                sentiment_data = self.stock_analyzer.get_comprehensive_sentiment(
                    symbol)

                # Extract relevant data
                overall_sentiment = {
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 0.0
                }

                # Calculate weighted sentiment
                if "weighted_aggregated" in sentiment_data:
                    weighted = sentiment_data["weighted_aggregated"]
                    total_weight = weighted.get("total_weight", 1.0)
                    if total_weight > 0:
                        overall_sentiment = {
                            "positive": weighted.get("positive", 0) / total_weight,
                            "negative": weighted.get("negative", 0) / total_weight,
                            "neutral": weighted.get("neutral", 0) / total_weight
                        }

                # Calculate confidence score
                confidence = 0.0
                if "comprehensive_analysis" in sentiment_data:
                    confidence = sentiment_data["comprehensive_analysis"].get(
                        "confidence_score", 0.0)

                # Prepare result
                result_data = {
                    "symbol": symbol,
                    "overall_sentiment": overall_sentiment,
                    "confidence": confidence,
                    "sentiment_breakdown": {},
                    "news_items_count": 0,
                    "sources_used": sources
                }

                # Add breakdown by source if available
                for source in ["newsapi", "gnews", "reddit", "google_news", "indian_news"]:
                    if source in sentiment_data:
                        result_data["sentiment_breakdown"][source] = sentiment_data[source]

                # Add news items if requested
                if include_news_items and "news_items" in sentiment_data:
                    # Limit to 10 items
                    result_data["news_items"] = sentiment_data["news_items"][:10]
                    result_data["news_items_count"] = len(
                        sentiment_data["news_items"])

                execution_time = time.time() - start_time

                return MCPToolResult(
                    status="SUCCESS",
                    data=result_data,
                    confidence=confidence,
                    execution_time=execution_time,
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            except Exception as e:
                logger.error(f"Error analyzing sentiment for {symbol}: {e}")
                return MCPToolResult(
                    status="ERROR",
                    error=f"Sentiment analysis failed: {str(e)}",
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat()
                    }
                )

        except Exception as e:
            logger.error(f"Error in sentiment tool: {e}")
            return MCPToolResult(
                status="ERROR",
                error=str(e),
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "timestamp": datetime.now().isoformat()
                }
            )

    def _get_market_sentiment_multiplier(self) -> float:
        """
        Get market sentiment multiplier based on overall market conditions
        """
        try:
            if not self.stock_analyzer:
                return 1.0

            # In a real implementation, this would analyze overall market sentiment
            # For now, we'll return a neutral multiplier
            return 1.0

        except Exception as e:
            logger.error(f"Error getting market sentiment multiplier: {e}")
            return 1.0


# Tool availability flag
SENTIMENT_TOOL_AVAILABLE = True
