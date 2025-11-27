#!/usr/bin/env python3
"""
MCP Professional Trading Tool
============================

Professional-grade trading tool for the Model Context Protocol server
that integrates institutional buy/sell logic with comprehensive risk management.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from ..mcp_trading_server import MCPToolResult

# Import professional trading components
try:
    from backend.core.professional_buy_integration import ProfessionalBuyIntegration
    from backend.core.professional_sell_integration import ProfessionalSellIntegration
    from backend.utils.ml_components.stock_analysis_complete import predict_stock_price
    PROFESSIONAL_TRADING_AVAILABLE = True
except ImportError as e:
    PROFESSIONAL_TRADING_AVAILABLE = False
    logging.warning(f"Professional trading components not available: {e}")

logger = logging.getLogger(__name__)


class ProfessionalTradingTool:
    """
    MCP Professional Trading Tool
    Integrates institutional-grade buy/sell logic with comprehensive risk management
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "professional_trading_tool")

        # Initialize professional trading components
        if PROFESSIONAL_TRADING_AVAILABLE:
            try:
                self.buy_integration = ProfessionalBuyIntegration(config)
                self.sell_integration = ProfessionalSellIntegration(config)
                logger.info(
                    f"Professional Trading Tool {self.tool_id} initialized successfully")
            except Exception as e:
                logger.error(
                    f"Failed to initialize professional trading components: {e}")
                self.buy_integration = None
                self.sell_integration = None
        else:
            self.buy_integration = None
            self.sell_integration = None
            logger.warning("Professional trading components not available")

        # Tool interconnections
        self.predict_tool = None
        self.analyze_tool = None
        self.risk_management_tool = None

        logger.info(f"Professional Trading Tool {self.tool_id} initialized")

    def connect_tools(self, tool_registry: Dict[str, Any]):
        """Connect to other tools for interconnection"""
        if "predict" in tool_registry:
            self.predict_tool = tool_registry["predict"]
        if "analyze" in tool_registry:
            self.analyze_tool = tool_registry["analyze"]
        if "risk_management" in tool_registry:
            self.risk_management_tool = tool_registry["risk_management"]
        logger.info(
            f"Professional Trading Tool {self.tool_id} connected to other tools")

    async def execute_trading_decision(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Execute a professional trading decision (buy/sell/hold)

        Args:
            arguments: Tool arguments containing trading parameters
            session_id: Session identifier

        Returns:
            MCPToolResult with trading decision
        """
        start_time = time.time()

        try:
            # Extract parameters
            symbol = arguments.get("symbol")
            action = arguments.get("action", "analyze").lower()
            portfolio_context = arguments.get("portfolio_context", {})
            analysis_depth = arguments.get("analysis_depth", "comprehensive")

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

            if not PROFESSIONAL_TRADING_AVAILABLE or not self.buy_integration or not self.sell_integration:
                return MCPToolResult(
                    status="ERROR",
                    error="Professional trading components not available",
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            # Get ML prediction for the symbol
            try:
                prediction = predict_stock_price(
                    symbol, horizon="day", verbose=True)
                current_price = prediction.get(
                    "current_price", 0.0) if prediction else 0.0
            except Exception as e:
                logger.warning(
                    f"Failed to get ML prediction for {symbol}: {e}")
                current_price = 0.0
                prediction = None

            # Prepare analysis data
            analysis_data = {
                "technical_indicators": {},
                "sentiment_analysis": {},
                "ml_analysis": prediction or {}
            }

            # Execute requested action
            if action == "buy":
                # Professional buy evaluation
                buy_result = self.buy_integration.evaluate_professional_buy(
                    ticker=symbol,
                    current_price=current_price,
                    analysis_data=analysis_data,
                    portfolio_context=portfolio_context
                )

                execution_time = time.time() - start_time

                return MCPToolResult(
                    status="SUCCESS" if buy_result.get(
                        "success", False) else "PARTIAL",
                    data={
                        "action": "buy",
                        "symbol": symbol,
                        "decision": buy_result,
                        "analysis_depth": analysis_depth
                    },
                    confidence=buy_result.get("confidence_score", 0.0),
                    execution_time=execution_time,
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "symbol": symbol,
                        "action": "buy",
                        "timestamp": datetime.now().isoformat()
                    }
                )

            elif action == "sell":
                # Professional sell evaluation
                sell_result = self.sell_integration.evaluate_professional_sell(
                    ticker=symbol,
                    current_price=current_price,
                    portfolio_holdings=portfolio_context.get("holdings", {}),
                    analysis_data=analysis_data
                )

                execution_time = time.time() - start_time

                return MCPToolResult(
                    status="SUCCESS" if sell_result.get(
                        "success", False) else "PARTIAL",
                    data={
                        "action": "sell",
                        "symbol": symbol,
                        "decision": sell_result,
                        "analysis_depth": analysis_depth
                    },
                    confidence=sell_result.get("confidence_score", 0.0),
                    execution_time=execution_time,
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "symbol": symbol,
                        "action": "sell",
                        "timestamp": datetime.now().isoformat()
                    }
                )

            else:
                # Comprehensive analysis (default)
                buy_result = self.buy_integration.evaluate_professional_buy(
                    ticker=symbol,
                    current_price=current_price,
                    analysis_data=analysis_data,
                    portfolio_context=portfolio_context
                )

                sell_result = self.sell_integration.evaluate_professional_sell(
                    ticker=symbol,
                    current_price=current_price,
                    portfolio_holdings=portfolio_context.get("holdings", {}),
                    analysis_data=analysis_data
                )

                execution_time = time.time() - start_time

                return MCPToolResult(
                    status="SUCCESS",
                    data={
                        "action": "analyze",
                        "symbol": symbol,
                        "buy_decision": buy_result,
                        "sell_decision": sell_result,
                        "current_price": current_price,
                        "ml_prediction": prediction,
                        "analysis_depth": analysis_depth
                    },
                    confidence=max(buy_result.get(
                        "confidence_score", 0.0), sell_result.get("confidence_score", 0.0)),
                    execution_time=execution_time,
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "symbol": symbol,
                        "action": "analyze",
                        "timestamp": datetime.now().isoformat()
                    }
                )

        except Exception as e:
            logger.error(
                f"Error in professional trading tool: {e}", exc_info=True)
            return MCPToolResult(
                status="ERROR",
                error=str(e),
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "timestamp": datetime.now().isoformat()
                }
            )

    async def get_trading_recommendation(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Get a professional trading recommendation with risk assessment

        Args:
            arguments: Tool arguments containing recommendation parameters
            session_id: Session identifier

        Returns:
            MCPToolResult with trading recommendation
        """
        start_time = time.time()

        try:
            # Extract parameters
            symbols = arguments.get("symbols", [])
            portfolio_context = arguments.get("portfolio_context", {})
            risk_profile = arguments.get("risk_profile", "MODERATE")
            market_outlook = arguments.get("market_outlook", "neutral")

            if not symbols:
                return MCPToolResult(
                    status="ERROR",
                    error="No symbols provided",
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            if not PROFESSIONAL_TRADING_AVAILABLE or not self.buy_integration or not self.sell_integration:
                return MCPToolResult(
                    status="ERROR",
                    error="Professional trading components not available",
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            # Analyze each symbol
            recommendations = []

            for symbol in symbols:
                try:
                    # Get ML prediction
                    prediction = predict_stock_price(
                        symbol, horizon="day", verbose=True)
                    current_price = prediction.get(
                        "current_price", 0.0) if prediction else 0.0

                    # Prepare analysis data
                    analysis_data = {
                        "technical_indicators": {},
                        "sentiment_analysis": {},
                        "ml_analysis": prediction or {}
                    }

                    # Get buy recommendation
                    buy_result = self.buy_integration.evaluate_professional_buy(
                        ticker=symbol,
                        current_price=current_price,
                        analysis_data=analysis_data,
                        portfolio_context=portfolio_context
                    )

                    # Get sell recommendation for existing positions
                    sell_result = self.sell_integration.evaluate_professional_sell(
                        ticker=symbol,
                        current_price=current_price,
                        portfolio_holdings=portfolio_context.get(
                            "holdings", {}),
                        analysis_data=analysis_data
                    )

                    # Create recommendation
                    recommendation = {
                        "symbol": symbol,
                        "current_price": current_price,
                        "buy_recommendation": buy_result,
                        "sell_recommendation": sell_result,
                        "ml_prediction": prediction,
                        "confidence": max(buy_result.get("confidence_score", 0.0), sell_result.get("confidence_score", 0.0))
                    }

                    recommendations.append(recommendation)

                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    recommendations.append({
                        "symbol": symbol,
                        "error": str(e),
                        "confidence": 0.0
                    })

            # Sort recommendations by confidence
            recommendations.sort(key=lambda x: x.get(
                "confidence", 0.0), reverse=True)

            execution_time = time.time() - start_time

            return MCPToolResult(
                status="SUCCESS",
                data={
                    "recommendations": recommendations,
                    "risk_profile": risk_profile,
                    "market_outlook": market_outlook,
                    "total_analyzed": len(recommendations)
                },
                confidence=0.9 if recommendations else 0.0,
                execution_time=execution_time,
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "symbols_count": len(symbols),
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            logger.error(
                f"Error in trading recommendation tool: {e}", exc_info=True)
            return MCPToolResult(
                status="ERROR",
                error=str(e),
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "timestamp": datetime.now().isoformat()
                }
            )


# Tool availability flag
PROFESSIONAL_TRADING_TOOL_AVAILABLE = PROFESSIONAL_TRADING_AVAILABLE
