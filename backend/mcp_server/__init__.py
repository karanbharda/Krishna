"""
Production-Grade MCP Server for Advanced Trading Bot
====================================================

This module implements a Model Context Protocol (MCP) server for the Indian Stock Trading Bot,
providing advanced AI-powered trading capabilities with real-time market data integration.

Features:
- Real-time market analysis with Fyers API
- Advanced reasoning with Llama models
- Modular tool-based architecture
- Production-grade error handling and monitoring
- Comprehensive logging and metrics
"""

__version__ = "1.0.0"
__author__ = "Trading Bot AI Team"

import logging

logger = logging.getLogger(__name__)

# Import main components - NO FALLBACKS
from .mcp_trading_server import MCPTradingServer, MCPToolResult, MCPToolStatus
MCP_SERVER_AVAILABLE = True
logger.info("MCP Trading Server components loaded successfully")

# Import agents - NO FALLBACKS
from .agents.trading_agent import TradingAgent
from .agents.explanation_agent import ExplanationAgent
from .agents.insight_agent import InsightAgent
TRADING_AGENT_AVAILABLE = True
EXPLANATION_AGENT_AVAILABLE = True
logger.info("All MCP agents loaded successfully")

__all__ = [
    "MCPTradingServer",
    "TradingAgent",
    "ExplanationAgent",
    "InsightAgent",
    "MCPToolResult",
    "MCPToolStatus",
    "MCP_SERVER_AVAILABLE"
]
