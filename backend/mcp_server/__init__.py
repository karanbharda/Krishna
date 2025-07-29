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

# Import main components with fallback handling
try:
    from .mcp_trading_server import MCPTradingServer, MCPToolResult, MCPToolStatus
    MCP_SERVER_AVAILABLE = True
    logger.info("MCP Trading Server components loaded successfully")
except ImportError as e:
    logger.warning(f"MCP Trading Server not available: {e}")
    MCP_SERVER_AVAILABLE = False

    # Fallback implementations
    class MCPTradingServer:
        def __init__(self, config=None):
            self.config = config or {}
            logger.warning("Using fallback MCP Trading Server")

        async def start(self):
            logger.info("Fallback MCP server started")
            return True

        async def stop(self):
            logger.info("Fallback MCP server stopped")
            return True

        def get_status(self):
            return {
                "status": "fallback",
                "mcp_available": False,
                "message": "Using fallback implementation"
            }

    class MCPToolResult:
        def __init__(self, status="SUCCESS", data=None, error=None, reasoning="", confidence=0.8):
            self.status = status
            self.data = data or {}
            self.error = error
            self.reasoning = reasoning
            self.confidence = confidence

    class MCPToolStatus:
        SUCCESS = "SUCCESS"
        ERROR = "ERROR"
        PARTIAL = "PARTIAL"

try:
    from .agents.trading_agent import TradingAgent
    TRADING_AGENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Trading Agent not available: {e}")
    TRADING_AGENT_AVAILABLE = False

    class TradingAgent:
        def __init__(self, config=None):
            self.config = config or {}
            logger.warning("Using fallback Trading Agent")

try:
    from .agents.explanation_agent import ExplanationAgent
    EXPLANATION_AGENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Explanation Agent not available: {e}")
    EXPLANATION_AGENT_AVAILABLE = False

    class ExplanationAgent:
        def __init__(self, config=None):
            self.config = config or {}
            logger.warning("Using fallback Explanation Agent")

__all__ = [
    "MCPTradingServer",
    "TradingAgent",
    "ExplanationAgent",
    "MCPToolResult",
    "MCPToolStatus",
    "MCP_SERVER_AVAILABLE"
]
