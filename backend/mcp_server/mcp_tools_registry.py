#!/usr/bin/env python3
"""
MCP Tools Registry
==================

Centralized registry for MCP tools with standardized schemas and initialization.
"""

import logging
from typing import Dict, Any, Callable
from .tools.predict_tool import PredictTool, PREDICT_TOOL_AVAILABLE
from .tools.analyze_tool import AnalyzeTool, ANALYZE_TOOL_AVAILABLE
from .tools.scan_all_tool import ScanAllTool, SCAN_ALL_TOOL_AVAILABLE
from .tools.confirm_tool import ConfirmTool, CONFIRM_TOOL_AVAILABLE
from .tools.sentiment_tool import SentimentTool, SENTIMENT_TOOL_AVAILABLE
from .tools.professional_trading_tool import ProfessionalTradingTool, PROFESSIONAL_TRADING_TOOL_AVAILABLE
from .tools.risk_management_tool import RiskManagementTool, RISK_MANAGEMENT_TOOL_AVAILABLE
from .tools.technical_analysis_tool import TechnicalAnalysisTool, TECHNICAL_ANALYSIS_TOOL_AVAILABLE

logger = logging.getLogger(__name__)


class MCPToolsRegistry:
    """
    Centralized registry for MCP tools
    Manages tool registration, initialization, and schema definition
    """

    def __init__(self):
        self.tools = {}
        self.tool_schemas = {}
        self.tool_instances = {}  # Store tool instances for interconnection
        self.initialized = False

        # Register standard tools
        self._register_standard_tools()

        logger.info("MCP Tools Registry initialized")

    def _register_standard_tools(self):
        """Register the standard MCP tools"""
        # Predict Tool
        if PREDICT_TOOL_AVAILABLE:
            self.register_tool(
                name="predict",
                tool_class=PredictTool,
                description="Generate predictions using LightGBM + RL (LinUCB/PPO-Lite)",
                schema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "model_type": {"type": "string"},
                        "horizon": {"type": "string"},
                        "include_explanations": {"type": "boolean"}
                    },
                    "required": []
                }
            )

        # Analyze Tool
        if ANALYZE_TOOL_AVAILABLE:
            self.register_tool(
                name="analyze",
                tool_class=AnalyzeTool,
                description="Send prediction output to Groq API to get reasoning and insights",
                schema={
                    "type": "object",
                    "properties": {
                        "predictions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string"},
                                    "prediction_score": {"type": "number"},
                                    "confidence": {"type": "number"},
                                    "model_type": {"type": "string"},
                                    "features": {"type": "object"}
                                }
                            }
                        },
                        "analysis_depth": {"type": "string"},
                        "include_risk_assessment": {"type": "boolean"}
                    },
                    "required": ["predictions"]
                }
            )

        # Scan All Tool
        if SCAN_ALL_TOOL_AVAILABLE:
            self.register_tool(
                name="scan_all",
                tool_class=ScanAllTool,
                description="Batch scan all cached stocks and rank by RL agent",
                schema={
                    "type": "object",
                    "properties": {
                        "min_score": {"type": "number"},
                        "max_results": {"type": "number"},
                        "sectors": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "market_caps": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "sort_by": {"type": "string"}
                    },
                    "required": []
                }
            )

        # Confirm Tool
        if CONFIRM_TOOL_AVAILABLE:
            self.register_tool(
                name="confirm",
                tool_class=ConfirmTool,
                description="Validate results with Executor via FastMCP and log confirmations",
                schema={
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string"},
                                    "action": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "analysis": {"type": "object"}
                                }
                            }
                        },
                        "portfolio_value": {"type": "number"},
                        "risk_check": {"type": "boolean"}
                    },
                    "required": ["actions"]
                }
            )

        # Sentiment Tool
        if SENTIMENT_TOOL_AVAILABLE:
            self.register_tool(
                name="sentiment",
                tool_class=SentimentTool,
                description="Analyze news sentiment for stock symbols with Indian market focus",
                schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "lookback_days": {"type": "number"},
                        "include_news_items": {"type": "boolean"}
                    },
                    "required": ["symbol"]
                }
            )

        # Professional Trading Tool
        if PROFESSIONAL_TRADING_TOOL_AVAILABLE:
            self.register_tool(
                name="professional_trading",
                tool_class=ProfessionalTradingTool,
                description="Execute professional-grade trading decisions with institutional buy/sell logic",
                schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "action": {"type": "string", "enum": ["buy", "sell", "analyze"]},
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "portfolio_context": {"type": "object"},
                        "risk_profile": {"type": "string"},
                        "market_outlook": {"type": "string"},
                        "analysis_depth": {"type": "string"}
                    },
                    "required": []
                }
            )

            # Trading Recommendations Tool
            self.register_tool(
                name="trading_recommendations",
                tool_class=ProfessionalTradingTool,
                description="Get professional trading recommendations with risk assessment for multiple symbols",
                schema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "portfolio_context": {"type": "object"},
                        "risk_profile": {"type": "string"},
                        "market_outlook": {"type": "string"}
                    },
                    "required": ["symbols"]
                }
            )

        # Risk Management Tool
        if RISK_MANAGEMENT_TOOL_AVAILABLE:
            self.register_tool(
                name="risk_management",
                tool_class=RiskManagementTool,
                description="Professional risk management with position sizing and portfolio risk assessment",
                schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "position_size": {"type": "number"},
                        "entry_price": {"type": "number"},
                        "stop_loss": {"type": "number"},
                        "portfolio_value": {"type": "number"},
                        "volatility": {"type": "number"},
                        "confidence": {"type": "number"},
                        "positions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string"},
                                    "value": {"type": "number"},
                                    "weight": {"type": "number"},
                                    "volatility": {"type": "number"}
                                }
                            }
                        },
                        "risk_free_rate": {"type": "number"},
                        "time_horizon": {"type": "number"}
                    },
                    "required": []
                }
            )

        # Technical Analysis Tool
        if TECHNICAL_ANALYSIS_TOOL_AVAILABLE:
            self.register_tool(
                name="technical_analysis",
                tool_class=TechnicalAnalysisTool,
                description="Professional technical analysis with advanced indicators and pattern recognition",
                schema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "timeframe": {"type": "string"},
                        "include_patterns": {"type": "boolean"},
                        "risk_profile": {"type": "string"}
                    },
                    "required": ["symbols"]
                }
            )

            # Pattern Scanning Tool
            self.register_tool(
                name="pattern_scan",
                tool_class=TechnicalAnalysisTool,
                description="Scan for technical patterns across multiple symbols",
                schema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "patterns": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "min_confidence": {"type": "number"},
                        "timeframe": {"type": "string"}
                    },
                    "required": ["symbols"]
                }
            )

    def register_tool(self, name: str, tool_class: type, description: str, schema: Dict[str, Any]):
        """
        Register a new MCP tool

        Args:
            name: Tool name
            tool_class: Tool class
            description: Tool description
            schema: JSON schema for tool parameters
        """
        self.tools[name] = tool_class
        self.tool_schemas[name] = {
            "description": description,
            "schema": schema
        }
        logger.info(f"Registered MCP tool: {name}")

    def get_tool(self, name: str, config: Dict[str, Any]):
        """
        Get an initialized tool instance

        Args:
            name: Tool name
            config: Tool configuration

        Returns:
            Initialized tool instance
        """
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found in registry")

        tool_class = self.tools[name]
        tool_instance = tool_class(config)

        # Store instance for interconnection
        self.tool_instances[name] = tool_instance
        return tool_instance

    def connect_tools(self):
        """Connect all tools to each other for full interconnection"""
        # Connect each tool to all other tools
        for tool_name, tool_instance in self.tool_instances.items():
            if hasattr(tool_instance, 'connect_tools'):
                tool_instance.connect_tools(self.tool_instances)
        logger.info("All MCP tools interconnected successfully")

    def get_tool_schema(self, name: str):
        """
        Get tool schema

        Args:
            name: Tool name

        Returns:
            Tool schema
        """
        return self.tool_schemas.get(name)

    def list_tools(self):
        """
        List all registered tools

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def get_tool_descriptions(self):
        """
        Get tool descriptions

        Returns:
            Dictionary of tool descriptions
        """
        return {name: info["description"] for name, info in self.tool_schemas.items()}


# Global registry instance
mcp_tools_registry = MCPToolsRegistry()

# Tool availability flag
MCP_TOOLS_REGISTRY_AVAILABLE = True
