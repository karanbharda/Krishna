#!/usr/bin/env python3
"""
Unified MCP Interface
=====================

Provides a unified interface for all MCP components to work together seamlessly
"""

import logging
from typing import Dict, Any, Optional, List
from .mcp_trading_server import MCPTradingServer
from .agents.trading_agent import TradingAgent
from .agents.portfolio_agent import PortfolioAgent
from .agents.market_regime_agent import MarketRegimeAgent
from .agents.sentiment_agent import SentimentAgent
from .agents.explanation_agent import ExplanationAgent
from .tools.predict_tool import PredictTool
from .tools.analyze_tool import AnalyzeTool
from .tools.scan_all_tool import ScanAllTool
from .tools.confirm_tool import ConfirmTool
from .tools.sentiment_tool import SentimentTool
from .tools.professional_trading_tool import ProfessionalTradingTool
from .tools.risk_management_tool import RiskManagementTool
from .tools.technical_analysis_tool import TechnicalAnalysisTool

logger = logging.getLogger(__name__)


class MCPInterface:
    """
    Unified MCP Interface
    Orchestrates all MCP components for seamless interconnection
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server: Optional[MCPTradingServer] = None
        self.trading_agent: Optional[TradingAgent] = None
        self.portfolio_agent: Optional[PortfolioAgent] = None
        self.regime_agent: Optional[MarketRegimeAgent] = None
        self.sentiment_agent: Optional[SentimentAgent] = None
        self.explanation_agent: Optional[ExplanationAgent] = None

        # Tools
        self.predict_tool: Optional[PredictTool] = None
        self.analyze_tool: Optional[AnalyzeTool] = None
        self.scan_all_tool: Optional[ScanAllTool] = None
        self.confirm_tool: Optional[ConfirmTool] = None
        self.sentiment_tool: Optional[SentimentTool] = None
        self.professional_trading_tool: Optional[ProfessionalTradingTool] = None
        self.risk_management_tool: Optional[RiskManagementTool] = None
        self.technical_analysis_tool: Optional[TechnicalAnalysisTool] = None

        # Tool registry for interconnection
        self.tool_registry: Dict[str, Any] = {}

        logger.info("MCP Interface initialized")

    async def initialize(self):
        """Initialize all MCP components"""
        try:
            # Initialize server
            server_config = {
                "host": self.config.get("host", "localhost"),
                "port": self.config.get("port", 8002),
                "monitoring_port": self.config.get("monitoring_port", 8003),
                "max_sessions": self.config.get("max_sessions", 100),
                "groq_api_key": self.config.get("groq_api_key", ""),
                "groq_base_url": self.config.get("groq_base_url", "https://api.groq.com/openai/v1"),
                "groq_model": self.config.get("groq_model", "llama-3.1-8b-instant")
            }
            self.server = MCPTradingServer(server_config)

            # Initialize agents
            agent_config = self.config.get("agents", {})

            self.trading_agent = TradingAgent(agent_config.get("trading", {
                "agent_id": "mcp_trading_agent",
                "risk_tolerance": 0.02,
                "max_positions": 5,
                "min_confidence": 0.7
            }))
            await self.trading_agent.initialize()

            self.portfolio_agent = PortfolioAgent(agent_config.get("portfolio", {
                "agent_id": "mcp_portfolio_agent",
                "risk_tolerance": 0.5,
                "optimization_method": "mean_variance",
                "max_positions": 10
            }))
            await self.portfolio_agent.initialize()

            self.regime_agent = MarketRegimeAgent(agent_config.get("regime", {
                "agent_id": "mcp_regime_agent",
                "lookback_period": 20,
                "confidence_threshold": 0.7
            }))
            await self.regime_agent.initialize()

            self.sentiment_agent = SentimentAgent(agent_config.get("sentiment", {
                "agent_id": "mcp_sentiment_agent",
                "min_confidence": 0.7,
                "sources": ["news", "social_media", "analyst_ratings"]
            }))
            await self.sentiment_agent.initialize()

            self.explanation_agent = ExplanationAgent(agent_config.get("explanation", {
                "agent_id": "mcp_explanation_agent"
            }))
            await self.explanation_agent.initialize()

            # Initialize tools
            tool_config = self.config.get("tools", {})

            self.predict_tool = PredictTool(tool_config.get("predict", {
                "tool_id": "mcp_predict_tool",
                "lightgbm_enabled": True,
                "rl_model_type": "linucb"
            }))

            self.analyze_tool = AnalyzeTool(tool_config.get("analyze", {
                "tool_id": "mcp_analyze_tool",
                "groq_enabled": True,
                "groq_api_key": self.config.get("groq_api_key", ""),
                "groq_base_url": "https://api.groq.com/openai/v1",
                "groq_model": "llama-3.1-8b-instant",
                "langgraph_enabled": True
            }))

            self.scan_all_tool = ScanAllTool(tool_config.get("scan_all", {
                "tool_id": "mcp_scan_all_tool",
                "rl_model_type": "linucb"
            }))

            self.confirm_tool = ConfirmTool(tool_config.get("confirm", {
                "tool_id": "mcp_confirm_tool",
                "executor_enabled": True,
                "trading_mode": "paper",
                "max_position_size": 0.1,
                "risk_tolerance": 0.05
            }))

            self.sentiment_tool = SentimentTool(tool_config.get("sentiment", {
                "tool_id": "mcp_sentiment_tool"
            }))

            self.professional_trading_tool = ProfessionalTradingTool(tool_config.get("professional_trading", {
                "tool_id": "mcp_professional_trading_tool"
            }))

            self.risk_management_tool = RiskManagementTool(tool_config.get("risk_management", {
                "tool_id": "mcp_risk_management_tool"
            }))

            self.technical_analysis_tool = TechnicalAnalysisTool(tool_config.get("technical_analysis", {
                "tool_id": "mcp_technical_analysis_tool"
            }))

            # Build tool registry
            self._build_tool_registry()

            # Connect all components
            self._connect_components()

            # Register tools with server
            self._register_tools()

            logger.info(
                "MCP Interface initialized successfully with all components interconnected")

        except Exception as e:
            logger.error(f"Error initializing MCP Interface: {e}")
            raise

    def _build_tool_registry(self):
        """Build tool registry for interconnection"""
        self.tool_registry = {
            "predict": self.predict_tool,
            "analyze": self.analyze_tool,
            "scan_all": self.scan_all_tool,
            "confirm": self.confirm_tool,
            "sentiment": self.sentiment_tool,
            "professional_trading": self.professional_trading_tool,
            "trading_recommendations": self.professional_trading_tool,
            "risk_management": self.risk_management_tool,
            "technical_analysis": self.technical_analysis_tool,
            "pattern_scan": self.technical_analysis_tool
        }

        # Add agents to registry
        self.tool_registry["trading_agent"] = self.trading_agent
        self.tool_registry["portfolio_agent"] = self.portfolio_agent
        self.tool_registry["regime_agent"] = self.regime_agent
        self.tool_registry["sentiment_agent"] = self.sentiment_agent
        self.tool_registry["explanation_agent"] = self.explanation_agent

    def _connect_components(self):
        """Connect all components for full interconnection"""
        # Connect tools to each other
        for tool_name, tool_instance in self.tool_registry.items():
            if hasattr(tool_instance, 'connect_tools'):
                tool_instance.connect_tools(self.tool_registry)

        # Connect agents to tools
        if self.trading_agent and hasattr(self.trading_agent, 'connect_tools'):
            self.trading_agent.connect_tools(self.tool_registry)

        if self.portfolio_agent and hasattr(self.portfolio_agent, 'connect_tools'):
            self.portfolio_agent.connect_tools(self.tool_registry)

        if self.regime_agent and hasattr(self.regime_agent, 'connect_tools'):
            self.regime_agent.connect_tools(self.tool_registry)

        if self.sentiment_agent and hasattr(self.sentiment_agent, 'connect_tools'):
            self.sentiment_agent.connect_tools(self.tool_registry)

        if self.explanation_agent and hasattr(self.explanation_agent, 'connect_tools'):
            self.explanation_agent.connect_tools(self.tool_registry)

        logger.info("All MCP components interconnected successfully")

    def _register_tools(self):
        """Register all tools with the server"""
        if not self.server:
            return

        # Register tools with instances for interconnection
        self.server.register_tool(
            name="predict",
            function=self.predict_tool.predict,
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
            },
            instance=self.predict_tool
        )

        self.server.register_tool(
            name="analyze",
            function=self.analyze_tool.analyze,
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
            },
            instance=self.analyze_tool
        )

        self.server.register_tool(
            name="scan_all",
            function=self.scan_all_tool.scan_all,
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
            },
            instance=self.scan_all_tool
        )

        self.server.register_tool(
            name="confirm",
            function=self.confirm_tool.confirm,
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
            },
            instance=self.confirm_tool
        )

        self.server.register_tool(
            name="sentiment",
            function=self.sentiment_tool.analyze_sentiment,
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
            },
            instance=self.sentiment_tool
        )

        self.server.register_tool(
            name="professional_trading",
            function=self.professional_trading_tool.execute_trading_decision,
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
            },
            instance=self.professional_trading_tool
        )

        self.server.register_tool(
            name="trading_recommendations",
            function=self.professional_trading_tool.get_trading_recommendation,
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
            },
            instance=self.professional_trading_tool
        )

        self.server.register_tool(
            name="risk_management",
            function=self.risk_management_tool.assess_position_risk,
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
            },
            instance=self.risk_management_tool
        )

        self.server.register_tool(
            name="technical_analysis",
            function=self.technical_analysis_tool.analyze_technical_indicators,
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
            },
            instance=self.technical_analysis_tool
        )

        self.server.register_tool(
            name="pattern_scan",
            function=self.technical_analysis_tool.scan_for_patterns,
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
            },
            instance=self.technical_analysis_tool
        )

    async def start_server(self):
        """Start the MCP server"""
        if self.server:
            await self.server.start()

    async def shutdown(self):
        """Shutdown all components"""
        if self.server:
            await self.server.shutdown()
        logger.info("MCP Interface shutdown completed")


# Interface availability flag
MCP_INTERFACE_AVAILABLE = True
