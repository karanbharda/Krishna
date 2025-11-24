#!/usr/bin/env python3
"""
Script to start the MCP Trading Server
"""

from dotenv import load_dotenv
import asyncio
from mcp_server.tools.confirm_tool import ConfirmTool
from mcp_server.tools.scan_all_tool import ScanAllTool
from mcp_server.tools.analyze_tool import AnalyzeTool
from mcp_server.tools.predict_tool import PredictTool
from mcp_server.mcp_trading_server import MCPTradingServer
import sys
import os

# Add project root to path (two levels up from backend)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env file
load_dotenv(os.path.join(project_root, '.env'))

# Check if Groq API key is available
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_ENABLED = bool(GROQ_API_KEY) and not GROQ_API_KEY.startswith(
    'gs_')  # Invalid keys start with 'gs'

# Now import with relative paths since we're in the backend directory


async def main():
    """Start the MCP Trading Server"""
    config = {
        "host": "localhost",
        "port": 8014,  # Changed from 8011 to 8014 to avoid port conflict
        "monitoring_port": 8007,
        "groq_api_key": GROQ_API_KEY,
        "groq_base_url": "https://api.groq.com/openai/v1",
        "groq_model": "llama-3.1-8b-instant"
    }

    server = MCPTradingServer(config)

    # Initialize and register tools
    predict_tool = PredictTool({
        "tool_id": "predict_tool",
        "lightgbm_enabled": True,
        "rl_model_type": "linucb"
    })

    analyze_tool = AnalyzeTool({
        "tool_id": "analyze_tool",
        "groq_enabled": GROQ_ENABLED,
        "groq_api_key": GROQ_API_KEY,
        "groq_base_url": "https://api.groq.com/openai/v1",
        "groq_model": "llama-3.1-8b-instant",
        "langgraph_enabled": True
    })

    scan_all_tool = ScanAllTool({
        "tool_id": "scan_all_tool",
        "rl_model_type": "linucb"
    })

    confirm_tool = ConfirmTool({
        "tool_id": "confirm_tool",
        "executor_enabled": True,
        "trading_mode": "paper",
        "max_position_size": 0.1,
        "risk_tolerance": 0.05
    })

    # Register tools
    server.register_tool(
        name="predict",
        function=predict_tool.predict,
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

    server.register_tool(
        name="analyze",
        function=analyze_tool.analyze,
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

    server.register_tool(
        name="scan_all",
        function=scan_all_tool.scan_all,
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

    server.register_tool(
        name="confirm",
        function=confirm_tool.confirm,
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

    if GROQ_ENABLED:
        print("Groq API integration enabled")
    else:
        print("Groq API integration disabled - set GROQ_API_KEY environment variable to enable")
    print(f"Starting MCP Trading Server on {config['host']}:{config['port']}")

    try:
        await server.start()
    except KeyboardInterrupt:
        print("Shutting down MCP Trading Server...")
        await server.shutdown()
    except Exception as e:
        print(f"Error starting MCP server: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
