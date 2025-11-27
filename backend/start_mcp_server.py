#!/usr/bin/env python3
"""
Script to start the MCP Trading Server
"""

from dotenv import load_dotenv
import asyncio
from mcp_server.mcp_interface import MCPInterface, MCP_INTERFACE_AVAILABLE
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


async def main():
    """Start the MCP Trading Server using unified interface"""
    if not MCP_INTERFACE_AVAILABLE:
        print("MCP Interface not available")
        return 1

    config = {
        "host": "localhost",
        "port": 8014,  # Changed from 8011 to 8014 to avoid port conflict
        "monitoring_port": 8007,
        "groq_api_key": GROQ_API_KEY,
        "groq_base_url": "https://api.groq.com/openai/v1",
        "groq_model": "llama-3.1-8b-instant",
        "agents": {
            "trading": {
                "agent_id": "production_trading_agent",
                "risk_tolerance": 0.02,
                "max_positions": 5,
                "min_confidence": 0.7
            },
            "portfolio": {
                "agent_id": "production_portfolio_agent",
                "risk_tolerance": 0.5,
                "optimization_method": "mean_variance",
                "max_positions": 10
            },
            "regime": {
                "agent_id": "production_regime_agent",
                "lookback_period": 20,
                "confidence_threshold": 0.7
            },
            "sentiment": {
                "agent_id": "production_sentiment_agent",
                "min_confidence": 0.7,
                "sources": ["news", "social_media", "analyst_ratings"]
            },
            "explanation": {
                "agent_id": "production_explanation_agent"
            }
        },
        "tools": {
            "predict": {
                "tool_id": "predict_tool",
                "lightgbm_enabled": True,
                "rl_model_type": "linucb"
            },
            "analyze": {
                "tool_id": "analyze_tool",
                "groq_enabled": GROQ_ENABLED,
                "groq_api_key": GROQ_API_KEY,
                "groq_base_url": "https://api.groq.com/openai/v1",
                "groq_model": "llama-3.1-8b-instant",
                "langgraph_enabled": True
            },
            "scan_all": {
                "tool_id": "scan_all_tool",
                "rl_model_type": "linucb"
            },
            "confirm": {
                "tool_id": "confirm_tool",
                "executor_enabled": True,
                "trading_mode": "paper",
                "max_position_size": 0.1,
                "risk_tolerance": 0.05
            },
            "sentiment": {
                "tool_id": "sentiment_tool"
            },
            "professional_trading": {
                "tool_id": "professional_trading_tool"
            },
            "risk_management": {
                "tool_id": "risk_management_tool"
            },
            "technical_analysis": {
                "tool_id": "technical_analysis_tool"
            }
        }
    }

    try:
        # Initialize unified MCP interface
        mcp_interface = MCPInterface(config)
        await mcp_interface.initialize()

        if GROQ_ENABLED:
            print("Groq API integration enabled")
        else:
            print(
                "Groq API integration disabled - set GROQ_API_KEY environment variable to enable")
        print(
            f"Starting MCP Trading Server on {config['host']}:{config['port']}")

        # Start the server
        await mcp_interface.start_server()

    except KeyboardInterrupt:
        print("Shutting down MCP Trading Server...")
        if 'mcp_interface' in locals():
            await mcp_interface.shutdown()
    except Exception as e:
        print(f"Error starting MCP server: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
