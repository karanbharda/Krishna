#!/usr/bin/env python3
"""
MCP Integration Test
====================
Tests if the MCP integration with Groq API is working correctly
"""

import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
backend_path = project_root / "backend"
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(project_root))


async def test_mcp_integration():
    """Test the MCP integration with Groq API"""

    try:
        print("🚀 Testing MCP Integration with Groq API")
        print("=" * 40)

        # Load environment variables
        from dotenv import load_dotenv
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            print("✅ Environment variables loaded")
        else:
            print("⚠️  .env file not found")
            return False

        # Get API key
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("❌ GROQ_API_KEY not found in environment variables")
            return False

        print(f"🔑 API Key Found: {api_key[:10]}...{api_key[-5:]}")

        # Test GroqAPIEngine methods
        print("\n📋 Testing GroqAPIEngine Methods")
        try:
            from groq_api import GroqAPIEngine, TradingContext

            config = {
                "groq_api_key": api_key,
                "groq_base_url": "https://api.groq.com/openai/v1",
                "groq_model": "llama-3.1-8b-instant"
            }

            # Test health check
            engine = GroqAPIEngine(config)
            health_status = await engine.health_check()
            print(f"   Health Status: {health_status}")

            if health_status.get("status") == "healthy":
                print("   ✅ Health check passed")

                # Test that all required methods exist
                required_methods = [
                    'get_trade_recommendation',
                    'analyze_market_decision',
                    'analyze_market_conditions',
                    'assess_risk',
                    'optimize_portfolio',
                    'get_general_trading_advice'
                ]

                missing_methods = []
                for method in required_methods:
                    if not hasattr(engine, method):
                        missing_methods.append(method)

                if missing_methods:
                    print(f"   ❌ Missing methods: {missing_methods}")
                    return False
                else:
                    print("   ✅ All required methods are present")
                    return True
            else:
                print("   ❌ Health check failed")
                return False

        except Exception as e:
            print(f"   ❌ GroqAPIEngine test failed: {e}")
            return False

    except Exception as e:
        print(f"\n💥 MCP Integration Test Failed: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_mcp_integration())

    print("\n" + "=" * 40)
    if success:
        print("🎉 MCP Integration Test Passed!")
        print("✅ Groq API integration with MCP is working correctly")
    else:
        print("💥 MCP Integration Test Failed!")

    sys.exit(0 if success else 1)
