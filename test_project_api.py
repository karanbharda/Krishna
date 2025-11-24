#!/usr/bin/env python3
"""
Project API Test
================
Tests if the Groq API integration is working correctly within the project context
"""

import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
backend_path = project_root / "backend"
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(project_root))


def test_project_groq_integration():
    """Test the Groq API integration within the project context"""

    try:
        print("🚀 Testing Project Groq API Integration")
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

        # Test 1: Direct GroqAPIEngine usage
        print("\n📋 Test 1: Direct GroqAPIEngine Usage")
        try:
            # Import from backend directory
            sys.path.insert(0, str(backend_path))
            from groq_api import GroqAPIEngine

            config = {
                "groq_api_key": api_key,
                "groq_base_url": "https://api.groq.com/openai/v1",
                "groq_model": "llama-3.1-8b-instant"
            }

            import asyncio

            async def test_engine():
                # Test health check
                engine = GroqAPIEngine(config)
                health_status = await engine.health_check()
                print(f"   Health Status: {health_status}")

                if health_status.get("status") == "healthy":
                    # Test simple query within context manager
                    async with GroqAPIEngine(config) as groq_engine:
                        response = await groq_engine.get_general_trading_advice("What is the current market trend?")
                        print(f"   Response: {response.content[:100]}...")
                        return True
                return False

            success = asyncio.run(test_engine())
            if success:
                print("   ✅ Direct GroqAPIEngine test passed")
            else:
                print("   ❌ Direct GroqAPIEngine test failed")
                return False

        except Exception as e:
            print(f"   ❌ Direct GroqAPIEngine test failed: {e}")
            return False

        # Test 2: GroqLLM usage (from testindia.py)
        print("\n📋 Test 2: GroqLLM Usage")
        try:
            # Import from backend directory
            sys.path.insert(0, str(backend_path))
            from testindia import GroqLLM

            llm = GroqLLM(model_name="llama-3.1-8b-instant", api_key=api_key)

            # Test simple query
            response = llm._call("What is the current market trend?")
            print(f"   Response: {response[:100]}...")
            print("   ✅ GroqLLM test passed")

        except Exception as e:
            print(f"   ❌ GroqLLM test failed: {e}")
            return False

        # Test 3: Environment Manager
        print("\n📋 Test 3: Environment Manager")
        try:
            # Import from backend directory
            sys.path.insert(0, str(backend_path / "config"))
            from environment_manager import EnvironmentManager

            env_manager = EnvironmentManager()
            all_config = env_manager.get_all_config()

            if all_config.get("groq_api_key"):
                print("   ✅ Environment Manager test passed")
            else:
                print("   ❌ Environment Manager test failed: API key not found")
                return False

        except Exception as e:
            print(f"   ❌ Environment Manager test failed: {e}")
            return False

        print("\n" + "=" * 40)
        print("🎉 All Project API Tests Passed!")
        print("✅ Groq API integration is working correctly within the project")
        return True

    except Exception as e:
        print(f"\n💥 Project API Test Failed: {e}")
        return False


if __name__ == "__main__":
    success = test_project_groq_integration()
    sys.exit(0 if success else 1)
