import os
import sys
from pathlib import Path


async def test_groq_api():
    """Test the Groq API key and make a simple request"""

    # Add backend to path so we can import the groq_api module
    backend_path = Path(__file__).parent / "backend"
    sys.path.insert(0, str(backend_path))

    try:
        # Import the Groq API engine
        from groq_api import GroqAPIEngine

        # Get API key from environment
        api_key = os.getenv('GROQ_API_KEY')

        if not api_key:
            print("❌ GROQ_API_KEY not found in environment variables")
            print(
                "Please make sure the .env file is loaded or set the environment variable")
            return False

        print(f"🔑 API Key Found: {api_key[:10]}...{api_key[-5:]}")

        # Test with the currently supported model
        config = {
            "groq_api_key": api_key,
            "groq_base_url": "https://api.groq.com/openai/v1",
            "groq_model": "llama-3.1-8b-instant"  # This is the working model
        }

        # Test health check (this creates its own context)
        print("\n🏥 Testing health check...")
        groq_engine = GroqAPIEngine(config)
        health_status = await groq_engine.health_check()
        print(f"Health Status: {health_status}")

        if health_status.get("status") == "healthy":
            print("✅ Health check passed")

            # Test with the working model
            model = "llama-3.1-8b-instant"
            print(f"\n💬 Testing model: {model}")
            try:
                async with GroqAPIEngine(config) as groq_engine:
                    response = await groq_engine.get_general_trading_advice("What is the current market trend?")
                    print(f"Response: {response.content[:200]}...")
                    print(f"✅ Model {model} works!")
                    return True
            except Exception as e:
                print(f"❌ Model {model} failed: {e}")
                return False
        else:
            print("❌ Health check failed")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Simple Groq API Test")
    print("=" * 30)

    # Load environment variables
    from dotenv import load_dotenv
    project_root = Path(__file__).parent
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)
        print("✅ Environment variables loaded")
    else:
        print("⚠️  .env file not found")

    import asyncio
    success = asyncio.run(test_groq_api())

    print("\n" + "=" * 30)
    if success:
        print("🎉 All tests passed! Groq API is working correctly.")
    else:
        print("💥 Tests failed! Please check your API key and configuration.")

    sys.exit(0 if success else 1)
