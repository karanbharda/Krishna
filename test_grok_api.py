#!/usr/bin/env python3
"""
Simple Grok API Test Script
==========================

Tests the Grok API key configuration and connectivity.
"""

import os
import sys
import json
import requests
from pathlib import Path


def test_grok_api_key():
    """Test the Grok API key from .env file"""

    # Load environment variables
    project_root = Path(__file__).parent
    env_file = project_root / ".env"

    if not env_file.exists():
        print("❌ .env file not found")
        return False

    # Read the .env file manually to get the API key
    grok_api_key = None
    with open(env_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                if 'GROK_API_KEY=' in line:
                    grok_api_key = line.split('=', 1)[1].strip()
                    break

    if not grok_api_key:
        print("❌ GROK_API_KEY not found in .env file")
        return False

    print(f"🔑 API Key Found: {grok_api_key[:10]}...{grok_api_key[-5:]}")
    print(f"📏 API Key Length: {len(grok_api_key)} characters")

    # Test the API key format
    if grok_api_key.startswith('sk-or-v1-'):
        print("⚠️  Warning: This appears to be an OpenRouter API key, not an X.AI Grok API key")
        print("   X.AI Grok API keys should start with 'xai-'")

    # Test API connectivity
    print("\n🌐 Testing API connectivity...")

    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json"
    }

    # Test models endpoint
    try:
        response = requests.get(
            "https://api.x.ai/v1/models",
            headers=headers,
            timeout=10
        )

        print(f"📊 Models API Response: {response.status_code}")

        if response.status_code == 200:
            print("✅ API Key is valid and working!")
            try:
                data = response.json()
                print(
                    f"📋 Available models: {[model['id'] for model in data.get('data', [])]}")
            except:
                print(
                    "📄 Response:", response.text[:200] + "..." if len(response.text) > 200 else response.text)
        elif response.status_code == 401:
            print("❌ Authentication failed - Invalid API key")
            print("📝 Response:", response.text)
        elif response.status_code == 403:
            print("❌ Permission denied - Check your account and credits")
            print("📝 Response:", response.text)
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            print("📝 Response:", response.text)

    except requests.exceptions.Timeout:
        print("❌ Request timeout - API might be unreachable")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - Check your internet connection")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🚀 Grok API Key Test Script")
    print("=" * 50)

    success = test_grok_api_key()

    print("\n" + "=" * 50)
    if success:
        print("🎉 Test completed")
    else:
        print("💥 Test failed")

    sys.exit(0 if success else 1)
