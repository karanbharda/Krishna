#!/usr/bin/env python3
"""
Startup script for the Indian Stock Trading Bot Web Interface
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading
from pathlib import Path

def check_requirements():
    """Check if required files exist"""
    required_files = [
        'web_interface.html',
        'styles.css', 
        'app.js',
        'web_backend.py',
        'testindia.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    try:
        # Check if requirements file exists
        if os.path.exists('requirements_web.txt'):
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements_web.txt'
            ])
            print("✅ Dependencies installed successfully")
        else:
            print("⚠️  requirements_web.txt not found, skipping dependency installation")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("Please install dependencies manually:")
        print("pip install fastapi uvicorn pydantic numpy pandas yfinance requests python-dotenv")
        return False
    
    return True

def check_env_file():
    """Check if .env file exists and create template if not"""
    if not os.path.exists('.env'):
        print("📝 Creating .env template file...")
        
        env_template = """# Indian Stock Trading Bot Configuration
# Copy this file and rename to .env, then fill in your API keys

# Dhan API Credentials (for live trading)
DHAN_CLIENT_ID=your_dhan_client_id_here
DHAN_ACCESS_TOKEN=your_dhan_access_token_here

# News API Keys (optional)
NEWSAPI_KEY=your_newsapi_key_here
GNEWS_API_KEY=your_gnews_api_key_here

# Reddit API (optional)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=your_reddit_user_agent_here

# Grok API (for AI chat)
GROK_API_KEY=your_grok_api_key_here

# Risk Management Settings
STOP_LOSS_PCT=0.05
MAX_CAPITAL_PER_TRADE=0.25
MAX_TRADE_LIMIT=10
"""
        
        with open('.env.template', 'w') as f:
            f.write(env_template)
        
        print("✅ Created .env.template file")
        print("📋 Please copy .env.template to .env and configure your API keys")
    else:
        print("✅ .env file found")

def open_browser_delayed(url, delay=3):
    """Open browser after a delay"""
    time.sleep(delay)
    try:
        webbrowser.open(url)
        print(f"🌐 Opened browser at {url}")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print(f"Please open your browser and go to: {url}")

def run_web_interface():
    """Run the web interface"""
    print("🚀 Starting Indian Stock Trading Bot Web Interface...")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Cannot start web interface due to missing files")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Continuing without installing dependencies...")
    
    # Check environment file
    check_env_file()
    
    print("\n🔧 Configuration:")
    print("   - Mode: Paper Trading (default)")
    print("   - Host: 127.0.0.1")
    print("   - Port: 5000")
    print("   - URL: http://127.0.0.1:5000")
    
    # Start browser in background
    browser_thread = threading.Thread(
        target=open_browser_delayed, 
        args=("http://127.0.0.1:5000", 3)
    )
    browser_thread.daemon = True
    browser_thread.start()
    
    print("\n🌟 Starting FastAPI web server...")
    print("   Press Ctrl+C to stop the server")
    print("   API docs available at: http://127.0.0.1:5000/docs")
    print("=" * 60)

    try:
        # Import and run the web backend
        from backend.web_backend import run_web_server
        run_web_server(host='127.0.0.1', port=5000, debug=False)
        
    except ImportError as e:
        print(f"❌ Error importing web backend: {e}")
        print("Make sure web_backend.py and testindia.py are in the same directory")
        return False
        
    except KeyboardInterrupt:
        print("\n\n🛑 Web server stopped by user")
        print("Thank you for using the Indian Stock Trading Bot!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error starting web server: {e}")
        return False

def main():
    """Main function"""
    print("🇮🇳 Indian Stock Trading Bot - Web Interface")
    print("=" * 60)
    
    try:
        success = run_web_interface()
        if success:
            print("\n✅ Web interface stopped successfully")
        else:
            print("\n❌ Web interface encountered errors")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
