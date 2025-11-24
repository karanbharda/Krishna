#!/usr/bin/env python3
"""
Production MCP Deployment Script
===============================

Automated deployment script for the production-grade MCP server integration.
Handles dependency installation, service setup, and health monitoring.
"""

import os
import sys
import subprocess
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCPDeploymentManager:
    """Production deployment manager for MCP integration"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / ".venv"
        self.requirements_file = self.project_root / "requirements.txt"

    def deploy(self):
        """Execute complete deployment pipeline"""
        logger.info("🚀 Starting MCP Production Deployment")

        try:
            # Step 1: Environment setup
            self.setup_environment()

            # Step 2: Install dependencies
            self.install_dependencies()

            # Step 3: Setup Groq API (if needed)
            self.setup_groq_api()

            # Step 4: Configure services
            self.configure_services()

            # Step 5: Run health checks
            self.run_health_checks()

            # Step 6: Start services
            self.start_services()

            logger.info("✅ MCP deployment completed successfully!")

        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            sys.exit(1)

    def setup_environment(self):
        """Setup Python virtual environment"""
        logger.info("🔧 Setting up Python environment...")

        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            raise RuntimeError("Python 3.8+ required")

        logger.info(
            f"✅ Python {python_version.major}.{python_version.minor} detected")

        # Create virtual environment if it doesn't exist
        if not self.venv_path.exists():
            logger.info("Creating virtual environment...")
            subprocess.run([
                sys.executable, "-m", "venv", str(self.venv_path)
            ], check=True)

        logger.info("✅ Virtual environment ready")

    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("📦 Installing dependencies...")

        # Get pip executable
        if os.name == 'nt':  # Windows
            pip_exe = self.venv_path / "Scripts" / "pip.exe"
            python_exe = self.venv_path / "Scripts" / "python.exe"
        else:  # Unix/Linux/macOS
            pip_exe = self.venv_path / "bin" / "pip"
            python_exe = self.venv_path / "bin" / "python"

        # Upgrade pip
        subprocess.run([
            str(python_exe), "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)

        # Install requirements
        if self.requirements_file.exists():
            subprocess.run([
                str(pip_exe), "install", "-r", str(self.requirements_file)
            ], check=True)
        else:
            logger.warning(
                "⚠️ requirements.txt not found, installing core dependencies")
            core_deps = [
                "fastapi>=0.104.0",
                "uvicorn[standard]>=0.24.0",
                "aiohttp>=3.9.0",
                "requests>=2.31.0",
                "pandas>=2.1.0",
                "numpy>=1.24.0",
                "scikit-learn>=1.3.0"
            ]
            subprocess.run([
                str(pip_exe), "install"
            ] + core_deps, check=True)

        # Install optional dependencies with error handling
        optional_deps = [
            ("fyers-apiv3", "Fyers API integration"),
            ("prometheus-client", "Monitoring"),
            ("psutil", "System monitoring"),
            ("groq", "Groq API integration")
        ]

        for dep, description in optional_deps:
            try:
                subprocess.run([
                    str(pip_exe), "install", dep
                ], check=True, capture_output=True)
                logger.info(f"✅ Installed {dep} ({description})")
            except subprocess.CalledProcessError:
                logger.warning(
                    f"⚠️ Failed to install {dep} - {description} may not work")

        logger.info("✅ Dependencies installed")

    def setup_groq_api(self):
        """Setup Groq API integration"""
        logger.info("🤖 Setting up Groq API...")

        # Check if Groq API key is configured
        from dotenv import load_dotenv
        import os
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        load_dotenv(os.path.join(project_root, '.env'))

        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            logger.warning(
                "⚠️ GROQ_API_KEY not found in environment variables")
            logger.info("Please set your Groq API key in the .env file")
        else:
            logger.info("✅ Groq API key configured")

    def configure_services(self):
        """Configure system services and environment"""
        logger.info("⚙️ Configuring services...")

        # Create data directories
        data_dirs = [
            "data",
            "logs",
            "mcp_server/logs",
            "reports"
        ]

        for dir_name in data_dirs:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.info(f"✅ Created directory: {dir_name}")

        # Create environment template if not exists
        env_file = self.project_root / ".env"
        if not env_file.exists():
            env_template = """# Trading Bot Environment Configuration
# Fyers API Credentials
FYERS_APP_ID=your_app_id_here
FYERS_ACCESS_TOKEN=your_access_token_here

# Dhan API Credentials (for live trading)
DHAN_CLIENT_ID=your_client_id_here
DHAN_ACCESS_TOKEN=your_access_token_here

# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.1-8b-instant

# MCP Server Configuration
MCP_MONITORING_PORT=8002
MCP_MAX_SESSIONS=100

# Trading Configuration
DEFAULT_RISK_LEVEL=MEDIUM
DEFAULT_STARTING_BALANCE=10000
"""
            with open(env_file, 'w') as f:
                f.write(env_template)
            logger.info(
                "✅ Created .env template - please configure your API keys")

        logger.info("✅ Services configured")

    def run_health_checks(self):
        """Run comprehensive health checks"""
        logger.info("🏥 Running health checks...")

        checks = [
            ("Python Environment", self.check_python_env),
            ("Dependencies", self.check_dependencies),
            ("Groq API", self.check_groq_api),
            ("API Credentials", self.check_credentials),
            ("File Permissions", self.check_permissions)
        ]

        for check_name, check_func in checks:
            try:
                result = check_func()
                if result:
                    logger.info(f"✅ {check_name}: OK")
                else:
                    logger.warning(f"⚠️ {check_name}: Issues detected")
            except Exception as e:
                logger.error(f"❌ {check_name}: {e}")

    def check_python_env(self) -> bool:
        """Check Python environment"""
        return self.venv_path.exists() and (self.venv_path / "pyvenv.cfg").exists()

    def check_dependencies(self) -> bool:
        """Check critical dependencies"""
        try:
            import fastapi
            import uvicorn
            import aiohttp
            import pandas
            import numpy
            return True
        except ImportError:
            return False

    def check_groq_api(self) -> bool:
        """Check Groq API service"""
        try:
            from dotenv import load_dotenv
            import os
            project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            load_dotenv(os.path.join(project_root, '.env'))

            groq_api_key = os.getenv('GROQ_API_KEY')
            if not groq_api_key:
                return False

            import requests
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                "https://api.groq.com/openai/v1/models", headers=headers, timeout=5)
            return response.status_code == 200
        except:
            return False

    def check_credentials(self) -> bool:
        """Check API credentials"""
        from dotenv import load_dotenv
        import os
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        load_dotenv(os.path.join(project_root, '.env'))

        required_vars = ["FYERS_APP_ID", "FYERS_ACCESS_TOKEN"]
        return all(os.getenv(var) for var in required_vars)

    def check_permissions(self) -> bool:
        """Check file permissions"""
        test_file = self.project_root / "data" / "test_write.tmp"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()
            return True
        except:
            return False

    def start_services(self):
        """Start MCP services"""
        logger.info("🚀 Starting services...")

        # Check Groq API key
        if not self.check_groq_api():
            logger.warning("⚠️ Groq API key not configured or invalid")
            logger.info("Please set your Groq API key in the .env file")

        # Start the trading bot with MCP integration
        logger.info("🤖 Starting Trading Bot with MCP integration...")
        logger.info("Run the following command to start the server:")
        logger.info(f"cd {self.project_root}")

        if os.name == 'nt':  # Windows
            logger.info(f"{self.venv_path}/Scripts/python.exe web_backend.py")
        else:  # Unix/Linux/macOS
            logger.info(f"{self.venv_path}/bin/python web_backend.py")

        logger.info(
            "🌐 Web interface will be available at: http://localhost:5000")
        logger.info("📊 MCP monitoring at: http://localhost:8002")
        logger.info("📚 API docs at: http://localhost:5000/docs")


def main():
    """Main deployment function"""
    print("🚀 MCP Production Deployment Manager")
    print("=" * 50)

    manager = MCPDeploymentManager()
    manager.deploy()

    print("\n" + "=" * 50)
    print("🎉 Deployment Complete!")
    print("\nNext Steps:")
    print("1. Configure your API keys in .env file")
    print("2. Configure your Groq API key in .env file")
    print("3. Start the trading bot: python web_backend.py")
    print("4. Open browser: http://localhost:5000")
    print("5. Test MCP integration: python test_mcp_integration.py")


if __name__ == "__main__":
    main()
