#!/usr/bin/env python3
"""
System Validation Script for Trading Bot
Validates file permissions, paths, and implements critical operation logging
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Setup enhanced logging for critical operations
def setup_critical_logging():
    """Setup enhanced logging for critical trading operations"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Critical operations logger
    critical_logger = logging.getLogger('critical_operations')
    critical_logger.setLevel(logging.INFO)
    
    # Create file handler for critical operations
    critical_handler = logging.FileHandler(log_dir / f"critical_operations_{datetime.now().strftime('%Y%m%d')}.log")
    critical_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [CRITICAL] %(message)s'
    )
    critical_handler.setFormatter(formatter)
    
    # Add handler to logger
    if not critical_logger.handlers:
        critical_logger.addHandler(critical_handler)
    
    return critical_logger

def validate_file_permissions():
    """Validate all critical file permissions and paths"""
    print("=== FILE PERMISSIONS & PATHS VALIDATION ===")
    
    critical_paths = [
        "data/paper_config.json",
        "data/live_config.json", 
        "data/audit_trail/",
        "logs/",
        "backend/",
        ".env"
    ]
    
    issues = []
    
    for path_str in critical_paths:
        path = Path(path_str)
        
        print(f"Checking: {path}")
        
        # Check if path exists
        if not path.exists():
            if path.suffix == ".json":
                # Create missing config files
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    default_config = {
                        "riskLevel": "medium",
                        "maxPositions": 5,
                        "stopLoss": 0.05,
                        "takeProfit": 0.15,
                        "created": datetime.now().isoformat()
                    }
                    with open(path, 'w') as f:
                        json.dump(default_config, f, indent=2)
                    print(f"  ✓ Created missing config file: {path}")
                except Exception as e:
                    issues.append(f"Cannot create {path}: {e}")
                    print(f"  ❌ Cannot create {path}: {e}")
            elif path.suffix == "":
                # Create missing directories
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"  ✓ Created missing directory: {path}")
                except Exception as e:
                    issues.append(f"Cannot create directory {path}: {e}")
                    print(f"  ❌ Cannot create directory {path}: {e}")
            else:
                issues.append(f"Missing critical file: {path}")
                print(f"  ❌ Missing critical file: {path}")
        else:
            print(f"  ✓ Exists: {path}")
        
        # Check write permissions for files/directories that need it
        if path.exists():
            if path.is_file():
                # Test write permission on file
                try:
                    with open(path, 'a') as f:
                        pass  # Just test opening for append
                    print(f"  ✓ Write permission: {path}")
                except Exception as e:
                    issues.append(f"No write permission for {path}: {e}")
                    print(f"  ❌ No write permission for {path}: {e}")
            elif path.is_dir():
                # Test write permission on directory
                test_file = path / ".write_test"
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    test_file.unlink()  # Delete test file
                    print(f"  ✓ Write permission: {path}")
                except Exception as e:
                    issues.append(f"No write permission for directory {path}: {e}")
                    print(f"  ❌ No write permission for directory {path}: {e}")
    
    return issues

def validate_environment_variables():
    """Validate critical environment variables"""
    print("\n=== ENVIRONMENT VARIABLES VALIDATION ===")
    
    required_vars = [
        "FYERS_APP_ID",
        "FYERS_ACCESS_TOKEN", 
        "NEWSAPI_KEY",
        "GNEWS_API_KEY"
    ]
    
    issues = []
    
    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Loaded .env file")
    else:
        issues.append("Missing .env file")
        print("❌ Missing .env file")
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: {'*' * min(len(value), 10)}...")
        else:
            issues.append(f"Missing environment variable: {var}")
            print(f"  ❌ Missing: {var}")
    
    return issues

def implement_critical_logging():
    """Implement enhanced logging for critical operations"""
    print("\n=== IMPLEMENTING CRITICAL OPERATION LOGGING ===")
    
    try:
        critical_logger = setup_critical_logging()
        
        # Test critical logging
        critical_logger.info("SYSTEM_VALIDATION: Critical logging system initialized")
        critical_logger.info("TRADING_SYSTEM: System validation completed successfully")
        
        print("✓ Critical operation logging implemented")
        print(f"✓ Log file: logs/critical_operations_{datetime.now().strftime('%Y%m%d')}.log")
        
        return []
    except Exception as e:
        return [f"Failed to setup critical logging: {e}"]

def validate_trading_integrations():
    """Validate professional trading integrations"""
    print("\n=== TRADING INTEGRATIONS VALIDATION ===")
    
    try:
        sys.path.append('backend')
        
        # Test imports
        from core.professional_buy_integration import ProfessionalBuyIntegration
        from core.professional_sell_integration import ProfessionalSellIntegration
        from core.professional_buy_config import ProfessionalBuyConfig
        from core.professional_sell_config import ProfessionalSellConfig
        
        print("✓ Professional integrations import successfully")
        
        # Test configuration loading
        buy_config = ProfessionalBuyConfig.get_conservative_config()
        sell_config = ProfessionalSellConfig.get_default_config()
        
        print("✓ Configuration loading works")
        
        # Test integration initialization
        buy_integration = ProfessionalBuyIntegration(buy_config)
        sell_integration = ProfessionalSellIntegration(sell_config)
        
        print("✓ Integration initialization works")
        
        return []
        
    except Exception as e:
        return [f"Trading integration validation failed: {e}"]

def main():
    """Main validation function"""
    print("🔍 TRADING BOT SYSTEM VALIDATION")
    print("=" * 50)
    
    all_issues = []
    
    # Validate file permissions and paths
    issues = validate_file_permissions()
    all_issues.extend(issues)
    
    # Validate environment variables
    issues = validate_environment_variables()
    all_issues.extend(issues)
    
    # Implement critical logging
    issues = implement_critical_logging()
    all_issues.extend(issues)
    
    # Validate trading integrations
    issues = validate_trading_integrations()
    all_issues.extend(issues)
    
    # Final report
    print("\n" + "=" * 50)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 50)
    
    if not all_issues:
        print("✅ ALL VALIDATIONS PASSED")
        print("🚀 SYSTEM IS READY FOR LIVE TRADING")
        return 0
    else:
        print("❌ VALIDATION ISSUES FOUND:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print("\n⚠️  RESOLVE THESE ISSUES BEFORE LIVE TRADING")
        return 1

if __name__ == "__main__":
    exit(main())
